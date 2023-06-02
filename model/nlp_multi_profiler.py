import torch
import torch.nn as nn
from utils import Optimizer, CPUThread, ProfilerMultiGPUModel
from itertools import chain
from collections import OrderedDict
from .nlp_single import NLP_Block, NLP_Predictor
# from transformer.encoder import TransformerEncoder

class NLP_MultiGPU(ProfilerMultiGPUModel):
    def __init__(self, configs):
        super(NLP_MultiGPU, self).__init__()
        self.configs = configs

        # Data
        self._init_data(configs)
        # Model
        self._init_model(configs)
        # Optimizers
        self._init_optimizers(configs)

    def _init_data(self, configs):
        # Data
        self.devices = configs["gpus"]
        self.dataset = configs["dataset"]
        self.train_loader = configs["train_loader"]
        self.test_loader = configs["test_loader"]
        self.proj_type = None
        self.pred_type = None
        self.num_layers = configs['layers']
        assert configs['layers'] >= 2, "Model layer setting error! The number of layers must be greater than 2."
        self.num_devices = set(self.devices)
        self.save_path = configs["save_path"] if configs["save_path"] != None else './{}'.format(self.__class__.__name__)

    def _init_model(self, configs):
        self.num_classes = configs["n_classes"]
        self.word_vec = configs["word_vec"]
        self.vocab_size = configs["vocab_size"]
        self.emb_dim = configs["emb_dim"]
        self.h_dim = configs["h_dim"]
    
    def _init_optimizers(self, configs):
        self.base_lr = configs["base_lr"]
        self.end_lr = configs["end_lr"]
        self.max_step = configs["max_steps"]
        self.global_steps = 0

    def inference(self, X, Y):
        pass

    def train_step(self, X, Y, multi_t=True):
        pass

    def forward(self, X, Y, multi_t=True):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,]) as prof:
            if self.training:
                _ = self.train_step(X, Y, multi_t)
            else:
                _ = self.inference(X, Y)
        torch.cuda.synchronize()
        prof.export_chrome_trace('{}_profile_{}.json'.format(self.save_path, ('train' if self.training else 'eval')))
        return _
    
    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask

class LSTM_BP_m_d(NLP_MultiGPU):
    def __init__(self, configs):
        super(LSTM_BP_m_d, self).__init__(configs)
        self.pred_type = None

    def _init_data(self, configs):
        super()._init_data(configs)

    def _init_model(self, configs):
        # Setting Model
        super()._init_model(configs)

        self.layer_cfg = dict()
        
        # Embedding
        self.layer_cfg[0] = {
            "inp_dim":self.vocab_size, "out_dim":self.emb_dim, "f":"emb", "h_dim":self.h_dim, "word_vec":self.word_vec, 
            "device":self.devices[0], "pred_type":self.pred_type, "num_classes":self.num_classes}

        # LSTM
        for i in range(self.num_layers-1):
            if i == 0:
                self.layer_cfg[i+1] = {
                    "inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"lstm", "h_dim":self.h_dim, "word_vec":None, 
                    "device":self.devices[i+1], "pred_type":self.pred_type, "num_classes":self.num_classes}
            else:
                self.layer_cfg[i+1] = {
                    "inp_dim":self.h_dim*2, "out_dim":self.h_dim, "f":"lstm", "h_dim":self.h_dim, "word_vec":None, 
                    "device":self.devices[i+1], "pred_type":self.pred_type, "num_classes":self.num_classes}
        
        # Predict
        self.layer_cfg[self.num_layers] = {
            "inp_dim":self.h_dim, "out_dim":self.num_classes, "hid_dim":self.h_dim, 
            "act_fun":nn.Tanh(), "device":self.devices[-1]}

        # Make Model
        self.model = []
        # Embedding and Encoder
        for i in range(self.num_layers):
            layer_cfg = self.layer_cfg[i]
            self.model.append(("backbone-"+str(i), NLP_Block(**layer_cfg).to(layer_cfg["device"])))
        # Predictor
        pred_cfg = self.layer_cfg[self.num_layers]
        self.model.append(("predictor", NLP_Predictor(**pred_cfg).to(pred_cfg["device"])))

        self.model = torch.nn.Sequential(OrderedDict(self.model))

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = list()

        self.opts.append(Optimizer(
            chain(self.model.parameters()), base_lr=self.base_lr, 
            end_lr=self.end_lr, max_step=self.max_step))

    def train_step(self, X, Y, multi_t=True):
        with torch.profiler.record_function("Init"):
            Xs = list()
            hidden =list()
            layer_fs = list()
            tasks = list()
            true_Ys = list()
            loss_all = 0
            gpu_losses = list()

        with torch.profiler.record_function("train mode"):
            for layer in self.model:
                layer.train()

        with torch.profiler.record_function("opt - zero_grad"):
            for opt in self.opts:
                opt.zero_grad()

        with torch.profiler.record_function("Y to gpu"):
            true_Ys.append(Y.to(self.devices[-1]))

        # Forward: Block 0 ~ num_layers
        ## Forward: Block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask

            Xs.append(hat_Y.to(self.devices[1]))
            hidden.append(None)
            layer_fs.append(hat_Y.mean(1))

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.to(self.devices[i+1]))
                hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))
                layer_fs.append((h[0] + h[1])/2)

        ## Forward: Block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask

            Xs.append(((h[0] + h[1])/2))
            layer_fs.append((h[0] + h[1])/2)

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)

        ## Loss,Backward,Update: Block -1
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(self.num_layers-1)):
            args = ([self.model[-1]], self.opts[-1].optimizer, [layer_fs[-1]], [true_Ys[-1]])
            if not multi_t:
                gpu_losses.all(self._loss_backward_update(*args)) # GPU -1
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        with torch.profiler.record_function("Wait"):
            if multi_t:
                for t in range(len(tasks)):
                    loss_all += tasks[t].get_result().item()
            else:
                for loss in gpu_losses:
                    loss_all += loss.item()
    
        return layer_fs[-1], loss_all
    
    def inference(self, X, Y):
        with torch.profiler.record_function("Init"):
            Xs = list()
            hidden =list()
            layer_fs = list()

        with torch.profiler.record_function("Y to gpu"):
            true_Y = Y.to(self.devices[-1])

        ## Forward: Block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask
            Xs.append(hat_Y.to(self.devices[1]))
            hidden.append(None)

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                Xs.append(hat_Y.to(self.devices[i+1]))
                hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))

        ## Forward: Block i
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            Xs.append(((h[0] + h[1])/2))

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)

        return [layer_fs[-1]], [true_Y]

class LSTM_BP_p_m_d(LSTM_BP_m_d):
    def __init__(self, configs):
        super(LSTM_BP_p_m_d, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.proj_type = None
        self.pred_type = configs['pred_type']
        assert self.pred_type not in [None, ''], "Setting error, pred_type is none or empty. pred_type: {}".format(self.pred_type)

    def train_step(self, X, Y, multi_t=True):
        with torch.profiler.record_function("Init"):
            Xs = list()
            hidden =list()
            layer_fs = list()
            tasks = list()
            true_Ys = list()
            loss_all = 0
            gpu_losses = list()

        with torch.profiler.record_function("train mode"):
            for layer in self.model:
                layer.train()

        with torch.profiler.record_function("opt - zero_grad"):
            for opt in self.opts:
                opt.zero_grad()

        with torch.profiler.record_function("Y to gpu"):
            for i in range(self.num_layers):
                true_Ys.append(Y.to(self.devices[i], non_blocking=True)) 

        # Forward: Block 0 ~ num_layers
        ## Forward: Block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask

            Xs.append(hat_Y.to(self.devices[1]))
            hidden.append(None)
            layer_fs.append(hat_Y.mean(1))

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.to(self.devices[i+1]))
                hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))
                layer_fs.append((h[0] + h[1])/2)

        ## Forward: Block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask

            Xs.append(((h[0] + h[1])/2))
            layer_fs.append((h[0] + h[1])/2)

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)

        ## Loss,Backward,Update: Block -1
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(self.num_layers-1)):
            args = (
                [model for model in self.model], 
                self.opts[-1].optimizer, 
                [layer_f for layer_f in layer_fs], 
                [true_Y for true_Y in true_Ys]+[true_Ys[-1]], True)
            if not multi_t:
                gpu_losses.all(self._loss_backward_update(*args)) # GPU -1
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        with torch.profiler.record_function("Wait"):
            if multi_t:
                for t in range(len(tasks)):
                    loss_all += tasks[t].get_result().item()
            else:
                for loss in gpu_losses:
                    loss_all += loss.item()
    
        return layer_fs[-1], loss_all
    
    def inference(self, X, Y):
        with torch.profiler.record_function("Init"):
            Xs = list()
            hidden =list()
            layer_fs = list()
            true_Ys = list()

        with torch.profiler.record_function("Y to gpu"):
            for i in range(self.num_layers):
                true_Ys.append(Y.to(self.devices[i], non_blocking=True)) 
            true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        ## Forward: Block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask
            layer_fs.append(self.model[0].loss(hat_Y.mean(1)))
            Xs.append(hat_Y.to(self.devices[1]))
            hidden.append(None)

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                layer_fs.append(self.model[i].loss((h[0] + h[1])/2))
                Xs.append(hat_Y.to(self.devices[i+1]))
                hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))

        ## Forward: Block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            layer_fs.append(self.model[-2].loss((h[0] + h[1])/2))
            Xs.append(((h[0] + h[1])/2))

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)
        
        return layer_fs, true_Ys

class LSTM_SCPL_m_d(NLP_MultiGPU):
    def __init__(self, configs):
        super(LSTM_SCPL_m_d, self).__init__(configs)

    def _init_data(self, configs, check_flag=True):
        super()._init_data(configs)
        self.proj_type = configs['proj_type']
        if check_flag:
            assert self.proj_type not in [None, ''], "Setting error, proj_type is none or empty. proj_type: {}".format(self.proj_type)
        self.pred_type = None
        self.temperature = configs['temperature']

    def _init_model(self, configs):
        # Setting Model
        super()._init_model(configs)

        self.layer_cfg = dict()

        # Embedding
        self.layer_cfg[0] = {
            "inp_dim":self.vocab_size, "out_dim":self.emb_dim, "f":"emb", "h_dim":self.h_dim, 
            "word_vec":self.word_vec, "temperature": self.temperature, "device":self.devices[0], 
            "proj_type":self.proj_type, "pred_type":self.pred_type, "num_classes":self.num_classes}

        # LSTM
        for i in range(self.num_layers-1):
            if i == 0:
                self.layer_cfg[i+1] = {
                    "inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"lstm", "h_dim":self.h_dim, "word_vec":None, 
                    "temperature": self.temperature, "device":self.devices[i+1], "proj_type":self.proj_type, 
                    "pred_type":self.pred_type, "num_classes":self.num_classes}
            else:
                self.layer_cfg[i+1] = {
                    "inp_dim":self.h_dim*2, "out_dim":self.h_dim, "f":"lstm", "h_dim":self.h_dim, "word_vec":None, 
                    "temperature": self.temperature,  "device":self.devices[i+1], "proj_type":self.proj_type, 
                    "pred_type":self.pred_type, "num_classes":self.num_classes}
        
        # Predict
        self.layer_cfg[self.num_layers] = {
            "inp_dim":self.h_dim, "out_dim":self.num_classes, "hid_dim":self.h_dim, 
            "act_fun":nn.Tanh(), "device":self.devices[-1]}

        # Make Model
        self.model = []
        # Embedding and Encoder
        for i in range(self.num_layers):
            layer_cfg = self.layer_cfg[i]
            self.model.append(("backbone-"+str(i), NLP_Block(**layer_cfg).to(layer_cfg["device"])))
        # Predictor
        pred_cfg = self.layer_cfg[self.num_layers]
        self.model.append(("predictor", NLP_Predictor(**pred_cfg).to(pred_cfg["device"])))

        self.model = torch.nn.Sequential(OrderedDict(self.model))
        
    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = list()

        self.opts.append(Optimizer(
            chain(self.model[0].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

        for i in range(1, self.num_layers-1):
            self.opts.append(Optimizer(
                chain(self.model[i].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

        self.opts.append(Optimizer(
            chain(self.model[-2].parameters(),self.model[-1].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

    def train_step(self, X, Y, multi_t=True):
        with torch.profiler.record_function("Init"):
            Xs = list()
            hidden =list()
            layer_fs = list()
            tasks = list()
            true_Ys = list()
            loss_all = 0
            gpu_losses = list()

        with torch.profiler.record_function("train mode"):
            for layer in self.model:
                layer.train()

        with torch.profiler.record_function("opt - zero_grad"):
            for opt in self.opts:
                opt.zero_grad()

        with torch.profiler.record_function("Y to gpu"):
            for i in range(self.num_layers):
                true_Ys.append(Y.to(self.devices[i], non_blocking=True)) 

        # Forward: block 0 ~ num_layers
        ## Forward: block 0
        with torch.profiler.record_function("Forward: Block {}".format(0)):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask

            Xs.append(hat_Y.detach().to(self.devices[1]))
            hidden.append(None)
            layer_fs.append(hat_Y.mean(1))

        ## Loss,Backward,Update: block 0
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(0)):
            args = ([self.model[0]], self.opts[0].optimizer, [layer_fs[-1]], [true_Ys[0]])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # block 0
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
        
        for i in range(1, self.num_layers-1):
            ## Forward: block i
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.detach().to(self.devices[i+1]))
                hidden.append(((h.detach().to(self.devices[i+1]), c.detach().to(self.devices[i+1]))))
                layer_fs.append((h[0] + h[1])/2)

            ## Loss,Backward,Update: block i
            with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(i)):
                args = ([self.model[i]], self.opts[i].optimizer, [layer_fs[-1]], [true_Ys[i]])
                if not multi_t:
                    gpu_losses.append(self._loss_backward_update(*args)) # block i
                else:
                    tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                    tasks[-1].start()

        ## Forward: block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask

            Xs.append(((h[0] + h[1])/2).detach())
            layer_fs.append((h[0] + h[1])/2)

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)

        ## Loss,Backward,Update: block -1
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(self.num_layers-1)):
            args = ([self.model[-2], self.model[-1]], self.opts[-1].optimizer, [layer_fs[-2], layer_fs[-1]], [true_Ys[-1]]*2)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # block -1
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
        
        with torch.profiler.record_function("Wait"):
            # Computing all the losses will take a lot of time
            # Because the function ".item()" takes a long time
            if multi_t:
                for t in range(len(tasks)):
                    loss_all += tasks[t].get_result().item()
            else:
                for loss in gpu_losses:
                    loss_all += loss.item()

        return layer_fs[-1], loss_all
    
    def inference(self, X, Y):
        with torch.profiler.record_function("Init"):
            Xs = list()
            hidden =list()
            layer_fs = list()
            true_Ys = list()

        with torch.profiler.record_function("Y to gpu"):
            true_Ys.append(Y.to(self.devices[-1], non_blocking=True)) 

        ## Forward: Block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask
            Xs.append(hat_Y.to(self.devices[1]))
            hidden.append(None)

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                Xs.append(hat_Y.to(self.devices[i+1]))
                hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))

        ## Forward: Block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            Xs.append(((h[0] + h[1])/2))

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)
        
        return layer_fs, true_Ys

class LSTM_DASCPL_m_d(LSTM_SCPL_m_d):
    def __init__(self, configs):
        super(LSTM_DASCPL_m_d, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.proj_type = configs['proj_type']
        assert self.proj_type not in [None, ''], "Setting error, proj_type is none or empty. proj_type: {}".format(self.proj_type)
        self.pred_type = configs['pred_type']
        assert self.pred_type not in [None, ''], "Setting error, pred_type is none or empty. pred_type: {}".format(self.pred_type)
        self.temperature = configs['temperature']
    
    def inference(self, X, Y):
        with torch.profiler.record_function("Init"):
            Xs = list()
            hidden =list()
            layer_fs = list()
            true_Ys = list()

        with torch.profiler.record_function("Y to gpu"):
            for i in range(self.num_layers):
                true_Ys.append(Y.to(self.devices[i], non_blocking=True)) 
            true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        ## Forward: Block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask
            layer_fs.append(self.model[0].loss(hat_Y.mean(1)))
            Xs.append(hat_Y.to(self.devices[1]))
            hidden.append(None)

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                layer_fs.append(self.model[i].loss((h[0] + h[1])/2))
                Xs.append(hat_Y.to(self.devices[i+1]))
                hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))

        ## Forward: Block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            layer_fs.append(self.model[-2].loss((h[0] + h[1])/2))
            Xs.append(((h[0] + h[1])/2))

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)
        
        return layer_fs, true_Ys

class LSTM_EE_m_d(LSTM_SCPL_m_d):
    def __init__(self, configs):
        super(LSTM_EE_m_d, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs, check_flag=False)
        self.proj_type = None
        self.pred_type = configs['pred_type']
        assert self.pred_type not in [None, ''], "Setting error, pred_type is none or empty. pred_type: {}".format(self.pred_type)
        self.pred_type  = self.pred_type  + ",non-detach"

class Trans_BP_m_d(NLP_MultiGPU):
    def __init__(self, configs):
        super(Trans_BP_m_d, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)

    def _init_model(self, configs):
        # Setting Model
        super()._init_model(configs)
        self.n_heads = configs["head"]

        self.layer_cfg = dict()

        # Embedding
        self.layer_cfg[0] = {
            "inp_dim":self.vocab_size, "out_dim":self.emb_dim, "f":"emb", "h_dim":self.emb_dim, "num_classes":self.num_classes, 
            "word_vec":self.word_vec, "device":self.devices[0], "pred_type":self.pred_type}

        # Transformer
        for i in range(1, self.num_layers):
            self.layer_cfg[i] = {
                "inp_dim":self.emb_dim, "out_dim":self.h_dim, 
                "f":"trans", "h_dim":self.emb_dim, "n_heads":self.n_heads, "num_classes":self.num_classes, 
                "word_vec":None, "device":self.devices[i], "pred_type":self.pred_type}

        # Predict
        self.layer_cfg[self.num_layers] = {
            "inp_dim":self.emb_dim, "out_dim":self.num_classes, 
            "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.devices[-1]}

        # Make Model
        self.model = []
        # Embedding and Encoder
        for i in range(self.num_layers):
            layer_cfg = self.layer_cfg[i]
            self.model.append(("backbone-"+str(i), NLP_Block(**layer_cfg).to(layer_cfg["device"])))
        # Predictor
        pred_cfg = self.layer_cfg[self.num_layers]
        self.model.append(("predictor", NLP_Predictor(**pred_cfg).to(pred_cfg["device"])))

        self.model = torch.nn.Sequential(OrderedDict(self.model))

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = list()

        self.opts.append(Optimizer(
            chain(self.model.parameters()), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

    def train_step(self, X, Y, multi_t=True):
        with torch.profiler.record_function("Init"):
            mask = self.get_mask(X)
            Xs = list()
            masks = list()
            hidden =list()
            layer_fs = list()
            tasks = list()
            true_Ys = list()
            loss_all = 0
            gpu_losses = list()

        with torch.profiler.record_function("train mode"):
            for layer in self.model:
                layer.train()

        with torch.profiler.record_function("opt - zero_grad"):
            for opt in self.opts:
                opt.zero_grad()

        with torch.profiler.record_function("Y to gpu"):
            true_Ys.append(Y.to(self.devices[-1]))

        # Forward: block 0 ~ num_layers
        ## Forward: block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            masks.append(mask.to(self.devices[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

            Xs.append(hat_Y.to(self.devices[1]))
            masks.append(mask.to(self.devices[1]))

        ## Forward: block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.to(self.devices[i+1]))
                masks.append(mask.to(self.devices[i+1]))

        ## Forward: block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y)
            masks.append(mask)
            layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

            hat_Y = self.model[-1](layer_fs[-1])
            layer_fs.append(hat_Y)

        ## Loss,Backward,Update: block -1:
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(self.num_layers-1)):
            args = ([self.model[-1]], self.opts[-1].optimizer, [layer_fs[-1]], [true_Ys[-1]])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # block -1
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
        
        with torch.profiler.record_function("Wait"):
            if multi_t:
                for t in range(len(tasks)):
                    loss_all += tasks[t].get_result().item()
            else:
                for loss in gpu_losses:
                    loss_all += loss.item()
    
        return layer_fs[-1], loss_all

    def inference(self, X, Y):
        with torch.profiler.record_function("Init"):
            mask = self.get_mask(X)
            Xs = list()
            masks =list()
            layer_fs = list()

        with torch.profiler.record_function("Y to gpu"):
            true_Y = Y.to(self.devices[-1])

        # Forward: block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            masks.append(mask.to(self.devices[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            Xs.append(hat_Y.to(self.devices[1]))
            masks.append(mask.to(self.devices[1]))

        # Forward: block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.to(self.devices[i+1]))
                masks.append(mask.to(self.devices[i+1]))

        # Forward: block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

            Xs.append(hat_Y)
            masks.append(mask)
            layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

            hat_Y = self.model[-1](layer_fs[-1])
            layer_fs.append(hat_Y)

        return [layer_fs[-1]], [true_Y]

class Trans_BP_p_m_d(Trans_BP_m_d):
    def __init__(self, configs):
        super(Trans_BP_p_m_d, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.proj_type = None
        self.pred_type = configs['pred_type']
        assert self.pred_type not in [None, ''], "Setting error, pred_type is none or empty. pred_type: {}".format(self.pred_type)

    def train_step(self, X, Y, multi_t=True):
        with torch.profiler.record_function("Init"):
            mask = self.get_mask(X)
            Xs = list()
            masks = list()
            hidden =list()
            layer_fs = list()
            tasks = list()
            true_Ys = list()
            loss_all = 0
            gpu_losses = list()

        with torch.profiler.record_function("train mode"):
            for layer in self.model:
                layer.train()

        with torch.profiler.record_function("opt - zero_grad"):
            for opt in self.opts:
                opt.zero_grad()

        with torch.profiler.record_function("Y to gpu"):
            for i in range(self.num_layers):
                true_Ys.append(Y.to(self.devices[i], non_blocking=True))

        # Forward: block 0 ~ layer_num
        ## Forward: block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            masks.append(mask.to(self.devices[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

            Xs.append(hat_Y.to(self.devices[1]))
            masks.append(mask.to(self.devices[1]))
            layer_fs.append(self.model[0].reduction(hat_Y, hidden, mask))

        ## Forward: block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.to(self.devices[i+1]))
                masks.append(mask.to(self.devices[i+1]))
                layer_fs.append(self.model[i].reduction(hat_Y, hidden, mask))

        ## Forward: block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y)
            masks.append(mask)
            layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

            hat_Y = self.model[-1](layer_fs[-1])
            layer_fs.append(hat_Y)

        ## Loss,Backward,Update: block -1:
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(self.num_layers-1)):
            args = (
                [model for model in self.model], 
                self.opts[-1].optimizer, 
                [layer_f for layer_f in layer_fs], 
                [true_Y for true_Y in true_Ys]+[true_Ys[-1]], True)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # block -1
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
        
        with torch.profiler.record_function("Wait"):
            if multi_t:
                for t in range(len(tasks)):
                    loss_all += tasks[t].get_result().item()
            else:
                for loss in gpu_losses:
                    loss_all += loss.item()
    
        return layer_fs[-1], loss_all

    def inference(self, X, Y):
        with torch.profiler.record_function("Init"):
            mask = self.get_mask(X)
            Xs = list()
            masks =list()
            layer_fs = list()
            true_Ys = list()

        with torch.profiler.record_function("Y to gpu"):
            for i in range(self.num_layers):
                true_Ys.append(Y.to(self.devices[i], non_blocking=True))
            true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        # Forward: block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            masks.append(mask.to(self.devices[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mas
            layer_fs.append(self.model[0].loss(self.model[0].reduction(hat_Y, hidden, mask)))
            Xs.append(hat_Y.to(self.devices[1]))
            masks.append(mask.to(self.devices[1]))

        # Forward: block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                layer_fs.append(self.model[i].loss(self.model[i].reduction(hat_Y, hidden, mask)))
                Xs.append(hat_Y.to(self.devices[i+1]))
                masks.append(mask.to(self.devices[i+1]))

        # Forward: block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            layer_fs.append(self.model[-2].loss(self.model[-2].reduction(hat_Y, hidden, mask)))

            Xs.append(hat_Y)
            masks.append(mask)
            last_layer_fs = self.model[-2].reduction(hat_Y, hidden, mask)

            hat_Y = self.model[-1](last_layer_fs)
            layer_fs.append(hat_Y)

        return layer_fs, true_Ys

class Trans_SCPL_m_d(NLP_MultiGPU):
    def __init__(self, configs):
        super(Trans_SCPL_m_d, self).__init__(configs)

    def _init_data(self, configs, check_flag=True):
        super()._init_data(configs)
        self.proj_type = configs['proj_type']
        if check_flag:
            assert self.proj_type not in [None, ''], "Setting error, proj_type is none or empty. proj_type: {}".format(self.proj_type)
        self.pred_type = None
        self.temperature = configs['temperature']

    def _init_model(self, configs):
        # Setting Model
        super()._init_model(configs)
        self.n_heads = configs["head"]

        self.layer_cfg = dict()

        # Embedding
        self.layer_cfg[0] = {
            "inp_dim":self.vocab_size, "out_dim":self.emb_dim, "f":"emb", "h_dim":self.emb_dim, 
            "num_classes":self.num_classes, "temperature": self.temperature, "word_vec":self.word_vec, 
            "device":self.devices[0], "proj_type":self.proj_type, "pred_type":self.pred_type}

        # Transformer
        for i in range(1, self.num_layers):
            self.layer_cfg[i] = {
                "inp_dim":self.emb_dim, "out_dim":self.h_dim, 
                "f":"trans", "h_dim":self.emb_dim, "n_heads":self.n_heads, "num_classes":self.num_classes, 
                "temperature": self.temperature, "word_vec":None, 
                "device":self.devices[i], "proj_type":self.proj_type, "pred_type":self.pred_type}

        # Predict
        self.layer_cfg[self.num_layers] = {
            "inp_dim":self.emb_dim, "out_dim":self.num_classes, 
            "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.devices[-1]}

        # Make Model
        self.model = []
        # Embedding and Encoder
        for i in range(self.num_layers):
            layer_cfg = self.layer_cfg[i]
            self.model.append(("backbone-"+str(i), NLP_Block(**layer_cfg).to(layer_cfg["device"])))
        # Predictor
        pred_cfg = self.layer_cfg[self.num_layers]
        self.model.append(("predictor", NLP_Predictor(**pred_cfg).to(pred_cfg["device"])))

        self.model = torch.nn.Sequential(OrderedDict(self.model))

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = list()

        self.opts.append(Optimizer(
            chain(self.model[0].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

        for i in range(1, self.num_layers-1):
            self.opts.append(Optimizer(
                chain(self.model[i].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

        self.opts.append(Optimizer(
            chain(self.model[-2].parameters(),self.model[-1].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

    def train_step(self, X, Y, multi_t=True):
        with torch.profiler.record_function("Init"):
            mask = self.get_mask(X)
            Xs = list()
            masks = list()
            hidden =list()
            layer_fs = list()
            tasks = list()
            true_Ys = list()
            loss_all = 0
            gpu_losses = list()

        with torch.profiler.record_function("train mode"):
            for layer in self.model:
                layer.train()

        with torch.profiler.record_function("opt - zero_grad"):
            for opt in self.opts:
                opt.zero_grad()

        with torch.profiler.record_function("Y to gpu"):
            for i in range(self.num_layers):
                true_Ys.append(Y.to(self.devices[i], non_blocking=True)) 

        # Forward: Block 0 ~ layer_num
        ## Forward: Block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            masks.append(mask.to(self.devices[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

            Xs.append(hat_Y.detach().to(self.devices[1]))
            masks.append(mask.to(self.devices[1]).detach())
            layer_fs.append(self.model[0].reduction(hat_Y, hidden, mask))

        ## Loss,Backward,Update: Block 0
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(0)):
            args = ([self.model[0]], self.opts[0].optimizer, [layer_fs[-1]], [true_Ys[0]])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 0
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                # print("gpu ", 0, tasks[-1].name)
                tasks[-1].start()

        for i in range(1, self.num_layers-1):
            ## Forward: Block i
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.detach().to(self.devices[i+1]))
                masks.append(mask.to(self.devices[i+1]).detach())
                layer_fs.append(self.model[i].reduction(hat_Y, hidden, mask))

            ## Loss,Backward,Update: Block i
            with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(i)):
                args = ([self.model[i]], self.opts[i].optimizer, [layer_fs[-1]], [true_Ys[i]])
                if not multi_t:
                    gpu_losses.append(self._loss_backward_update(*args)) # Block i
                else:
                    tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                    tasks[-1].start()

        ## Forward: Block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y.detach())
            masks.append(mask.detach())
            layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

            hat_Y = self.model[-1](layer_fs[-1].detach())
            layer_fs.append(hat_Y)

        ## Loss,Backward,Update: Block -1
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(i)):
            args = ([self.model[-2], self.model[-1]], self.opts[-1].optimizer, [layer_fs[-2], layer_fs[-1]], [true_Ys[-1]]*2)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block -1
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
    
        with torch.profiler.record_function("Wait"):
            # Computing all the losses will take a lot of time
            # Because the function ".item()" takes a long time
            if multi_t:
                for t in range(len(tasks)):
                    loss_all += tasks[t].get_result().item()
            else:
                for loss in gpu_losses:
                    loss_all += loss.item()

        return layer_fs[-1], loss_all

    def inference(self, X, Y):
        with torch.profiler.record_function("Init"):
            mask = self.get_mask(X)
            Xs = list()
            masks =list()
            layer_fs = list()
            true_Ys = list()

        with torch.profiler.record_function("Y to gpu"):
            true_Ys.append(Y.to(self.devices[-1], non_blocking=True)) 

        # Forward: block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            masks.append(mask.to(self.devices[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mas
            Xs.append(hat_Y.to(self.devices[1]))
            masks.append(mask.to(self.devices[1]))

        # Forward: block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                Xs.append(hat_Y.to(self.devices[i+1]))
                masks.append(mask.to(self.devices[i+1]))

        # Forward: block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

            Xs.append(hat_Y)
            masks.append(mask)
            last_layer_fs = self.model[-2].reduction(hat_Y, hidden, mask)

            hat_Y = self.model[-1](last_layer_fs)
            layer_fs.append(hat_Y)

        return layer_fs, true_Ys

class Trans_DASCPL_m_d(Trans_SCPL_m_d):
    def __init__(self, configs):
        super(Trans_DASCPL_m_d, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.proj_type = configs['proj_type']
        assert self.proj_type not in [None, ''], "Setting error, proj_type is none or empty. proj_type: {}".format(self.proj_type)
        self.pred_type = configs['pred_type']
        assert self.pred_type not in [None, ''], "Setting error, pred_type is none or empty. pred_type: {}".format(self.pred_type)
        self.temperature = configs['temperature']

    def inference(self, X, Y):
        with torch.profiler.record_function("Init"):
            mask = self.get_mask(X)
            Xs = list()
            masks =list()
            layer_fs = list()
            true_Ys = list()

        with torch.profiler.record_function("Y to gpu"):
            for i in range(self.num_layers):
                true_Ys.append(Y.to(self.devices[i], non_blocking=True))
            true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        # Forward: block 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.devices[0], non_blocking=True))
            masks.append(mask.to(self.devices[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mas
            layer_fs.append(self.model[0].loss(self.model[0].reduction(hat_Y, hidden, mask)))
            Xs.append(hat_Y.to(self.devices[1]))
            masks.append(mask.to(self.devices[1]))

        # Forward: block i
        for i in range(1, self.num_layers-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                layer_fs.append(self.model[i].loss(self.model[i].reduction(hat_Y, hidden, mask)))
                Xs.append(hat_Y.to(self.devices[i+1]))
                masks.append(mask.to(self.devices[i+1]))

        # Forward: block -1
        with torch.profiler.record_function("Forward: Block {}".format(self.num_layers-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            layer_fs.append(self.model[-2].loss(self.model[-2].reduction(hat_Y, hidden, mask)))

            Xs.append(hat_Y)
            masks.append(mask)
            last_layer_fs = self.model[-2].reduction(hat_Y, hidden, mask)

            hat_Y = self.model[-1](last_layer_fs)
            layer_fs.append(hat_Y)

        return layer_fs, true_Ys
    
class Trans_EE_m_d(Trans_SCPL_m_d):
    def __init__(self, configs):
        super(Trans_EE_m_d, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs, check_flag=False)
        self.proj_type = None
        self.pred_type = configs['pred_type']
        assert self.pred_type not in [None, ''], "Setting error, pred_type is none or empty. pred_type: {}".format(self.pred_type)
        self.pred_type  = self.pred_type  + ",non-detach"