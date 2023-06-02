import torch
import torch.nn as nn
from utils import Optimizer, CPUThread, MultiGPUModel
from itertools import chain
from collections import OrderedDict
from .nlp_single import NLP_Block, NLP_Predictor
# from transformer.encoder import TransformerEncoder

class NLP_MultiGPU(MultiGPUModel):
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

        if self.training:
            return self.train_step(X, Y, multi_t)
        else:
            return self.inference(X, Y)
    
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

        Xs = list()
        hidden =list()
        layer_fs = list()
        tasks = list()
        true_Ys = list()
        loss_all = 0
        gpu_losses = list()

        for layer in self.model:
            layer.train()

        for opt in self.opts:
            opt.zero_grad()

        true_Ys.append(Y.to(self.devices[-1]))

        # Forward: Block 0 ~ layer_num
        ## Forward: Block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask

        Xs.append(hat_Y.to(self.devices[1]))
        hidden.append(None)
        layer_fs.append(hat_Y.mean(1))

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            if i == 1:
                hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
            else:
                hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y.to(self.devices[i+1]))
            hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))
            layer_fs.append((h[0] + h[1])/2)

        ## Forward: Block -1
        hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask

        Xs.append(((h[0] + h[1])/2))
        layer_fs.append((h[0] + h[1])/2)

        hat_Y = self.model[-1](Xs[-1])
        layer_fs.append(hat_Y)

        ## Loss,Backward,Update: Block -1
        args = ([self.model[-1]], self.opts[-1].optimizer, [layer_fs[-1]], [true_Ys[-1]])
        if not multi_t:
            gpu_losses.all(self._loss_backward_update(*args)) # GPU -1
        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()

        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
    
        return layer_fs[-1], loss_all
    
    def inference(self, X, Y):
        Xs = list()
        hidden =list()
        layer_fs = list()

        true_Y = Y.to(self.devices[-1])

        ## Forward: Block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask
        Xs.append(hat_Y.to(self.devices[1]))
        hidden.append(None)

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            if i == 1:
                hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
            else:
                hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            Xs.append(hat_Y.to(self.devices[i+1]))
            hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))

        ## Forward: Block i
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
        Xs = list()
        hidden =list()
        layer_fs = list()
        tasks = list()
        true_Ys = list()
        loss_all = 0
        gpu_losses = list()

        for layer in self.model:
            layer.train()

        for opt in self.opts:
            opt.zero_grad()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True)) 

        # Forward: Block 0 ~ layer_num
        ## Forward: Block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask

        Xs.append(hat_Y.to(self.devices[1]))
        hidden.append(None)
        layer_fs.append(hat_Y.mean(1))

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            if i == 1:
                hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
            else:
                hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y.to(self.devices[i+1]))
            hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))
            layer_fs.append((h[0] + h[1])/2)

        ## Forward: Block -1
        hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask

        Xs.append(((h[0] + h[1])/2))
        layer_fs.append((h[0] + h[1])/2)

        hat_Y = self.model[-1](Xs[-1])
        layer_fs.append(hat_Y)

        ## Loss,Backward,Update: Block -1
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

        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
    
        return layer_fs[-1], loss_all
    
    def inference(self, X, Y):
        Xs = list()
        hidden =list()
        layer_fs = list()
        true_Ys = list()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True)) 
        true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        ## Forward: Block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask
        layer_fs.append(self.model[0].loss(hat_Y.mean(1)))
        Xs.append(hat_Y.to(self.devices[1]))
        hidden.append(None)

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            if i == 1:
                hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
            else:
                hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            layer_fs.append(self.model[i].loss((h[0] + h[1])/2))
            Xs.append(hat_Y.to(self.devices[i+1]))
            hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))

        ## Forward: Block -1
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
        Xs = list()
        hidden =list()
        layer_fs = list()
        tasks = list()
        true_Ys = list()
        loss_all = 0
        gpu_losses = list()

        for layer in self.model:
            layer.train()

        for opt in self.opts:
            opt.zero_grad()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True)) 

        # Forward: block 0 ~ layer_num
        ## Forward: block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask

        Xs.append(hat_Y.detach().to(self.devices[1]))
        hidden.append(None)
        layer_fs.append(hat_Y.mean(1))

        ## Loss,Backward,Update: block 0
        args = ([self.model[0]], self.opts[0].optimizer, [layer_fs[-1]], [true_Ys[0]])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # block 0
        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
        
        for i in range(1, self.num_layers-1):
            ## Forward: block i
            if i == 1:
                hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
            else:
                hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y.detach().to(self.devices[i+1]))
            hidden.append(((h.detach().to(self.devices[i+1]), c.detach().to(self.devices[i+1]))))
            layer_fs.append((h[0] + h[1])/2)

            ## Loss,Backward,Update: block i
            args = ([self.model[i]], self.opts[i].optimizer, [layer_fs[-1]], [true_Ys[i]])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # block i
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        ## Forward: block -1
        hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask

        Xs.append(((h[0] + h[1])/2).detach())
        layer_fs.append((h[0] + h[1])/2)

        hat_Y = self.model[-1](Xs[-1])
        layer_fs.append(hat_Y)

        ## Loss,Backward,Update: block -1
        args = ([self.model[-2], self.model[-1]], self.opts[-1].optimizer, [layer_fs[-2], layer_fs[-1]], [true_Ys[-1]]*2)
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # block -1
        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
        
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
        Xs = list()
        hidden =list()
        layer_fs = list()
        true_Ys = list()

        true_Ys.append(Y.to(self.devices[-1], non_blocking=True)) 

        ## Forward: Block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask
        Xs.append(hat_Y.to(self.devices[1]))
        hidden.append(None)

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            if i == 1:
                hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
            else:
                hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            Xs.append(hat_Y.to(self.devices[i+1]))
            hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))

        ## Forward: Block -1
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
        Xs = list()
        hidden =list()
        layer_fs = list()
        true_Ys = list()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True)) 
        true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        ## Forward: Block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask
        layer_fs.append(self.model[0].loss(hat_Y.mean(1)))
        Xs.append(hat_Y.to(self.devices[1]))
        hidden.append(None)

        ## Forward: Block i
        for i in range(1, self.num_layers-1):
            if i == 1:
                hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
            else:
                hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            layer_fs.append(self.model[i].loss((h[0] + h[1])/2))
            Xs.append(hat_Y.to(self.devices[i+1]))
            hidden.append(((h.to(self.devices[i+1]), c.to(self.devices[i+1]))))

        ## Forward: Block -1
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
        mask = self.get_mask(X)
        Xs = list()
        masks = list()
        hidden =list()
        layer_fs = list()
        tasks = list()
        true_Ys = list()
        loss_all = 0
        gpu_losses = list()

        for layer in self.model:
            layer.train()

        for opt in self.opts:
            opt.zero_grad()

        true_Ys.append(Y.to(self.devices[-1]))

        # Forward: block 0 ~ layer_num
        ## Forward: block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        masks.append(mask.to(self.devices[0], non_blocking=True))

        hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

        Xs.append(hat_Y.to(self.devices[1]))
        masks.append(mask.to(self.devices[1]))

        ## Forward: block i
        for i in range(1, self.num_layers-1):
            hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y.to(self.devices[i+1]))
            masks.append(mask.to(self.devices[i+1]))

        ## Forward: block -1
        hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
        
        Xs.append(hat_Y)
        masks.append(mask)
        layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

        hat_Y = self.model[-1](layer_fs[-1])
        layer_fs.append(hat_Y)

        ## Loss,Backward,Update: block -1:
        args = ([self.model[-1]], self.opts[-1].optimizer, [layer_fs[-1]], [true_Ys[-1]])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # block -1
        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
        
        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
    
        return layer_fs[-1], loss_all

    def inference(self, X, Y):
        mask = self.get_mask(X)
        Xs = list()
        masks =list()
        layer_fs = list()

        true_Y = Y.to(self.devices[-1])

        # Forward: block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        masks.append(mask.to(self.devices[0], non_blocking=True))

        hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
        Xs.append(hat_Y.to(self.devices[1]))
        masks.append(mask.to(self.devices[1]))

        # Forward: block i
        for i in range(1, self.num_layers-1):
            hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y.to(self.devices[i+1]))
            masks.append(mask.to(self.devices[i+1]))

        # Forward: block -1
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
        mask = self.get_mask(X)
        Xs = list()
        masks = list()
        hidden =list()
        layer_fs = list()
        tasks = list()
        true_Ys = list()
        loss_all = 0
        gpu_losses = list()

        for layer in self.model:
            layer.train()

        for opt in self.opts:
            opt.zero_grad()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True))

        # Forward: block 0 ~ layer_num
        ## Forward: block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        masks.append(mask.to(self.devices[0], non_blocking=True))

        hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

        Xs.append(hat_Y.to(self.devices[1]))
        masks.append(mask.to(self.devices[1]))
        layer_fs.append(self.model[0].reduction(hat_Y, hidden, mask))

        ## Forward: block i
        for i in range(1, self.num_layers-1):
            hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y.to(self.devices[i+1]))
            masks.append(mask.to(self.devices[i+1]))
            layer_fs.append(self.model[i].reduction(hat_Y, hidden, mask))

        ## Forward: block -1
        hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
        
        Xs.append(hat_Y)
        masks.append(mask)
        layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

        hat_Y = self.model[-1](layer_fs[-1])
        layer_fs.append(hat_Y)

        ## Loss,Backward,Update: block -1:
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
        
        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
    
        return layer_fs[-1], loss_all

    def inference(self, X, Y):
        mask = self.get_mask(X)
        Xs = list()
        masks =list()
        layer_fs = list()
        true_Ys = list()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True))
        true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        # Forward: block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        masks.append(mask.to(self.devices[0], non_blocking=True))

        hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mas
        layer_fs.append(self.model[0].loss(self.model[0].reduction(hat_Y, hidden, mask)))
        Xs.append(hat_Y.to(self.devices[1]))
        masks.append(mask.to(self.devices[1]))

        # Forward: block i
        for i in range(1, self.num_layers-1):
            hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            layer_fs.append(self.model[i].loss(self.model[i].reduction(hat_Y, hidden, mask)))
            Xs.append(hat_Y.to(self.devices[i+1]))
            masks.append(mask.to(self.devices[i+1]))

        # Forward: block -1
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
        mask = self.get_mask(X)
        Xs = list()
        masks = list()
        hidden =list()
        layer_fs = list()
        tasks = list()
        true_Ys = list()
        loss_all = 0
        gpu_losses = list()

        for layer in self.model:
            layer.train()

        for opt in self.opts:
            opt.zero_grad()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True)) 

        # Forward: Block 0 ~ layer_num
        ## Forward: Block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        masks.append(mask.to(self.devices[0], non_blocking=True))

        hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

        Xs.append(hat_Y.detach().to(self.devices[1]))
        masks.append(mask.to(self.devices[1]).detach())
        layer_fs.append(self.model[0].reduction(hat_Y, hidden, mask))

        ## Loss,Backward,Update: Block 0
        args = ([self.model[0]], self.opts[0].optimizer, [layer_fs[-1]], [true_Ys[0]])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 0
        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            # print("gpu ", 0, tasks[-1].name)
            tasks[-1].start()

        for i in range(1, self.num_layers-1):
            ## Forward: Block i
            hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y.detach().to(self.devices[i+1]))
            masks.append(mask.to(self.devices[i+1]).detach())
            layer_fs.append(self.model[i].reduction(hat_Y, hidden, mask))

            ## Loss,Backward,Update: Block i
            args = ([self.model[i]], self.opts[i].optimizer, [layer_fs[-1]], [true_Ys[i]])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block i
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        ## Forward: Block -1
        hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
        
        Xs.append(hat_Y.detach())
        masks.append(mask.detach())
        layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

        hat_Y = self.model[-1](layer_fs[-1].detach())
        layer_fs.append(hat_Y)

        ## Loss,Backward,Update: Block -1
        args = ([self.model[-2], self.model[-1]], self.opts[-1].optimizer, [layer_fs[-2], layer_fs[-1]], [true_Ys[-1]]*2)
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block -1
        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
    
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
        mask = self.get_mask(X)
        Xs = list()
        masks =list()
        layer_fs = list()
        true_Ys = list()

        true_Ys.append(Y.to(self.devices[-1], non_blocking=True)) 

        # Forward: block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        masks.append(mask.to(self.devices[0], non_blocking=True))

        hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mas
        Xs.append(hat_Y.to(self.devices[1]))
        masks.append(mask.to(self.devices[1]))

        # Forward: block i
        for i in range(1, self.num_layers-1):
            hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            Xs.append(hat_Y.to(self.devices[i+1]))
            masks.append(mask.to(self.devices[i+1]))

        # Forward: block -1
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
        mask = self.get_mask(X)
        Xs = list()
        masks =list()
        layer_fs = list()
        true_Ys = list()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True))
        true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        # Forward: block 0
        Xs.append(X.to(self.devices[0], non_blocking=True))
        masks.append(mask.to(self.devices[0], non_blocking=True))

        hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mas
        layer_fs.append(self.model[0].loss(self.model[0].reduction(hat_Y, hidden, mask)))
        Xs.append(hat_Y.to(self.devices[1]))
        masks.append(mask.to(self.devices[1]))

        # Forward: block i
        for i in range(1, self.num_layers-1):
            hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            layer_fs.append(self.model[i].loss(self.model[i].reduction(hat_Y, hidden, mask)))
            Xs.append(hat_Y.to(self.devices[i+1]))
            masks.append(mask.to(self.devices[i+1]))

        # Forward: block -1
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

class LSTM_SCPL_m_4(NLP_MultiGPU):
    def _init_model(self, configs):
        # Model
        self.num_classes = configs["n_classes"]
        self.word_vec = configs["word_vec"]
        self.vocab_size = configs["vocab_size"]
        self.emb_dim = configs["emb_dim"]
        self.h_dim = configs["h_dim"]
        self.proj_type = configs['proj_type']

        # self.layer_cfg = configs["layer_cfg"]

        self.layer_cfg = {
            0:{"inp_dim":self.vocab_size, "out_dim":self.emb_dim, 
               "f":"emb", "h_dim":self.h_dim, "word_vec":self.word_vec, "device":self.devices[0], "proj_type":self.proj_type}, 
            1:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.devices[1], "proj_type":self.proj_type},
            2:{"inp_dim":self.h_dim*2, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.devices[2], "proj_type":self.proj_type},
            3:{"inp_dim":self.h_dim*2, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.devices[3], "proj_type":self.proj_type},
            4:{"inp_dim":self.h_dim, "out_dim":self.num_classes, 
               "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.devices[3]}}

        self.layer0 = NLP_Block(**self.layer_cfg[0]).to(self.devices[0])
        self.layer1 = NLP_Block(**self.layer_cfg[1]).to(self.devices[1])
        self.layer2 = NLP_Block(**self.layer_cfg[2]).to(self.devices[2])
        self.layer3 = NLP_Block(**self.layer_cfg[3]).to(self.devices[3])
        self.layer4 = NLP_Predictor(**self.layer_cfg[4]).to(self.devices[3])

        self.model = [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def _init_optimizers(self, configs):
        # Optimizers
        self.base_lr = configs["base_lr"]
        self.end_lr = configs["end_lr"]
        self.max_step = configs["max_steps"]
        self.global_steps = 0

        self.gpu_0_opt = Optimizer(chain(self.layer0.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_1_opt = Optimizer(chain(self.layer1.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_2_opt = Optimizer(chain(self.layer2.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_3_opt = Optimizer(chain(self.layer3.parameters(),self.layer4.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)

        self.opts = [
            self.gpu_0_opt,
            self.gpu_1_opt,
            self.gpu_2_opt,
            self.gpu_3_opt,
        ]

    def train_step(self, X, Y, multi_t=True):
        
        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for opt in self.opts:
            opt.zero_grad()
        
        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0, _, _ = self.layer0(x0)
        
        x1 = hat_y0.detach().to(self.devices[1])
        hat_y0 = hat_y0.mean(1)
        
        ## Loss,Backward,Update: GPU 0
        if not multi_t:
            gpu_loss_0 = self._loss_backward_update([self.layer0], self.gpu_0_opt.optimizer, [hat_y0], true_y0) # GPU 0

        else:
            args = ([self.layer0], self.gpu_0_opt.optimizer, [hat_y0], true_y0)
            t0 = CPUThread(target=self._loss_backward_update, args=args)
            t0.start()
            
        ## Forward: GPU 1
        hat_y1, (h1, c1), _ = self.layer1(x1)      
        h_state1 = (h1[0] + h1[1])/2
        
        x2 = hat_y1.detach().to(self.devices[2])
        hidden2 = ((h1.detach().to(self.devices[2]), c1.detach().to(self.devices[2])))

        ## Loss,Backward,Update: GPU 1
        if not multi_t:
            gpu_loss_1 = self._loss_backward_update([self.layer1], self.gpu_1_opt.optimizer, [h_state1], true_y1) # GPU 1

        else:
            args = ([self.layer1], self.gpu_1_opt.optimizer, [h_state1], true_y1)
            t1 = CPUThread(target=self._loss_backward_update, args=args)
            t1.start() 
        
        ## Forward: GPU 2
        hat_y2, (h2, c2), _ = self.layer2(x2, hidden=hidden2)      
        h_state2 = (h2[0] + h2[1])/2

        x3 = hat_y2.detach().to(self.devices[3])
        hidden3 = ((h2.detach().to(self.devices[3]), c2.detach().to(self.devices[3])))
        # x3 = h_state2.detach().to(self.devices[3])
        
        ## Loss,Backward,Update: GPU 2
        if not multi_t:
            gpu_loss_2 = self._loss_backward_update([self.layer2], self.gpu_2_opt.optimizer, [h_state2], true_y2) # GPU 2

        else:
            args = ([self.layer2], self.gpu_2_opt.optimizer, [h_state2], true_y2)
            t2 = CPUThread(target=self._loss_backward_update, args=args)
            t2.start()

        ## Forward: GPU 3
        hat_y3, (h3, c3), _ = self.layer3(x3, hidden=hidden3)        
        h_state3 = (h3[0] + h3[1])/2
        
        # x4 = hat_y3.detach().to(self.devices[3])
        # hidden4 = ((h3.detach().to(self.devices[3]), c3.detach().to(self.devices[3])))
        x4 = h_state3.detach()
        
        hat_y4 = self.layer4(x4)       # GPU 2

        ## Loss,Backward,Update: GPU 2
        if not multi_t:
            gpu_loss_3 = self._loss_backward_update([self.layer3, self.layer4], self.gpu_3_opt.optimizer, [h_state3, hat_y4], true_y3) # GPU 3

        else:
            args = ([self.layer3, self.layer4], self.gpu_3_opt.optimizer, [h_state3, hat_y4], true_y3)
            t3 = CPUThread(target=self._loss_backward_update, args=args)
            t3.start()
        
        if multi_t:
            gpu_loss_0 = t0.get_result()
            gpu_loss_1 = t1.get_result()
            gpu_loss_2 = t2.get_result()
            gpu_loss_3 = t3.get_result()

        loss_all =  gpu_loss_0.item() + gpu_loss_1.item() + gpu_loss_2.item() + gpu_loss_3.item()
    
        return hat_y4, loss_all

    def inference(self, X, Y):
        x0 = X.to(self.devices[0], non_blocking=True)
        true_y3 = Y.to(self.devices[3])
    
        hat_y0, _, _ = self.layer0(x0)
        x1 = hat_y0.to(self.devices[1])
            
        hat_y1, (h1, c1), _ = self.layer1(x1) 
        x2 = hat_y1.to(self.devices[2])
        hidden2 = (h1.to(self.devices[2]), c1.to(self.devices[2])) 
            
        hat_y2, (h2, c2), _ = self.layer2(x2, hidden=hidden2) 
        x3 = hat_y2.to(self.devices[3])
        hidden3 = (h2.to(self.devices[3]), c2.to(self.devices[3])) 
            
        hat_y3, (h3, c3), _ = self.layer3(x3, hidden=hidden3) 
        x4 = ((h3[0] + h3[1])/2)
            
        hat_y4 = self.layer4(x4) 

        return hat_y4, true_y3

class Trans_SCPL_m_4(NLP_MultiGPU):
    def _init_model(self, configs):
        # Model
        self.num_classes = configs["n_classes"]
        self.word_vec = configs["word_vec"]
        self.vocab_size = configs["vocab_size"]
        self.emb_dim = configs["emb_dim"]
        self.h_dim = configs["h_dim"]
        self.n_heads = configs["head"]
        self.proj_type = configs['proj_type']

        # self.layer_cfg = configs["layer_cfg"]

        print("Trans_SCPL_Model - 3 layers Transformer - multi_t ver.")

        self.layer_cfg = {
            0:{"inp_dim":self.vocab_size, "out_dim":self.emb_dim, "f":"emb", 
            "h_dim":self.emb_dim, "word_vec":self.word_vec, "device":self.devices[0], "proj_type":self.proj_type}, 
            1:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.devices[1], "proj_type":self.proj_type},
            2:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.devices[2], "proj_type":self.proj_type},
            3:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.devices[3], "proj_type":self.proj_type},
            4:{"inp_dim":self.emb_dim, "out_dim":self.num_classes, 
            "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.devices[3]}}

        self.layer0 = NLP_Block(**self.layer_cfg[0]).to(self.devices[0])
        self.layer1 = NLP_Block(**self.layer_cfg[1]).to(self.devices[1])
        self.layer2 = NLP_Block(**self.layer_cfg[2]).to(self.devices[2])
        self.layer3 = NLP_Block(**self.layer_cfg[3]).to(self.devices[3])
        self.layer4 = NLP_Predictor(**self.layer_cfg[4]).to(self.devices[3])

        self.model = [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def _init_optimizers(self, configs):
        # Optimizers
        self.base_lr = configs["base_lr"]
        self.end_lr = configs["end_lr"]
        self.max_step = configs["max_steps"]
        self.global_steps = 0

        self.gpu_0_opt = Optimizer(chain(self.layer0.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_1_opt = Optimizer(chain(self.layer1.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_2_opt = Optimizer(chain(self.layer2.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_3_opt = Optimizer(chain(self.layer3.parameters(),self.layer4.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)

        self.opts = [
            self.gpu_0_opt,
            self.gpu_1_opt,
            self.gpu_2_opt,
            self.gpu_3_opt,
        ]

    def train_step(self, X, Y, multi_t=True):
        mask = self.get_mask(X)

        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for opt in self.opts:
            opt.zero_grad()
        
        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        x0 = X.to(self.devices[0], non_blocking=True)
        mask0 = mask.to(self.devices[0], non_blocking=True)
        out0, hidden0, mask0 = self.layer0(x0, mask=mask0)
        feature0 = self.layer0.reduction(out0, hidden0, mask0)
        
        x1 = out0.detach().to(self.devices[1])
        mask1 = mask0.detach().to(self.devices[1])
        
        ## Loss,Backward,Update: GPU 0
        if not multi_t:
            gpu_loss_0 = self._loss_backward_update([self.layer0], self.gpu_0_opt.optimizer, [feature0], true_y0) # GPU 0

        else:
            args = ([self.layer0], self.gpu_0_opt.optimizer, [feature0], true_y0)
            t0 = CPUThread(target=self._loss_backward_update, args=args)
            t0.start()
            
        ## Forward: GPU 1
        out1, hidden1, mask1 = self.layer1(x1, mask=mask1)   
        feature1 = self.layer1.reduction(out1, hidden1, mask1)
        
        x2 = out1.detach().to(self.devices[2])
        mask2 = mask1.detach().to(self.devices[2])

        ## Loss,Backward,Update: GPU 1
        if not multi_t:
            gpu_loss_1 = self._loss_backward_update([self.layer1], self.gpu_1_opt.optimizer, [feature1], true_y1) # GPU 1

        else:
            args = ([self.layer1], self.gpu_1_opt.optimizer, [feature1], true_y1)
            t1 = CPUThread(target=self._loss_backward_update, args=args)
            t1.start() 
        
        ## Forward: GPU 2
        out2, hidden2, mask2 = self.layer2(x2, mask=mask2)   
        feature2 = self.layer2.reduction(out2, hidden2, mask2)
        
        x3 = out2.detach().to(self.devices[3])
        mask3 = mask2.detach().to(self.devices[3])
        
        ## Loss,Backward,Update: GPU 2
        if not multi_t:
            gpu_loss_2 = self._loss_backward_update([self.layer2], self.gpu_2_opt.optimizer, [feature2], true_y2) # GPU 2

        else:
            args = ([self.layer2], self.gpu_2_opt.optimizer, [feature2], true_y2)
            t2 = CPUThread(target=self._loss_backward_update, args=args)
            t2.start()

        ## Forward: GPU 3
        out3, hidden3, mask3 = self.layer3(x3, mask=mask3)   
        feature3 = self.layer3.reduction(out3, hidden3, mask3)
        
        x4 = feature3.detach()
        
        out4 = self.layer4(x4)       # GPU 2

        ## Loss,Backward,Update: GPU 2
        if not multi_t:
            gpu_loss_3 = self._loss_backward_update([self.layer3, self.layer4], self.gpu_3_opt.optimizer, [feature3, out4], true_y3) # GPU 3

        else:
            args = ([self.layer3, self.layer4], self.gpu_3_opt.optimizer, [feature3, out4], true_y3)
            t3 = CPUThread(target=self._loss_backward_update, args=args)
            t3.start()
        
        if multi_t:
            gpu_loss_0 = t0.get_result()
            gpu_loss_1 = t1.get_result()
            gpu_loss_2 = t2.get_result()
            gpu_loss_3 = t3.get_result()

        loss_all =  gpu_loss_0.item() + gpu_loss_1.item() + gpu_loss_2.item() + gpu_loss_3.item()
    
        return out4, loss_all

    def inference(self, X, Y):
        mask = self.get_mask(X)

        mask0 = mask.to(self.devices[0], non_blocking=True)
        mask1 = mask.to(self.devices[1])
        mask2 = mask.to(self.devices[2])
        mask3 = mask.to(self.devices[3])

        x0 = X.to(self.devices[0], non_blocking=True)
        true_y3 = Y.to(self.devices[3])
    
        out0, hidden0, mask0 = self.layer0(x0, mask=mask0)
        x1 = out0.to(self.devices[1])
            
        out1, hidden1, mask1= self.layer1(x1, mask=mask1) 
        x2 = out1.to(self.devices[2])
            
        out2, hidden2, mask2 = self.layer2(x2, mask=mask2) 
        x3 = out2.to(self.devices[3])
            
        out3, hidden3, mask3 = self.layer3(x3, mask=mask3) 
        x4 = self.layer3.reduction(out3, hidden3, mask3)
            
        out4 = self.layer4(x4) 

        return out4, true_y3

class LSTM_SCPL_m_4_old(nn.Module):
    def __init__(self, configs):
        super().__init__()

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

    def _init_model(self, configs):
        # Model
        self.num_classes = configs["n_classes"]
        self.word_vec = configs["word_vec"]
        self.vocab_size = configs["vocab_size"]
        self.emb_dim = configs["emb_dim"]
        self.h_dim = configs["h_dim"]
        self.proj_type = configs['proj_type']


        # self.layer_cfg = configs["layer_cfg"]

        self.layer_cfg = {
            0:{"inp_dim":self.vocab_size, "out_dim":self.emb_dim, 
               "f":"emb", "h_dim":self.h_dim, "word_vec":self.word_vec, "device":self.devices[0], "proj_type":self.proj_type}, 
            1:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.devices[1], "proj_type":self.proj_type},
            2:{"inp_dim":self.h_dim*2, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.devices[2], "proj_type":self.proj_type},
            3:{"inp_dim":self.h_dim*2, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.devices[3], "proj_type":self.proj_type},
            4:{"inp_dim":self.h_dim, "out_dim":self.num_classes, 
               "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.devices[3]}}

        self.layer0 = NLP_Block(**self.layer_cfg[0]).to(self.devices[0])
        self.layer1 = NLP_Block(**self.layer_cfg[1]).to(self.devices[1])
        self.layer2 = NLP_Block(**self.layer_cfg[2]).to(self.devices[2])
        self.layer3 = NLP_Block(**self.layer_cfg[3]).to(self.devices[3])
        self.layer4 = NLP_Predictor(**self.layer_cfg[4]).to(self.devices[3])

        self.model = [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def _init_optimizers(self, configs):
        # Optimizers
        self.base_lr = configs["base_lr"]
        self.end_lr = configs["end_lr"]
        self.max_step = configs["max_steps"]
        self.global_steps = 0

        self.gpu_0_opt = Optimizer(chain(self.layer0.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_1_opt = Optimizer(chain(self.layer1.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_2_opt = Optimizer(chain(self.layer2.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_3_opt = Optimizer(chain(self.layer3.parameters(),self.layer4.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)

        self.opts = [
            self.gpu_0_opt,
            self.gpu_1_opt,
            self.gpu_2_opt,
            self.gpu_3_opt,
        ]

    def _loss_backward_update(self, layer_model, optimizer, hat_y, true_y):
        loss = 0
        for i, layer in enumerate(layer_model):
            loss += layer.loss(hat_y[i], true_y)
        loss.backward()
        optimizer.step()
        return loss

    def train_step(self, X, Y, multi_t=True):
        
        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for opt in self.opts:
            opt.zero_grad()
        
        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0, _, _ = self.layer0(x0)
        
        x1 = hat_y0.detach().to(self.devices[1])
        hat_y0 = hat_y0.mean(1)
        
        ## Loss,Backward,Update: GPU 0
        if not multi_t:
            gpu_loss_0 = self._loss_backward_update([self.layer0], self.gpu_0_opt.optimizer, [hat_y0], true_y0) # GPU 0

        else:
            args = ([self.layer0], self.gpu_0_opt.optimizer, [hat_y0], true_y0)
            t0 = CPUThread(target=self._loss_backward_update, args=args)
            t0.start()
            
        ## Forward: GPU 1
        hat_y1, (h1, c1), _ = self.layer1(x1)      
        h_state1 = (h1[0] + h1[1])/2
        
        x2 = hat_y1.detach().to(self.devices[2])
        hidden2 = ((h1.detach().to(self.devices[2]), c1.detach().to(self.devices[2])))

        ## Loss,Backward,Update: GPU 1
        if not multi_t:
            gpu_loss_1 = self._loss_backward_update([self.layer1], self.gpu_1_opt.optimizer, [h_state1], true_y1) # GPU 1

        else:
            args = ([self.layer1], self.gpu_1_opt.optimizer, [h_state1], true_y1)
            t1 = CPUThread(target=self._loss_backward_update, args=args)
            t1.start() 
        
        ## Forward: GPU 2
        hat_y2, (h2, c2), _ = self.layer2(x2, hidden=hidden2)      
        h_state2 = (h2[0] + h2[1])/2

        x3 = hat_y2.detach().to(self.devices[3])
        hidden3 = ((h2.detach().to(self.devices[3]), c2.detach().to(self.devices[3])))
        # x3 = h_state2.detach().to(self.devices[3])
        
        ## Loss,Backward,Update: GPU 2
        if not multi_t:
            gpu_loss_2 = self._loss_backward_update([self.layer2], self.gpu_2_opt.optimizer, [h_state2], true_y2) # GPU 2

        else:
            args = ([self.layer2], self.gpu_2_opt.optimizer, [h_state2], true_y2)
            t2 = CPUThread(target=self._loss_backward_update, args=args)
            t2.start()

        ## Forward: GPU 3
        hat_y3, (h3, c3), _ = self.layer3(x3, hidden=hidden3)        
        h_state3 = (h3[0] + h3[1])/2
        
        # x4 = hat_y3.detach().to(self.devices[3])
        # hidden4 = ((h3.detach().to(self.devices[3]), c3.detach().to(self.devices[3])))
        x4 = h_state3.detach()
        
        hat_y4 = self.layer4(x4)       # GPU 2

        ## Loss,Backward,Update: GPU 2
        if not multi_t:
            gpu_loss_3 = self._loss_backward_update([self.layer3, self.layer4], self.gpu_3_opt.optimizer, [h_state3, hat_y4], true_y3) # GPU 3

        else:
            args = ([self.layer3, self.layer4], self.gpu_3_opt.optimizer, [h_state3, hat_y4], true_y3)
            t3 = CPUThread(target=self._loss_backward_update, args=args)
            t3.start()
        
        if multi_t:
            gpu_loss_0 = t0.get_result()
            gpu_loss_1 = t1.get_result()
            gpu_loss_2 = t2.get_result()
            gpu_loss_3 = t3.get_result()

        loss_all =  gpu_loss_0.item() + gpu_loss_1.item() + gpu_loss_2.item() + gpu_loss_3.item()
    
        return hat_y4, loss_all

    def inference(self, X, Y):
        x0 = X.to(self.devices[0], non_blocking=True)
        true_y3 = Y.to(self.devices[3])
    
        hat_y0, _, _ = self.layer0(x0)
        x1 = hat_y0.to(self.devices[1])
            
        hat_y1, (h1, c1), _ = self.layer1(x1) 
        x2 = hat_y1.to(self.devices[2])
        hidden2 = (h1.to(self.devices[2]), c1.to(self.devices[2])) 
            
        hat_y2, (h2, c2), _ = self.layer2(x2, hidden=hidden2) 
        x3 = hat_y2.to(self.devices[3])
        hidden3 = (h2.to(self.devices[3]), c2.to(self.devices[3])) 
            
        hat_y3, (h3, c3), _ = self.layer3(x3, hidden=hidden3) 
        x4 = ((h3[0] + h3[1])/2)
            
        hat_y4 = self.layer4(x4) 

        return hat_y4, true_y3

    def forward(self, X, Y, multi_t=True):

        if self.training:
            return self.train_step(X, Y, multi_t)
        else:
            return self.inference(X, Y)

    def opt_step(self, global_steps):
        for opt in self.opts:
            lr = opt.step(global_steps)
        return lr

class Trans_SCPL_m_4_old(nn.Module):
    def __init__(self, configs):
        super().__init__()

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

    def _init_model(self, configs):
        # Model
        self.num_classes = configs["n_classes"]
        self.word_vec = configs["word_vec"]
        self.vocab_size = configs["vocab_size"]
        self.emb_dim = configs["emb_dim"]
        self.h_dim = configs["h_dim"]
        self.n_heads = configs["head"]
        self.proj_type = configs['proj_type']

        # self.layer_cfg = configs["layer_cfg"]

        print("Trans_SCPL_Model - 3 layers Transformer - MultiGPU ver.")

        self.layer_cfg = {
            0:{"inp_dim":self.vocab_size, "out_dim":self.emb_dim, "f":"emb", 
            "h_dim":self.emb_dim, "word_vec":self.word_vec, "device":self.devices[0], "proj_type":self.proj_type}, 
            1:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.devices[1], "proj_type":self.proj_type},
            2:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.devices[2], "proj_type":self.proj_type},
            3:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.devices[3], "proj_type":self.proj_type},
            4:{"inp_dim":self.emb_dim, "out_dim":self.num_classes, 
            "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.devices[3]}}

        self.layer0 = NLP_Block(**self.layer_cfg[0]).to(self.devices[0])
        self.layer1 = NLP_Block(**self.layer_cfg[1]).to(self.devices[1])
        self.layer2 = NLP_Block(**self.layer_cfg[2]).to(self.devices[2])
        self.layer3 = NLP_Block(**self.layer_cfg[3]).to(self.devices[3])
        self.layer4 = NLP_Predictor(**self.layer_cfg[4]).to(self.devices[3])

        self.model = [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def _init_optimizers(self, configs):
        # Optimizers
        self.base_lr = configs["base_lr"]
        self.end_lr = configs["end_lr"]
        self.max_step = configs["max_steps"]
        self.global_steps = 0

        self.gpu_0_opt = Optimizer(chain(self.layer0.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_1_opt = Optimizer(chain(self.layer1.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_2_opt = Optimizer(chain(self.layer2.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)
        self.gpu_3_opt = Optimizer(chain(self.layer3.parameters(),self.layer4.parameters(),), base_lr=self.base_lr, 
                                        end_lr=self.end_lr, max_step=self.max_step)

        self.opts = [
            self.gpu_0_opt,
            self.gpu_1_opt,
            self.gpu_2_opt,
            self.gpu_3_opt,
        ]

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask

    def _loss_backward_update(self, layer_model, optimizer, hat_y, true_y):
        loss = 0
        for i, layer in enumerate(layer_model):
            loss += layer.loss(hat_y[i], true_y)
        loss.backward()
        optimizer.step()
        return loss

    def train_step(self, X, Y, multi_t=True):
        mask = self.get_mask(X)

        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for opt in self.opts:
            opt.zero_grad()
        
        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        x0 = X.to(self.devices[0], non_blocking=True)
        mask0 = mask.to(self.devices[0], non_blocking=True)
        out0, hidden0, mask0 = self.layer0(x0, mask=mask0)
        feature0 = self.layer0.reduction(out0, hidden0, mask0)
        
        x1 = out0.detach().to(self.devices[1])
        mask1 = mask0.detach().to(self.devices[1])
        
        ## Loss,Backward,Update: GPU 0
        if not multi_t:
            gpu_loss_0 = self._loss_backward_update([self.layer0], self.gpu_0_opt.optimizer, [feature0], true_y0) # GPU 0

        else:
            args = ([self.layer0], self.gpu_0_opt.optimizer, [feature0], true_y0)
            t0 = CPUThread(target=self._loss_backward_update, args=args)
            t0.start()
            
        ## Forward: GPU 1
        out1, hidden1, mask1 = self.layer1(x1, mask=mask1)   
        feature1 = self.layer1.reduction(out1, hidden1, mask1)
        
        x2 = out1.detach().to(self.devices[2])
        mask2 = mask1.detach().to(self.devices[2])

        ## Loss,Backward,Update: GPU 1
        if not multi_t:
            gpu_loss_1 = self._loss_backward_update([self.layer1], self.gpu_1_opt.optimizer, [feature1], true_y1) # GPU 1

        else:
            args = ([self.layer1], self.gpu_1_opt.optimizer, [feature1], true_y1)
            t1 = CPUThread(target=self._loss_backward_update, args=args)
            t1.start() 
        
        ## Forward: GPU 2
        out2, hidden2, mask2 = self.layer2(x2, mask=mask2)   
        feature2 = self.layer2.reduction(out2, hidden2, mask2)
        
        x3 = out2.detach().to(self.devices[3])
        mask3 = mask2.detach().to(self.devices[3])
        
        ## Loss,Backward,Update: GPU 2
        if not multi_t:
            gpu_loss_2 = self._loss_backward_update([self.layer2], self.gpu_2_opt.optimizer, [feature2], true_y2) # GPU 2

        else:
            args = ([self.layer2], self.gpu_2_opt.optimizer, [feature2], true_y2)
            t2 = CPUThread(target=self._loss_backward_update, args=args)
            t2.start()

        ## Forward: GPU 3
        out3, hidden3, mask3 = self.layer3(x3, mask=mask3)   
        feature3 = self.layer3.reduction(out3, hidden3, mask3)
        
        x4 = feature3.detach()
        
        out4 = self.layer4(x4)       # GPU 2

        ## Loss,Backward,Update: GPU 2
        if not multi_t:
            gpu_loss_3 = self._loss_backward_update([self.layer3, self.layer4], self.gpu_3_opt.optimizer, [feature3, out4], true_y3) # GPU 3

        else:
            args = ([self.layer3, self.layer4], self.gpu_3_opt.optimizer, [feature3, out4], true_y3)
            t3 = CPUThread(target=self._loss_backward_update, args=args)
            t3.start()
        
        if multi_t:
            gpu_loss_0 = t0.get_result()
            gpu_loss_1 = t1.get_result()
            gpu_loss_2 = t2.get_result()
            gpu_loss_3 = t3.get_result()

        loss_all =  gpu_loss_0.item() + gpu_loss_1.item() + gpu_loss_2.item() + gpu_loss_3.item()
    
        return out4, loss_all

    def inference(self, X, Y):
        mask = self.get_mask(X)

        mask0 = mask.to(self.devices[0], non_blocking=True)
        mask1 = mask.to(self.devices[1])
        mask2 = mask.to(self.devices[2])
        mask3 = mask.to(self.devices[3])

        x0 = X.to(self.devices[0], non_blocking=True)
        true_y3 = Y.to(self.devices[3])
    
        out0, hidden0, mask0 = self.layer0(x0, mask=mask0)
        x1 = out0.to(self.devices[1])
            
        out1, hidden1, mask1= self.layer1(x1, mask=mask1) 
        x2 = out1.to(self.devices[2])
            
        out2, hidden2, mask2 = self.layer2(x2, mask=mask2) 
        x3 = out2.to(self.devices[3])
            
        out3, hidden3, mask3 = self.layer3(x3, mask=mask3) 
        x4 = self.layer3.reduction(out3, hidden3, mask3)
            
        out4 = self.layer4(x4) 

        return out4, true_y3

    def forward(self, X, Y, multi_t=True):

        if self.training:
            return self.train_step(X, Y, multi_t)
        else:
            return self.inference(X, Y)

    def opt_step(self, global_steps):
        for opt in self.opts:
            lr = opt.step(global_steps)
        return lr
