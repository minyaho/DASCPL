import torch
import torch.nn as nn
from utils import Optimizer, CPUThread, ProfilerMultiGPUModel
from itertools import chain
from .nlp_single import NLP_LocalLoss_Component, NLP_BP_Component, NLP_Loss_Predictor
# from transformer.encoder import TransformerEncoder

class NLP_MultiGPU(ProfilerMultiGPUModel):
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
        self.gpus = configs["gpus"]
        self.dataset = configs["dataset"]
        self.train_loader = configs["train_loader"]
        self.test_loader = configs["test_loader"]

    def _init_model(self, configs):
        self.num_classes = configs["n_classes"]
        self.word_vec = configs["word_vec"]
        self.vocab_size = configs["vocab_size"]
        self.emb_dim = configs["emb_dim"]
        self.h_dim = configs["h_dim"]
        self.save_path = configs["save_path"] if configs["save_path"] != None else './{}'.format(self.__class__.__name__)
    
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
    def _init_data(self, configs):
        super()._init_data(configs)

        assert configs['layers'] >= 2, "Model layer setting error! The number of layers must be greater than 2."

        self.layer_num = configs['layers']
        self.gpu_num = set(self.gpus)
        # print("[Model] LSTM_BP_Model - {0} layers".format(self.layer_num))

    def _init_model(self, configs):
        # Setting Model
        super()._init_model(configs)

        self.layer_cfg = dict()

        # Embedding
        self.layer_cfg[0] = {"inp_dim":self.vocab_size, "out_dim":self.emb_dim, "f":"emb", "h_dim":self.h_dim, "word_vec":self.word_vec, "device":self.gpus[0]}

        # LSTM
        for i in range(self.layer_num-1):
            if i == 0:
                self.layer_cfg[i+1] = {"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.gpus[i+1]}
            else:
                self.layer_cfg[i+1] = {"inp_dim":self.h_dim*2, "out_dim":self.h_dim, "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.gpus[i+1]}
        
        # Predict
        self.layer_cfg[self.layer_num] = {"inp_dim":self.h_dim, "out_dim":self.num_classes, "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.gpus[-1]}

        # Make Model
        self.model = []
        # Embedding and LSTM
        for i in range(self.layer_num):
            self.model.append(NLP_BP_Component(**self.layer_cfg[i]).to(self.gpus[i]))
        # Predict
        self.model.append(NLP_Loss_Predictor(**self.layer_cfg[self.layer_num]).to(self.gpus[-1]))

        self.model = torch.nn.Sequential(*self.model)

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
            true_Ys.append(Y.to(self.gpus[-1]))

        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.gpus[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask

            Xs.append(hat_Y.to(self.gpus[1]))
            hidden.append(None)
            layer_fs.append(hat_Y.mean(1))

        for i in range(1, self.layer_num-1):
            ## Forward: GPU i
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.to(self.gpus[i+1]))
                hidden.append(((h.to(self.gpus[i+1]), c.to(self.gpus[i+1]))))
                layer_fs.append((h[0] + h[1])/2)

        ## Forward: GPU -1
        with torch.profiler.record_function("Forward: Block {}".format(self.layer_num-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask

            Xs.append(((h[0] + h[1])/2))
            layer_fs.append((h[0] + h[1])/2)

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)

        ## Loss,Backward,Update: GPU -1
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(self.layer_num-1)):
            args = ([self.model[-1]], self.opts[-1].optimizer, [layer_fs[-1]], true_Ys[-1])
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
            true_Y = Y.to(self.gpus[-1])

        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.gpus[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask
            Xs.append(hat_Y.to(self.gpus[1]))
            hidden.append(None)

        for i in range(1, self.layer_num-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                Xs.append(hat_Y.to(self.gpus[i+1]))
                hidden.append(((h.to(self.gpus[i+1]), c.to(self.gpus[i+1]))))

        with torch.profiler.record_function("Forward: Block {}".format(self.layer_num-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            Xs.append(((h[0] + h[1])/2))

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)

        return layer_fs[-1], true_Y

class LSTM_SCPL_m_d(NLP_MultiGPU):
    def _init_data(self, configs):
        super()._init_data(configs)

        assert configs['layers'] >= 2, "Model layer setting error! The number of layers must be greater than 2."

        self.layer_num = configs['layers']
        self.gpu_num = set(self.gpus)
        # print("[Model] LSTM_SCPL_Model - {0} layers".format(self.layer_num))

    def _init_model(self, configs):
        # Setting Model
        super()._init_model(configs)
        self.proj_type = configs['proj_type']

        self.layer_cfg = dict()

        # Embedding
        self.layer_cfg[0] = {
            "inp_dim":self.vocab_size, "out_dim":self.emb_dim, "f":"emb", "h_dim":self.h_dim, 
            "word_vec":self.word_vec, "device":self.gpus[0], "proj_type":self.proj_type}

        # LSTM
        for i in range(self.layer_num-1):
            if i == 0:
                self.layer_cfg[i+1] = {
                    "inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"lstm", "h_dim":self.h_dim, 
                    "word_vec":None, "device":self.gpus[i+1], "proj_type":self.proj_type}
            else:
                self.layer_cfg[i+1] = {
                    "inp_dim":self.h_dim*2, "out_dim":self.h_dim, "f":"lstm", "h_dim":self.h_dim, 
                    "word_vec":None, "device":self.gpus[i+1], "proj_type":self.proj_type}
        
        # Predict
        self.layer_cfg[self.layer_num] = {
            "inp_dim":self.h_dim, "out_dim":self.num_classes, "hid_dim":self.h_dim, 
            "act_fun":nn.Tanh(), "device":self.gpus[-1]}

        # Make Model
        self.model = []
        # Embedding and LSTM
        for i in range(self.layer_num):
            self.model.append(NLP_LocalLoss_Component(**self.layer_cfg[i]).to(self.gpus[i]))
        # Predictor
        self.model.append(NLP_Loss_Predictor(**self.layer_cfg[self.layer_num]).to(self.gpus[-1]))

        self.model = torch.nn.Sequential(*self.model)

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = list()

        self.opts.append(Optimizer(
            chain(self.model[0].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

        for i in range(1, self.layer_num-1):
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
            for i in range(self.layer_num):
                true_Ys.append(Y.to(self.gpus[i], non_blocking=True)) 

        # Forward: GPU
        ## Forward: GPU 0
        with torch.profiler.record_function("Forward: Block {}".format(0)):
            Xs.append(X.to(self.gpus[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask

            Xs.append(hat_Y.detach().to(self.gpus[1]))
            hidden.append(None)
            layer_fs.append(hat_Y.mean(1))

        ## Loss,Backward,Update: GPU 0
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(0)):
            args = ([self.model[0]], self.opts[0].optimizer, [layer_fs[-1]], true_Ys[0])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # GPU 0
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                # print("gpu 1", tasks[-1].name)
                tasks[-1].start()

        for i in range(1, self.layer_num-1):
            ## Forward: GPU i
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.detach().to(self.gpus[i+1]))
                hidden.append(((h.detach().to(self.gpus[i+1]), c.detach().to(self.gpus[i+1]))))
                layer_fs.append((h[0] + h[1])/2)

            ## Loss,Backward,Update: GPU i
            with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(i)):
                args = ([self.model[i]], self.opts[i].optimizer, [layer_fs[-1]], true_Ys[i])
                if not multi_t:
                    gpu_losses.append(self._loss_backward_update(*args)) # GPU i
                else:
                    tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                    # print("gpu ", i, tasks[-1].name)
                    tasks[-1].start()

        ## Forward: GPU -1
        with torch.profiler.record_function("Forward: Block {}".format(self.layer_num-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask

            Xs.append(((h[0] + h[1])/2).detach())
            layer_fs.append((h[0] + h[1])/2)

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)

        ## Loss,Backward,Update: GPU -1
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(self.layer_num-1)):
            args = ([self.model[-2], self.model[-1]], self.opts[-1].optimizer, [layer_fs[-2], layer_fs[-1]], true_Ys[-1])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # GPU -1
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                # print("gpu -1", tasks[-1].name)
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

        with torch.profiler.record_function("Y to gpu"):
            true_Y = Y.to(self.gpus[-1])

        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.gpus[0], non_blocking=True))
            hat_Y, _, _= self.model[0](Xs[-1]) # Return output, hidden, mask
            Xs.append(hat_Y.to(self.gpus[1]))
            hidden.append(None)

        for i in range(1, self.layer_num-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                if i == 1:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1]) # Return output, hidden, mask
                else:
                    hat_Y, (h,c), _ = self.model[i](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
                Xs.append(hat_Y.to(self.gpus[i+1]))
                hidden.append(((h.to(self.gpus[i+1]), c.to(self.gpus[i+1]))))

        with torch.profiler.record_function("Forward: Block {}".format(self.layer_num-1)):
            hat_Y, (h,c), _ = self.model[-2](Xs[-1], hidden=hidden[-1]) # Return output, hidden, mask
            Xs.append(((h[0] + h[1])/2))

            hat_Y = self.model[-1](Xs[-1])
            layer_fs.append(hat_Y)

        return layer_fs[-1], true_Y

class Trans_BP_m_d(NLP_MultiGPU):
    def _init_data(self, configs):
        super()._init_data(configs)

        assert configs['layers'] >= 2, "Model layer setting error! The number of layers must be greater than 2."

        self.layer_num = configs['layers']
        self.gpu_num = set(self.gpus)
        # print("[Model] Trans_BP_Model - {0} layers".format(self.layer_num))

    def _init_model(self, configs):
        # Setting Model
        super()._init_model(configs)
        self.n_heads = configs["head"]

        self.layer_cfg = dict()

        # Embedding
        self.layer_cfg[0] = {
            "inp_dim":self.vocab_size, "out_dim":self.emb_dim, 
            "f":"emb", "h_dim":self.h_dim, 
            "word_vec":self.word_vec, "device":self.gpus[0]}

        # Transformer
        for i in range(1, self.layer_num):
            self.layer_cfg[i] = {
                "inp_dim":self.emb_dim, "out_dim":self.h_dim, 
                "f":"trans", "h_dim":self.h_dim, "n_heads":self.n_heads, 
                "word_vec":None, "device":self.gpus[i]}

        # Predict
        self.layer_cfg[self.layer_num] = {
            "inp_dim":self.emb_dim, "out_dim":self.num_classes, 
            "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.gpus[-1]}

        # Make Model
        self.model = []
        # Embedding and Transformer
        for i in range(self.layer_num):
            self.model.append(NLP_BP_Component(**self.layer_cfg[i]).to(self.gpus[i]))
        # Predict
        self.model.append(NLP_Loss_Predictor(**self.layer_cfg[self.layer_num]).to(self.gpus[-1]))

        self.model = torch.nn.Sequential(*self.model)

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
            true_Ys.append(Y.to(self.gpus[-1]))

        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.gpus[0], non_blocking=True))
            masks.append(mask.to(self.gpus[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

            Xs.append(hat_Y.to(self.gpus[1]))
            masks.append(mask.to(self.gpus[1]))

        for i in range(1, self.layer_num-1):
            ## Forward: GPU i
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.to(self.gpus[i+1]))
                masks.append(mask.to(self.gpus[i+1]))

        ## Forward: GPU -1
        with torch.profiler.record_function("Forward: Block {}".format(self.layer_num-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y)
            masks.append(mask)
            layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

            hat_Y = self.model[-1](layer_fs[-1])
            layer_fs.append(hat_Y)

        ## Loss,Backward,Update: GPU -1
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(self.layer_num-1)):
            args = ([self.model[-1]], self.opts[-1].optimizer, [layer_fs[-1]], true_Ys[-1])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # GPU -1
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
            true_Y = Y.to(self.gpus[-1])

        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.gpus[0], non_blocking=True))
            masks.append(mask.to(self.gpus[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            Xs.append(hat_Y.to(self.gpus[1]))
            masks.append(mask.to(self.gpus[1]))

        for i in range(1, self.layer_num-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.to(self.gpus[i+1]))
                masks.append(mask.to(self.gpus[i+1]))

        with torch.profiler.record_function("Forward: Block {}".format(self.layer_num-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

            Xs.append(hat_Y)
            masks.append(mask)
            layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

            hat_Y = self.model[-1](layer_fs[-1])
            layer_fs.append(hat_Y)

        return layer_fs[-1], true_Y

class Trans_SCPL_m_d(NLP_MultiGPU):
    def _init_data(self, configs):
        super()._init_data(configs)

        assert configs['layers'] >= 2, "Model layer setting error! The number of layers must be greater than 2."

        self.layer_num = configs['layers']
        self.gpu_num = set(self.gpus)
        # print("[Model] Trans_SCPL_Model - {0} layers".format(self.layer_num))

    def _init_model(self, configs):
        # Setting Model
        super()._init_model(configs)
        self.n_heads = configs["head"]
        self.proj_type = configs['proj_type']

        self.layer_cfg = dict()

        # Embedding
        self.layer_cfg[0] = {
            "inp_dim":self.vocab_size, "out_dim":self.emb_dim, 
            "f":"emb", "h_dim":self.emb_dim, 
            "word_vec":self.word_vec, "device":self.gpus[0], "proj_type":self.proj_type}

        # Transformer
        for i in range(1, self.layer_num):
            self.layer_cfg[i] = {
                "inp_dim":self.emb_dim, "out_dim":self.h_dim, 
                "f":"trans", "h_dim":self.emb_dim, "n_heads":self.n_heads, 
                "word_vec":None, "device":self.gpus[i], "proj_type":self.proj_type}

        # Predict
        self.layer_cfg[self.layer_num] = {
            "inp_dim":self.emb_dim, "out_dim":self.num_classes, 
            "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.gpus[-1]}

        # Make Model
        self.model = []
        # Embedding and Transformer
        for i in range(self.layer_num):
            self.model.append(NLP_LocalLoss_Component(**self.layer_cfg[i]).to(self.gpus[i]))
        # Predict
        self.model.append(NLP_Loss_Predictor(**self.layer_cfg[self.layer_num]).to(self.gpus[-1]))

        self.model = torch.nn.Sequential(*self.model)

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = list()

        self.opts.append(Optimizer(
            chain(self.model[0].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

        for i in range(1, self.layer_num-1):
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
            for i in range(self.layer_num):
                true_Ys.append(Y.to(self.gpus[i], non_blocking=True)) 

        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.gpus[0], non_blocking=True))
            masks.append(mask.to(self.gpus[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

            Xs.append(hat_Y.detach().to(self.gpus[1]))
            masks.append(mask.to(self.gpus[1]).detach())
            layer_fs.append(self.model[0].reduction(hat_Y, hidden, mask))

        ## Loss,Backward,Update: GPU 0
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(0)):
            args = ([self.model[0]], self.opts[0].optimizer, [layer_fs[-1]], true_Ys[0])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # GPU 0
                # loss_all += self._loss_backward_update(*args) # GPU 0
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                # print("gpu ", 0, tasks[-1].name)
                tasks[-1].start()

        for i in range(1, self.layer_num-1):
            ## Forward: GPU i
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.detach().to(self.gpus[i+1]))
                masks.append(mask.to(self.gpus[i+1]).detach())
                layer_fs.append(self.model[i].reduction(hat_Y, hidden, mask))

            ## Loss,Backward,Update: GPU i
            with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(i)):
                args = ([self.model[i]], self.opts[i].optimizer, [layer_fs[-1]], true_Ys[i])
                if not multi_t:
                    gpu_losses.append(self._loss_backward_update(*args)) # GPU i
                else:
                    tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                    tasks[-1].start()

        ## Forward: GPU -1
        with torch.profiler.record_function("Forward: Block {}".format(self.layer_num-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            
            Xs.append(hat_Y.detach())
            masks.append(mask.detach())
            layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

            hat_Y = self.model[-1](layer_fs[-1].detach())
            layer_fs.append(hat_Y)

        ## Loss,Backward,Update: GPU -1
        with torch.profiler.record_function("Loss,Backward,Update: Block {}".format(i)):
            args = ([self.model[-2], self.model[-1]], self.opts[-1].optimizer, [layer_fs[-2], layer_fs[-1]], true_Ys[-1])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # GPU -1
                # loss_all += self._loss_backward_update(*args) # GPU -1
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                # print("gpu ", -1, tasks[-1].name)
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

        with torch.profiler.record_function("Y to gpu"):
            true_Y = Y.to(self.gpus[-1])

        with torch.profiler.record_function("Forward: Block 0"):
            Xs.append(X.to(self.gpus[0], non_blocking=True))
            masks.append(mask.to(self.gpus[0], non_blocking=True))

            hat_Y, hidden, mask = self.model[0](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
            Xs.append(hat_Y.to(self.gpus[1]))
            masks.append(mask.to(self.gpus[1]))

        for i in range(1, self.layer_num-1):
            with torch.profiler.record_function("Forward: Block {}".format(i)):
                hat_Y, hidden, mask = self.model[i](Xs[-1], mask=masks[-1]) # Return output, hidden, mask
                
                Xs.append(hat_Y.to(self.gpus[i+1]))
                masks.append(mask.to(self.gpus[i+1]))

        with torch.profiler.record_function("Forward: Block {}".format(self.layer_num-1)):
            hat_Y, hidden, mask = self.model[-2](Xs[-1], mask=masks[-1]) # Return output, hidden, mask

            Xs.append(hat_Y)
            masks.append(mask)
            layer_fs.append(self.model[-2].reduction(hat_Y, hidden, mask))

            hat_Y = self.model[-1](layer_fs[-1])
            layer_fs.append(hat_Y)

        return layer_fs[-1], true_Y

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
               "f":"emb", "h_dim":self.h_dim, "word_vec":self.word_vec, "device":self.gpus[0], "proj_type":self.proj_type}, 
            1:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.gpus[1], "proj_type":self.proj_type},
            2:{"inp_dim":self.h_dim*2, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.gpus[2], "proj_type":self.proj_type},
            3:{"inp_dim":self.h_dim*2, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.gpus[3], "proj_type":self.proj_type},
            4:{"inp_dim":self.h_dim, "out_dim":self.num_classes, 
               "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.gpus[3]}}

        self.layer0 = NLP_LocalLoss_Component(**self.layer_cfg[0]).to(self.gpus[0])
        self.layer1 = NLP_LocalLoss_Component(**self.layer_cfg[1]).to(self.gpus[1])
        self.layer2 = NLP_LocalLoss_Component(**self.layer_cfg[2]).to(self.gpus[2])
        self.layer3 = NLP_LocalLoss_Component(**self.layer_cfg[3]).to(self.gpus[3])
        self.layer4 = NLP_Loss_Predictor(**self.layer_cfg[4]).to(self.gpus[3])

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
        
        true_y0 = Y.to(self.gpus[0], non_blocking=True)
        true_y1 = Y.to(self.gpus[1])
        true_y2 = Y.to(self.gpus[2])
        true_y3 = Y.to(self.gpus[3])

        for opt in self.opts:
            opt.zero_grad()
        
        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        x0 = X.to(self.gpus[0], non_blocking=True)
        hat_y0, _, _ = self.layer0(x0)
        
        x1 = hat_y0.detach().to(self.gpus[1])
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
        
        x2 = hat_y1.detach().to(self.gpus[2])
        hidden2 = ((h1.detach().to(self.gpus[2]), c1.detach().to(self.gpus[2])))

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

        x3 = hat_y2.detach().to(self.gpus[3])
        hidden3 = ((h2.detach().to(self.gpus[3]), c2.detach().to(self.gpus[3])))
        # x3 = h_state2.detach().to(self.gpus[3])
        
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
        
        # x4 = hat_y3.detach().to(self.gpus[3])
        # hidden4 = ((h3.detach().to(self.gpus[3]), c3.detach().to(self.gpus[3])))
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
        x0 = X.to(self.gpus[0], non_blocking=True)
        true_y3 = Y.to(self.gpus[3])
    
        hat_y0, _, _ = self.layer0(x0)
        x1 = hat_y0.to(self.gpus[1])
            
        hat_y1, (h1, c1), _ = self.layer1(x1) 
        x2 = hat_y1.to(self.gpus[2])
        hidden2 = (h1.to(self.gpus[2]), c1.to(self.gpus[2])) 
            
        hat_y2, (h2, c2), _ = self.layer2(x2, hidden=hidden2) 
        x3 = hat_y2.to(self.gpus[3])
        hidden3 = (h2.to(self.gpus[3]), c2.to(self.gpus[3])) 
            
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
            "h_dim":self.emb_dim, "word_vec":self.word_vec, "device":self.gpus[0], "proj_type":self.proj_type}, 
            1:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.gpus[1], "proj_type":self.proj_type},
            2:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.gpus[2], "proj_type":self.proj_type},
            3:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.gpus[3], "proj_type":self.proj_type},
            4:{"inp_dim":self.emb_dim, "out_dim":self.num_classes, 
            "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.gpus[3]}}

        self.layer0 = NLP_LocalLoss_Component(**self.layer_cfg[0]).to(self.gpus[0])
        self.layer1 = NLP_LocalLoss_Component(**self.layer_cfg[1]).to(self.gpus[1])
        self.layer2 = NLP_LocalLoss_Component(**self.layer_cfg[2]).to(self.gpus[2])
        self.layer3 = NLP_LocalLoss_Component(**self.layer_cfg[3]).to(self.gpus[3])
        self.layer4 = NLP_Loss_Predictor(**self.layer_cfg[4]).to(self.gpus[3])

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

        true_y0 = Y.to(self.gpus[0], non_blocking=True)
        true_y1 = Y.to(self.gpus[1])
        true_y2 = Y.to(self.gpus[2])
        true_y3 = Y.to(self.gpus[3])

        for opt in self.opts:
            opt.zero_grad()
        
        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        x0 = X.to(self.gpus[0], non_blocking=True)
        mask0 = mask.to(self.gpus[0], non_blocking=True)
        out0, hidden0, mask0 = self.layer0(x0, mask=mask0)
        feature0 = self.layer0.reduction(out0, hidden0, mask0)
        
        x1 = out0.detach().to(self.gpus[1])
        mask1 = mask0.detach().to(self.gpus[1])
        
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
        
        x2 = out1.detach().to(self.gpus[2])
        mask2 = mask1.detach().to(self.gpus[2])

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
        
        x3 = out2.detach().to(self.gpus[3])
        mask3 = mask2.detach().to(self.gpus[3])
        
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

        mask0 = mask.to(self.gpus[0], non_blocking=True)
        mask1 = mask.to(self.gpus[1])
        mask2 = mask.to(self.gpus[2])
        mask3 = mask.to(self.gpus[3])

        x0 = X.to(self.gpus[0], non_blocking=True)
        true_y3 = Y.to(self.gpus[3])
    
        out0, hidden0, mask0 = self.layer0(x0, mask=mask0)
        x1 = out0.to(self.gpus[1])
            
        out1, hidden1, mask1= self.layer1(x1, mask=mask1) 
        x2 = out1.to(self.gpus[2])
            
        out2, hidden2, mask2 = self.layer2(x2, mask=mask2) 
        x3 = out2.to(self.gpus[3])
            
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
        self.gpus = configs["gpus"]
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
               "f":"emb", "h_dim":self.h_dim, "word_vec":self.word_vec, "device":self.gpus[0], "proj_type":self.proj_type}, 
            1:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.gpus[1], "proj_type":self.proj_type},
            2:{"inp_dim":self.h_dim*2, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.gpus[2], "proj_type":self.proj_type},
            3:{"inp_dim":self.h_dim*2, "out_dim":self.h_dim, 
               "f":"lstm", "h_dim":self.h_dim, "word_vec":None, "device":self.gpus[3], "proj_type":self.proj_type},
            4:{"inp_dim":self.h_dim, "out_dim":self.num_classes, 
               "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.gpus[3]}}

        self.layer0 = NLP_LocalLoss_Component(**self.layer_cfg[0]).to(self.gpus[0])
        self.layer1 = NLP_LocalLoss_Component(**self.layer_cfg[1]).to(self.gpus[1])
        self.layer2 = NLP_LocalLoss_Component(**self.layer_cfg[2]).to(self.gpus[2])
        self.layer3 = NLP_LocalLoss_Component(**self.layer_cfg[3]).to(self.gpus[3])
        self.layer4 = NLP_Loss_Predictor(**self.layer_cfg[4]).to(self.gpus[3])

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
        
        true_y0 = Y.to(self.gpus[0], non_blocking=True)
        true_y1 = Y.to(self.gpus[1])
        true_y2 = Y.to(self.gpus[2])
        true_y3 = Y.to(self.gpus[3])

        for opt in self.opts:
            opt.zero_grad()
        
        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        x0 = X.to(self.gpus[0], non_blocking=True)
        hat_y0, _, _ = self.layer0(x0)
        
        x1 = hat_y0.detach().to(self.gpus[1])
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
        
        x2 = hat_y1.detach().to(self.gpus[2])
        hidden2 = ((h1.detach().to(self.gpus[2]), c1.detach().to(self.gpus[2])))

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

        x3 = hat_y2.detach().to(self.gpus[3])
        hidden3 = ((h2.detach().to(self.gpus[3]), c2.detach().to(self.gpus[3])))
        # x3 = h_state2.detach().to(self.gpus[3])
        
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
        
        # x4 = hat_y3.detach().to(self.gpus[3])
        # hidden4 = ((h3.detach().to(self.gpus[3]), c3.detach().to(self.gpus[3])))
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
        x0 = X.to(self.gpus[0], non_blocking=True)
        true_y3 = Y.to(self.gpus[3])
    
        hat_y0, _, _ = self.layer0(x0)
        x1 = hat_y0.to(self.gpus[1])
            
        hat_y1, (h1, c1), _ = self.layer1(x1) 
        x2 = hat_y1.to(self.gpus[2])
        hidden2 = (h1.to(self.gpus[2]), c1.to(self.gpus[2])) 
            
        hat_y2, (h2, c2), _ = self.layer2(x2, hidden=hidden2) 
        x3 = hat_y2.to(self.gpus[3])
        hidden3 = (h2.to(self.gpus[3]), c2.to(self.gpus[3])) 
            
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
        self.gpus = configs["gpus"]
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
            "h_dim":self.emb_dim, "word_vec":self.word_vec, "device":self.gpus[0], "proj_type":self.proj_type}, 
            1:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.gpus[1], "proj_type":self.proj_type},
            2:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.gpus[2], "proj_type":self.proj_type},
            3:{"inp_dim":self.emb_dim, "out_dim":self.h_dim, "f":"trans", 
            "h_dim":self.emb_dim, "n_heads":self.n_heads, "word_vec":None, "device":self.gpus[3], "proj_type":self.proj_type},
            4:{"inp_dim":self.emb_dim, "out_dim":self.num_classes, 
            "hid_dim":self.h_dim, "act_fun":nn.Tanh(), "device":self.gpus[3]}}

        self.layer0 = NLP_LocalLoss_Component(**self.layer_cfg[0]).to(self.gpus[0])
        self.layer1 = NLP_LocalLoss_Component(**self.layer_cfg[1]).to(self.gpus[1])
        self.layer2 = NLP_LocalLoss_Component(**self.layer_cfg[2]).to(self.gpus[2])
        self.layer3 = NLP_LocalLoss_Component(**self.layer_cfg[3]).to(self.gpus[3])
        self.layer4 = NLP_Loss_Predictor(**self.layer_cfg[4]).to(self.gpus[3])

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

        true_y0 = Y.to(self.gpus[0], non_blocking=True)
        true_y1 = Y.to(self.gpus[1])
        true_y2 = Y.to(self.gpus[2])
        true_y3 = Y.to(self.gpus[3])

        for opt in self.opts:
            opt.zero_grad()
        
        # Forward: GPU 0 ~ 3
        ## Forward: GPU 0
        x0 = X.to(self.gpus[0], non_blocking=True)
        mask0 = mask.to(self.gpus[0], non_blocking=True)
        out0, hidden0, mask0 = self.layer0(x0, mask=mask0)
        feature0 = self.layer0.reduction(out0, hidden0, mask0)
        
        x1 = out0.detach().to(self.gpus[1])
        mask1 = mask0.detach().to(self.gpus[1])
        
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
        
        x2 = out1.detach().to(self.gpus[2])
        mask2 = mask1.detach().to(self.gpus[2])

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
        
        x3 = out2.detach().to(self.gpus[3])
        mask3 = mask2.detach().to(self.gpus[3])
        
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

        mask0 = mask.to(self.gpus[0], non_blocking=True)
        mask1 = mask.to(self.gpus[1])
        mask2 = mask.to(self.gpus[2])
        mask3 = mask.to(self.gpus[3])

        x0 = X.to(self.gpus[0], non_blocking=True)
        true_y3 = Y.to(self.gpus[3])
    
        out0, hidden0, mask0 = self.layer0(x0, mask=mask0)
        x1 = out0.to(self.gpus[1])
            
        out1, hidden1, mask1= self.layer1(x1, mask=mask1) 
        x2 = out1.to(self.gpus[2])
            
        out2, hidden2, mask2 = self.layer2(x2, mask=mask2) 
        x3 = out2.to(self.gpus[3])
            
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
