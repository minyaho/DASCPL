import torch
import torch.nn as nn
from utils import Optimizer, CPUThread, MultiGPUModel
from itertools import chain
from collections import OrderedDict
from .vision_single import VGG_Block, VGG_Predictor, resnet18_Head, resnet18_Block, resnet18_Predictor
# from transformer.encoder import TransformerEncoder

class Vision_MultiGPU(MultiGPUModel):
    def __init__(self, configs):
        super(Vision_MultiGPU, self).__init__()

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
        self.num_classes = configs["n_classes"]
        self.train_loader = configs["train_loader"]
        self.test_loader = configs["test_loader"]
        self.aug_type = configs["aug_type"]
        self.proj_type = None
        self.pred_type = None
        self.num_layers = configs['layers']
        assert self.num_layers == 4, "The number of layers for vision model must be 4."

    def _init_model(self, configs):
        pass

    def _init_optimizers(self, configs):
        # Optimizers
        self.base_lr = configs["base_lr"]
        self.end_lr = configs["end_lr"]
        self.max_step = configs["max_steps"]
        self.global_steps = 0

    def forward(self, X, Y, multi_t=True):

        if self.training:
            return self.train_step(X, Y, multi_t)
        else:
            return self.inference(X, Y)
        
    def _shape_div_2(self):
        self.shape //= 2
        return self.shape

class VGG_BP_m(Vision_MultiGPU):
    def __init__(self, configs):
        super(VGG_BP_m, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.pred_type = None

    def _init_model(self, configs):
        self.shape = 32
        self.in_channels = 3
        self.layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}

        self.model_cfg = {
            'backbone-0': {
                "cfg":self.layer_cfg[0], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":3,"pred_type":self.pred_type, "device":self.devices[0]},
            'backbone-1': {
                "cfg":self.layer_cfg[1], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[0][-2], "pred_type":self.pred_type, "device":self.devices[1]},
            'backbone-2': {
                "cfg":self.layer_cfg[2], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[1][-2], "pred_type":self.pred_type, "device":self.devices[2]},
            'backbone-3': {
                "cfg":self.layer_cfg[3], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[2][-2], "pred_type":self.pred_type, "device":self.devices[3]},
            'predictor': {
                "shape":self.shape, "num_classes":self.num_classes,"in_channels":self.layer_cfg[3][-2], "device":self.devices[3]}
        }

        # Make Model
        self.model = []
        for key, values in self.model_cfg.items():
            if 'backbone' in key:
                self.model.append((key, VGG_Block(**values).to(values["device"])))
            elif 'predictor' in key:
                self.model.append((key, VGG_Predictor(**values).to(values["device"])))
        self.model = torch.nn.Sequential(OrderedDict(self.model))

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = list()

        self.opts.append(Optimizer(
            chain(self.model.parameters()), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

    def train_step(self, X, Y, multi_t=True):
        self.global_steps += 1
        tasks = list()
        gpu_losses = list()
        loss_all = 0
    
        true_y3 = Y.to(self.devices[3], non_blocking=True)

        for layer in self.model:
            layer.train()
    
        for opt in self.opts:
            opt.zero_grad()
        
        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0 = self.model[0](x0)       # Block 0
        x1 = hat_y0.to(self.devices[1])
    
        ## Forward: Block 1
        hat_y1 = self.model[1](x1)       # Block 1 
        x2 = hat_y1.to(self.devices[2])
    
        ## Forward: Block 2
        hat_y2 = self.model[2](x2)       # Block 2
        x3 = hat_y2.to(self.devices[3])

        ## Forward: Block 3
        hat_y3 = self.model[3](x3)       # Block 3
        
        x4 = hat_y3
        hat_y4 = self.model[4](x4)       # Block 3
    
        ## Loss,Backward,Update: Block 3
        args = ([self.model[4]], self.opts[-1].optimizer, [hat_y4], [true_y3])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 3

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
        
        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
        
        return [hat_y4], loss_all
    
    def inference(self, X, Y):
        true_y3 = Y.to(self.devices[3], non_blocking=True)

        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0 = self.model[0](x0)       # Block 0

        x1 = hat_y0.to(self.devices[1])
        hat_y1 = self.model[1](x1)       # Block 1 

        x2 = hat_y1.to(self.devices[2])
        hat_y2 = self.model[2](x2)       # Block 2

        x3 = hat_y2.to(self.devices[3])
        hat_y3 = self.model[3](x3)       # Block 3

        x4 = hat_y3
        hat_y4 = self.model[4](x4)       # Block 3

        return [hat_y4], [true_y3]

class VGG_BP_p_m(VGG_BP_m):
    def __init__(self, configs):
        super(VGG_BP_p_m, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.proj_type = None
        self.pred_type = configs['pred_type']

    def train_step(self, X, Y, multi_t=True):
        self.global_steps += 1
        tasks = list()
        gpu_losses = list()
        loss_all = 0
    
        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for layer in self.model:
            layer.train()
    
        for opt in self.opts:
            opt.zero_grad()
        
        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0 = self.model[0](x0)       # Block 0
        x1 = hat_y0.to(self.devices[1])
    
        ## Forward: Block 1
        hat_y1 = self.model[1](x1)       # Block 1 
        x2 = hat_y1.to(self.devices[2])
    
        ## Forward: Block 2
        hat_y2 = self.model[2](x2)       # Block 2
        x3 = hat_y2.to(self.devices[3])

        ## Forward: Block 3
        hat_y3 = self.model[3](x3)       # Block 3
        
        x4 = hat_y3
        hat_y4 = self.model[4](x4)       # Block 3
    
        ## Loss,Backward,Update: Block 3
        args = (
            [self.model[i] for i in range(self.num_layers+1)], 
            self.opts[-1].optimizer, 
            [hat_y0, hat_y1, hat_y2, hat_y3, hat_y4], 
            [true_y0, true_y1, true_y2, true_y3, true_y3]
        )
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 3
        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
        
        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
        
        return [hat_y4], loss_all
    
    def inference(self, X, Y):
        layer_fs = list()
        true_Ys = list()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True))
        true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0 = self.model[0](x0)       # Block 0
        layer_fs.append(self.model[0].loss(hat_y0))

        x1 = hat_y0.to(self.devices[1])
        hat_y1 = self.model[1](x1)       # Block 1 
        layer_fs.append(self.model[1].loss(hat_y1))

        x2 = hat_y1.to(self.devices[2])
        hat_y2 = self.model[2](x2)       # Block 2
        layer_fs.append(self.model[2].loss(hat_y2))

        x3 = hat_y2.to(self.devices[3])
        hat_y3 = self.model[3](x3)       # Block 3
        layer_fs.append(self.model[3].loss(hat_y3))

        x4 = hat_y3
        hat_y4 = self.model[4](x4)       # Predict Block
        layer_fs.append(hat_y4)

        return layer_fs, true_Ys

class VGG_SCPL_m(Vision_MultiGPU):
    def __init__(self, configs):
        super(VGG_SCPL_m, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.proj_type = configs['proj_type']

    def _init_model(self, configs):
        self.shape = 32
        self.in_channels = 3
        self.layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}

        self.model_cfg = {
            'backbone-0': {
                "cfg":self.layer_cfg[0], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":3,"proj_type":self.proj_type,"pred_type":self.pred_type,"device":self.devices[0]},
            'backbone-1': {
                "cfg":self.layer_cfg[1], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[0][-2],"proj_type":self.proj_type,"pred_type":self.pred_type,"device":self.devices[1]},
            'backbone-2': {
                "cfg":self.layer_cfg[2], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[1][-2],"proj_type":self.proj_type,"pred_type":self.pred_type,"device":self.devices[2]},
            'backbone-3': {
                "cfg":self.layer_cfg[3], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[2][-2],"proj_type":self.proj_type,"pred_type":self.pred_type,"device":self.devices[3]},
            'predictor': {
                "shape":self.shape, "num_classes":self.num_classes,"in_channels":self.layer_cfg[3][-2], "device":self.devices[3]}
        }

        # Make Model
        self.model = []
        for key, values in self.model_cfg.items():
            if 'backbone' in key:
                self.model.append((key, VGG_Block(**values).to(values["device"])))
            elif 'predictor' in key:
                self.model.append((key, VGG_Predictor(**values).to(values["device"])))
        self.model = torch.nn.Sequential(OrderedDict(self.model))

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = [
            Optimizer(chain(self.model[0].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step),
            Optimizer(chain(self.model[1].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step),
            Optimizer(chain(self.model[2].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step),
            Optimizer(chain(self.model[3].parameters(), self.model[4].parameters()), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step),
        ]

    def train_step(self, X, Y, multi_t=True):
        self.global_steps += 1
        tasks = list()
        gpu_losses = list()
        loss_all = 0

        for layer in self.model:
            layer.train()
    
        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for opt in self.opts:
            opt.zero_grad()

        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0 = self.model[0](x0)       # Block 0
        x1 = hat_y0.detach().to(self.devices[1])
    
        ## Loss,Backward,Update: Block 0
        args = ([self.model[0]], self.opts[0].optimizer, [hat_y0], [true_y0])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 0

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
        
        ## Forward: Block 1
        hat_y1 = self.model[1](x1)       # Block 1 
        x2 = hat_y1.detach().to(self.devices[2])
    
        ## Loss,Backward,Update: Block 1
        args = ([self.model[1]], self.opts[1].optimizer, [hat_y1], [true_y1])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 1

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
    
        ## Forward: Block 2
        hat_y2 = self.model[2](x2)       # Block 2
        x3 = hat_y2.detach().to(self.devices[3])
    
        ## Loss,Backward,Update: Block 2
        args = ([self.model[2]], self.opts[2].optimizer, [hat_y2], [true_y2])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 2

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()

        ## Forward: Block 3
        hat_y3 = self.model[3](x3)       # Block 3
        
        x4 = hat_y3.detach()
        hat_y4 = self.model[4](x4)       # Block 3
    
        ## Loss,Backward,Update: Block 3
        args = ([self.model[3], self.model[4]], self.opts[3].optimizer, [hat_y3, hat_y4], [true_y3]*2)
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 3

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
        
        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()

        return hat_y4, loss_all

    def inference(self, X, Y):
        true_y3 = Y.to(self.devices[3], non_blocking=True)

        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0 = self.model[0](x0)       # Block 0

        x1 = hat_y0.to(self.devices[1])
        hat_y1 = self.model[1](x1)       # Block 1 

        x2 = hat_y1.to(self.devices[2])
        hat_y2 = self.model[2](x2)       # Block 2

        x3 = hat_y2.to(self.devices[3])
        hat_y3 = self.model[3](x3)       # Block 3

        x4 = hat_y3
        hat_y4 = self.model[4](x4)       # Block 3

        return [hat_y4], [true_y3]

class VGG_DASCPL_m(VGG_SCPL_m):
    def __init__(self, configs):
        super(VGG_DASCPL_m, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.proj_type = configs['proj_type']
        self.pred_type = configs['pred_type']

    def inference(self, X, Y):
        layer_fs = list()
        true_Ys = list()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True))
        true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0 = self.model[0](x0)       # Block 0
        layer_fs.append(self.model[0].loss(hat_y0))

        x1 = hat_y0.to(self.devices[1])
        hat_y1 = self.model[1](x1)       # Block 1 
        layer_fs.append(self.model[1].loss(hat_y1))

        x2 = hat_y1.to(self.devices[2])
        hat_y2 = self.model[2](x2)       # Block 2
        layer_fs.append(self.model[2].loss(hat_y2))

        x3 = hat_y2.to(self.devices[3])
        hat_y3 = self.model[3](x3)       # Block 3
        layer_fs.append(self.model[3].loss(hat_y3))

        x4 = hat_y3
        hat_y4 = self.model[4](x4)       # Predict Block
        layer_fs.append(hat_y4)

        return layer_fs, true_Ys

class resnet18_BP_m(Vision_MultiGPU):
    def __init__(self, configs):
        super(resnet18_BP_m, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.pred_type = None

    def _init_model(self, configs):
        self.shape = 32
        self.in_channels = 3
        self.layer_cfg = {0:[64, 64, [1, 1]], 1:[64, 128, [2, 1]], 2:[128, 256, [2, 1]], 3:[256, 512, [2, 1]]}

        self.model_cfg = {
            'head': {"device": self.devices[0]},
            'backbone-0': {
                "cfg":self.layer_cfg[0], "shape":self.shape, "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[0][1], "pred_type":self.pred_type, "device":self.devices[0]},
            'backbone-1': {
                "cfg":self.layer_cfg[1], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[1][1], "pred_type":self.pred_type, "device":self.devices[1]},
            'backbone-2': {
                "cfg":self.layer_cfg[2], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[2][1], "pred_type":self.pred_type, "device":self.devices[2]},
            'backbone-3': {
                "cfg":self.layer_cfg[3], "shape":1, "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[3][1], "pred_type":self.pred_type,
                "avg_pool":torch.nn.AdaptiveAvgPool2d((1, 1)), "device":self.devices[3]},
            'predictor': {
                "num_classes":self.num_classes, "device":self.devices[3]}
        }

        # Make Model
        self.model = []
        for key, values in self.model_cfg.items():
            if 'backbone' in key:
                self.model.append((key, resnet18_Block(**values).to(values["device"])))
            elif 'head' in key: 
                self.model.append((key, resnet18_Head(**values).to(values["device"])))
            elif 'predictor' in key:
                self.model.append((key, resnet18_Predictor(**values).to(values["device"])))
        self.model = torch.nn.Sequential(OrderedDict(self.model))

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = list()

        self.opts.append(Optimizer(
            chain(self.model.parameters()), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

    def train_step(self, X, Y, multi_t=True):
        self.global_steps += 1
        tasks = list()
        gpu_losses = list()
        loss_all = 0

        true_y3 = Y.to(self.devices[3], non_blocking=True)

        for layer in self.model:
            layer.train()
    
        for opt in self.opts:
            opt.zero_grad()
    
        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        x0 = X.to(self.devices[0], non_blocking=True)
        x1 = self.model[0](x0)           # Block 0

        hat_y1 = self.model[1](x1)       # Block 0
        x2 = hat_y1.to(self.devices[1])

        ## Forward: Block 1
        hat_y2 = self.model[2](x2)       # Block 1 
        x3 = hat_y2.to(self.devices[2])

        ## Forward: Block 2
        hat_y3 = self.model[3](x3)       # Block 2
        x4 = hat_y3.to(self.devices[3])

        ## Forward: Block 3
        hat_y4 = self.model[4](x4)       # Block 3
        
        x5 = hat_y4
        hat_y5 = self.model[5](x5)       # Block 3

        ## Loss,Backward,Update: Block 3
        args = ([self.model[5]], self.opts[-1].optimizer, [hat_y5], [true_y3])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 3

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
    
        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
            
        return hat_y5, loss_all

    def inference(self, X, Y):
        true_y3 = Y.to(self.devices[3], non_blocking=True)

        x0 = X.to(self.devices[0], non_blocking=True)
        x1 = self.model[0](x0)           # Block 0
        hat_y1 = self.model[1](x1)       # Block 0

        x2 = hat_y1.to(self.devices[1])
        hat_y2 = self.model[2](x2)       # Block 1

        x3 = hat_y2.to(self.devices[2])
        hat_y3 = self.model[3](x3)       # Block 2

        x4 = hat_y3.to(self.devices[3])
        x5 = self.model[4](x4)           # Block 3
        hat_y5 = self.model[5](x5)       # Block 3
        
        return [hat_y5], [true_y3]

class resnet18_BP_p_m(resnet18_BP_m):
    def __init__(self, configs):
        super(resnet18_BP_p_m, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.proj_type = None
        self.pred_type = configs['pred_type']

    def train_step(self, X, Y, multi_t=True):
        self.global_steps += 1
        tasks = list()
        gpu_losses = list()
        loss_all = 0

        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for layer in self.model:
            layer.train()
    
        for opt in self.opts:
            opt.zero_grad()
    
        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        x0 = X.to(self.devices[0], non_blocking=True)
        x1 = self.model[0](x0)           # Block 0

        hat_y1 = self.model[1](x1)       # Block 0
        x2 = hat_y1.to(self.devices[1])

        ## Forward: Block 1
        hat_y2 = self.model[2](x2)       # Block 1 
        x3 = hat_y2.to(self.devices[2])

        ## Forward: Block 2
        hat_y3 = self.model[3](x3)       # Block 2
        x4 = hat_y3.to(self.devices[3])

        ## Forward: Block 3
        hat_y4 = self.model[4](x4)       # Block 3
        
        x5 = hat_y4
        hat_y5 = self.model[5](x5)       # Block 3

        ## Loss,Backward,Update: Block 3
        args = (
            [self.model[1], self.model[2], self.model[3], self.model[4], self.model[5]], 
            self.opts[-1].optimizer, 
            [hat_y1, hat_y2, hat_y3, hat_y4, hat_y5], 
            [true_y0, true_y1, true_y2, true_y3, true_y3]
        )

        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 3

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
    
        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
            
        return hat_y5, loss_all

    def inference(self, X, Y):
        layer_fs = list()
        true_Ys = list()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True))
        true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        x0 = X.to(self.devices[0], non_blocking=True)
        x1 = self.model[0](x0)           # Block 0
        hat_y1 = self.model[1](x1)       # Block 0
        layer_fs.append(self.model[1].loss(hat_y1))

        x2 = hat_y1.to(self.devices[1])
        hat_y2 = self.model[2](x2)       # Block 1
        layer_fs.append(self.model[2].loss(hat_y2))

        x3 = hat_y2.to(self.devices[2])
        hat_y3 = self.model[3](x3)       # Block 2
        layer_fs.append(self.model[3].loss(hat_y3))

        x4 = hat_y3.to(self.devices[3])
        x5 = self.model[4](x4)           # Block 3
        layer_fs.append(self.model[4].loss(x5))

        hat_y5 = self.model[5](x5)       # Predict Block
        layer_fs.append(hat_y5)
        
        return layer_fs, true_Ys

class resnet18_SCPL_m(Vision_MultiGPU):
    def __init__(self, configs):
        super(resnet18_SCPL_m, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.proj_type = configs['proj_type']

    def _init_model(self, configs):
        self.shape = 32
        self.in_channels = 3
        self.layer_cfg = {0:[64, 64, [1, 1]], 1:[64, 128, [2, 1]], 2:[128, 256, [2, 1]], 3:[256, 512, [2, 1]]}

        self.model_cfg = {
            'head': {"device": self.devices[0]},
            'backbone-0': {
                "cfg":self.layer_cfg[0], "shape":self.shape, "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[0][1], "proj_type":self.proj_type, "pred_type":self.pred_type, "device":self.devices[0]},
            'backbone-1': {
                "cfg":self.layer_cfg[1], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[1][1], "proj_type":self.proj_type, "pred_type":self.pred_type, "device":self.devices[1]},
            'backbone-2': {
                "cfg":self.layer_cfg[2], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[2][1], "proj_type":self.proj_type, "pred_type":self.pred_type, "device":self.devices[2]},
            'backbone-3': {
                "cfg":self.layer_cfg[3], "shape":1, "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[3][1], "proj_type":self.proj_type, "pred_type":self.pred_type,
                "avg_pool":torch.nn.AdaptiveAvgPool2d((1, 1)), "device":self.devices[3]},
            'predictor': {
                "num_classes":self.num_classes, "device":self.devices[3]}
        }

        # Make Model
        self.model = []
        for key, values in self.model_cfg.items():
            if 'backbone' in key:
                self.model.append((key, resnet18_Block(**values).to(values["device"])))
            elif 'head' in key: 
                self.model.append((key, resnet18_Head(**values).to(values["device"])))
            elif 'predictor' in key:
                self.model.append((key, resnet18_Predictor(**values).to(values["device"])))
        self.model = torch.nn.Sequential(OrderedDict(self.model))

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = [
            Optimizer(chain(self.model[0].parameters(), self.model[1].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step),
            Optimizer(chain(self.model[2].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step),
            Optimizer(chain(self.model[3].parameters(),), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step),
            Optimizer(chain(self.model[4].parameters(), self.model[5].parameters()), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step),
        ]

    def train_step(self, X, Y, multi_t=True):
        self.global_steps += 1
        tasks = list()
        gpu_losses = list()
        loss_all = 0

        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for layer in self.model:
            layer.train()

        for opt in self.opts:
            opt.zero_grad()

        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        x0 = X.to(self.devices[0], non_blocking=True)
        x1 = self.model[0](x0)       # Block 0

        hat_y1 = self.model[1](x1)       # Block 0
        x2 = hat_y1.detach().to(self.devices[1])

        ## Loss,Backward,Update: Block 0
        args = ([self.model[1]], self.opts[0].optimizer, [hat_y1], [true_y0])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 0

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
    
        ## Forward: GPU 1
        hat_y2 = self.model[2](x2)       # Block 1 
        x3 = hat_y2.detach().to(self.devices[2])
        
        ## Loss,Backward,Update: Block 1
        args = ([self.model[2]], self.opts[1].optimizer, [hat_y2], [true_y1])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 1

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()

        ## Forward: Block 2
        hat_y3 = self.model[3](x3)       # Block 2
        x4 = hat_y3.detach().to(self.devices[3])
        
        ## Loss,Backward,Update: Block 2
        args = ([self.model[3]], self.opts[2].optimizer, [hat_y3], [true_y2])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 2

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()

        ## Forward: Block 3
        hat_y4 = self.model[4](x4)       # Block 3
        
        x5 = hat_y4.detach()
        hat_y5 = self.model[5](x5)       # Block 3

        ## Loss,Backward,Update: Block 3
        args = ([self.model[4], self.model[5]], self.opts[3].optimizer, [hat_y4, hat_y5], [true_y3]*2)
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 3

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()

        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
    
        return hat_y4, loss_all

    def inference(self, X, Y):
        true_y3 = Y.to(self.devices[3], non_blocking=True)

        x0 = X.to(self.devices[0], non_blocking=True)
        x1 = self.model[0](x0)           # Block 0
        hat_y1 = self.model[1](x1)       # Block 0

        x2 = hat_y1.to(self.devices[1])
        hat_y2 = self.model[2](x2)       # Block 1

        x3 = hat_y2.to(self.devices[2])
        hat_y3 = self.model[3](x3)       # Block 2

        x4 = hat_y3.to(self.devices[3])
        x5 = self.model[4](x4)           # Block 3
        hat_y5 = self.model[5](x5)       # Block 3
        
        return [hat_y5], [true_y3]

class resnet18_DASCPL_m(resnet18_SCPL_m):
    def __init__(self, configs):
        super(resnet18_DASCPL_m, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.proj_type = configs['proj_type']
        self.pred_type = configs['pred_type']

    def inference(self, X, Y):
        layer_fs = list()
        true_Ys = list()

        for i in range(self.num_layers):
            true_Ys.append(Y.to(self.devices[i], non_blocking=True))
        true_Ys.append(Y.to(self.devices[-1], non_blocking=True))  # For predictor loss

        x0 = X.to(self.devices[0], non_blocking=True)
        x1 = self.model[0](x0)           # Block 0
        hat_y1 = self.model[1](x1)       # Block 0
        layer_fs.append(self.model[1].loss(hat_y1))

        x2 = hat_y1.to(self.devices[1])
        hat_y2 = self.model[2](x2)       # Block 1
        layer_fs.append(self.model[2].loss(hat_y2))

        x3 = hat_y2.to(self.devices[2])
        hat_y3 = self.model[3](x3)       # Block 2
        layer_fs.append(self.model[3].loss(hat_y3))

        x4 = hat_y3.to(self.devices[3])
        x5 = self.model[4](x4)           # Block 3
        layer_fs.append(self.model[4].loss(x5))

        hat_y5 = self.model[5](x5)       # Predict Block
        layer_fs.append(hat_y5)
        
        return layer_fs, true_Ys