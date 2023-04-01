import torch
import torch.nn as nn
from utils import Optimizer, CPUThread, ProfilerMultiGPUModel
from itertools import chain
from .VGG import VGG_SCPL_Component, VGG_SCPL_Tail, VGG_BP_Component, VGG_Tail
from .ResNet import resnet18_Head, resnet18_Tail, resnet18_SCPL_Component, resnet18_BP_Component
# from transformer.encoder import TransformerEncoder
import time

class Vision_MultiGPU(ProfilerMultiGPUModel):
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
        self.model_type = 'image'
        self.gpus = configs["gpus"]
        self.dataset = configs["dataset"]
        self.num_classes = configs["n_classes"]
        self.train_loader = configs["train_loader"]
        self.test_loader = configs["test_loader"]
        self.aug_type = configs["aug_type"]
        self.save_path = configs["save_path"] if configs["save_path"] != None else './{}'.format(self.__class__.__name__)

    def _init_model(self, configs):
        pass

    def _init_optimizers(self, configs):
        # Optimizers
        self.base_lr = configs["base_lr"]
        self.end_lr = configs["end_lr"]
        self.max_step = configs["max_steps"]
        self.global_steps = 0

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
    
    def _shape_div_2(self):
        self.shape //= 2
        return self.shape

class VGG_BP_m(Vision_MultiGPU):
    def _init_data(self, configs):
        # Data
        super()._init_data(configs)

    def _init_model(self, configs):
        self.shape = 32
        self.in_channels = 3
        self.layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}

        # Make Model
        self.model = []
        self.model.append(VGG_BP_Component(
            cfg=self.layer_cfg[0], shape=self._shape_div_2(), in_channels=3, device=self.gpus[0]).to(self.gpus[0]))
        self.model.append(VGG_BP_Component(
            cfg=self.layer_cfg[1], shape=self._shape_div_2(), in_channels=self.model[-1].in_channels, device=self.gpus[1]).to(self.gpus[1]))
        self.model.append(VGG_BP_Component(
            cfg=self.layer_cfg[2], shape=self._shape_div_2(), in_channels=self.model[-1].in_channels, device=self.gpus[2]).to(self.gpus[2]))
        self.model.append(VGG_BP_Component(
            cfg=self.layer_cfg[3], shape=self._shape_div_2(), in_channels=self.model[-1].in_channels, device=self.gpus[3]).to(self.gpus[3]))
        self.model.append(VGG_Tail(
            num_classes=self.num_classes, shape=self.shape, in_channels=self.model[-1].in_channels, device=self.gpus[3]).to(self.gpus[3]))

        self.model = torch.nn.Sequential(*self.model)

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = list()

        self.opts.append(Optimizer(
            chain(self.model.parameters()), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

    def train_step(self, X, Y, multi_t=True):
        with torch.profiler.record_function("Init"):
            self.global_steps += 1
            tasks = list()
            gpu_losses = list()
            loss_all = 0
        
        with torch.profiler.record_function("Y to gpu 3"):
            true_y3 = Y.to(self.gpus[3], non_blocking=True)

        with torch.profiler.record_function("train mode"):
            for layer in self.model:
                layer.train()
        
        with torch.profiler.record_function("opt - zero_grad"):
            for opt in self.opts:
                opt.zero_grad()
            
        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        with torch.profiler.record_function("Forward: Block 0"):
            x0 = X.to(self.gpus[0], non_blocking=True)
            hat_y0 = self.model[0](x0)       # Block 0
            x1 = hat_y0.to(self.gpus[1])
        
        with torch.profiler.record_function("Forward: Block 1"):
            ## Forward: Block 1
            hat_y1 = self.model[1](x1)       # Block 1 
            x2 = hat_y1.to(self.gpus[2])
        
        with torch.profiler.record_function("Forward: Block 2"):
            ## Forward: Block 2
            hat_y2 = self.model[2](x2)       # Block 2
            x3 = hat_y2.to(self.gpus[3])

        with torch.profiler.record_function("Forward: Block 3"):
            ## Forward: Block 3
            hat_y3 = self.model[3](x3)       # Block 3
            
            x4 = hat_y3
            hat_y4 = self.model[4](x4)       # Block 3
        
        with torch.profiler.record_function("Loss,Backward,Update: Block 3"):
            ## Loss,Backward,Update: Block 3
            args = ([self.model[4]], self.opts[-1].optimizer, [hat_y4], true_y3)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 3

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
            
        with torch.profiler.record_function("wait"):
            if multi_t:
                for t in range(len(tasks)):
                    loss_all += tasks[t].get_result().item()
            else:
                for loss in gpu_losses:
                    loss_all += loss.item()
        
        return hat_y4, loss_all
    
    def inference(self, X, Y):
        with torch.profiler.record_function("Y to gpu"):
            true_y3 = Y.to(self.gpus[3], non_blocking=True)

        with torch.profiler.record_function("Forward: Block 0"):
            x0 = X.to(self.gpus[0], non_blocking=True)
            hat_y0 = self.model[0](x0)       # Block 0

        with torch.profiler.record_function("Forward: Block 1"):
            x1 = hat_y0.to(self.gpus[1])
            hat_y1 = self.model[1](x1)       # Block 1 

        with torch.profiler.record_function("Forward: Block 2"):
            x2 = hat_y1.to(self.gpus[2])
            hat_y2 = self.model[2](x2)       # Block 2

        with torch.profiler.record_function("Forward: Block 3"):
            x3 = hat_y2.to(self.gpus[3])
            hat_y3 = self.model[3](x3)       # Block 3

            x4 = hat_y3
            hat_y4 = self.model[4](x4)       # Block 3

        return hat_y4, true_y3

class VGG_SCPL_m(Vision_MultiGPU):
    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.proj_type = configs['proj_type']

    def _init_model(self, configs):
        self.shape = 32
        self.in_channels = 3
        self.layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}

        # Make Model
        self.model = []
        self.model.append(VGG_SCPL_Component(cfg=self.layer_cfg[0], shape=self._shape_div_2(), 
                                             in_channels=3, proj_type=self.proj_type, device=self.gpus[0]).to(self.gpus[0]))
        self.model.append(VGG_SCPL_Component(cfg=self.layer_cfg[1], shape=self._shape_div_2(), in_channels=self.model[-1].in_channels, 
                                             proj_type=self.proj_type, device=self.gpus[1]).to(self.gpus[1]))
        self.model.append(VGG_SCPL_Component(cfg=self.layer_cfg[2], shape=self._shape_div_2(), in_channels=self.model[-1].in_channels, 
                                             proj_type=self.proj_type,device=self.gpus[2]).to(self.gpus[2]))
        self.model.append(VGG_SCPL_Component(cfg=self.layer_cfg[3], shape=self._shape_div_2(), in_channels=self.model[-1].in_channels,
                                             proj_type=self.proj_type, device=self.gpus[3]).to(self.gpus[3]))
        self.model.append(VGG_SCPL_Tail(num_classes=self.num_classes, shape=self.shape, in_channels=self.model[-1].in_channels, device=self.gpus[3]).to(self.gpus[3]))

        self.model = torch.nn.Sequential(*self.model)

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
        with torch.profiler.record_function("Init"):
            self.global_steps += 1
            tasks = list()
            gpu_losses = list()
            loss_all = 0

        with torch.profiler.record_function("train mode"):
            for layer in self.model:
                layer.train()
        
        with torch.profiler.record_function("Y to gpu"):
            true_y0 = Y.to(self.gpus[0], non_blocking=True)
            true_y1 = Y.to(self.gpus[1])
            true_y2 = Y.to(self.gpus[2])
            true_y3 = Y.to(self.gpus[3])
    
        with torch.profiler.record_function("opt - zero_grad"):
            for opt in self.opts:
                opt.zero_grad()
    
        # Forward: Block 0 ~ 3
        with torch.profiler.record_function("Forward: Block 0"):
            ## Forward: Block 0
            x0 = X.to(self.gpus[0], non_blocking=True)
            hat_y0 = self.model[0](x0)       # Block 0
            x1 = hat_y0.detach().to(self.gpus[1])
        
        with torch.profiler.record_function("Loss,Backward,Update: Block 0"):
            ## Loss,Backward,Update: Block 0
            args = ([self.model[0]], self.opts[0].optimizer, [hat_y0], true_y0)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 0

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
            
        with torch.profiler.record_function("Forward: Block 1"):
            ## Forward: Block 1
            hat_y1 = self.model[1](x1)       # Block 1 
            x2 = hat_y1.detach().to(self.gpus[2])
        
        with torch.profiler.record_function("Loss,Backward,Update: Block 1"):
            ## Loss,Backward,Update: Block 1
            args = ([self.model[1]], self.opts[1].optimizer, [hat_y1], true_y1)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 1

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
        
        with torch.profiler.record_function("Forward: Block 2"):
            ## Forward: Block 2
            hat_y2 = self.model[2](x2)       # Block 2
            x3 = hat_y2.detach().to(self.gpus[3])
        
        with torch.profiler.record_function("Loss,Backward,Update: Block 2"):
            ## Loss,Backward,Update: Block 2
            args = ([self.model[2]], self.opts[2].optimizer, [hat_y2], true_y2)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 2

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        with torch.profiler.record_function("Forward: Block 3"):
            ## Forward: Block 3
            hat_y3 = self.model[3](x3)       # Block 3
            
            x4 = hat_y3.detach()
            hat_y4 = self.model[4](x4)       # Block 3
        
        with torch.profiler.record_function("Loss,Backward,Update: Block 3"):
            ## Loss,Backward,Update: Block 3
            args = ([self.model[3], self.model[4]], self.opts[3].optimizer, [hat_y3, hat_y4], true_y3)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 3

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
            
        with torch.profiler.record_function("wait"):
            if multi_t:
                for t in range(len(tasks)):
                    loss_all += tasks[t].get_result().item()
            else:
                for loss in gpu_losses:
                    loss_all += loss.item()

        return hat_y4, loss_all

    def inference(self, X, Y):
        with torch.profiler.record_function("Y to gpu"):
            true_y3 = Y.to(self.gpus[3], non_blocking=True)

        with torch.profiler.record_function("Forward: Block 0"):
            x0 = X.to(self.gpus[0], non_blocking=True)
            hat_y0 = self.model[0](x0)       # Block 0

        with torch.profiler.record_function("Forward: Block 1"):
            x1 = hat_y0.to(self.gpus[1])
            hat_y1 = self.model[1](x1)       # Block 1 

        with torch.profiler.record_function("Forward: Block 2"):
            x2 = hat_y1.to(self.gpus[2])
            hat_y2 = self.model[2](x2)       # Block 2

        with torch.profiler.record_function("Forward: Block 3"):
            x3 = hat_y2.to(self.gpus[3])
            hat_y3 = self.model[3](x3)       # Block 3

            x4 = hat_y3
            hat_y4 = self.model[4](x4)       # Block 3

        return hat_y4, true_y3

class resnet18_BP_m(Vision_MultiGPU):
    def _init_data(self, configs):
        # Data
        super()._init_data(configs)

    def _init_model(self, configs):
        self.shape = 32
        self.in_channels = 3
        self.layer_cfg = {0:[64, 64, [1, 1]], 1:[64, 128, [2, 1]], 2:[128, 256, [2, 1]], 3:[256, 512, [2, 1]]}

        # Make Model
        self.model = []
        self.model.append(resnet18_Head(device=self.gpus[0]).to(self.gpus[0]))
        self.model.append(resnet18_BP_Component(cfg=self.layer_cfg[0], shape=self.shape, in_channels=self.layer_cfg[0][1], device=self.gpus[0]).to(self.gpus[0]))
        self.model.append(resnet18_BP_Component(cfg=self.layer_cfg[1], shape=self._shape_div_2(), in_channels=self.layer_cfg[1][1], device=self.gpus[1]).to(self.gpus[1]))
        self.model.append(resnet18_BP_Component(cfg=self.layer_cfg[2], shape=self._shape_div_2(), in_channels=self.layer_cfg[2][1], device=self.gpus[2]).to(self.gpus[2]))
        self.model.append(resnet18_BP_Component(cfg=self.layer_cfg[3], shape=1, 
                                              in_channels=self.layer_cfg[3][1], avg_pool=torch.nn.AdaptiveAvgPool2d((1, 1)),
                                              device=self.gpus[3]).to(self.gpus[3]))
        self.model.append(resnet18_Tail(num_classes=self.num_classes, device=self.gpus[3]).to(self.gpus[3]))

        self.model = torch.nn.Sequential(*self.model)

    def _init_optimizers(self, configs):
        # Optimizers
        super()._init_optimizers(configs)

        self.opts = list()

        self.opts.append(Optimizer(
            chain(self.model.parameters()), base_lr=self.base_lr, end_lr=self.end_lr, max_step=self.max_step))

    def train_step(self, X, Y, multi_t=True):
        with torch.profiler.record_function("Init"):
            self.global_steps += 1
            tasks = list()
            gpu_losses = list()
            loss_all = 0

        with torch.profiler.record_function("Y to gpu 3"):
            true_y3 = Y.to(self.gpus[3], non_blocking=True)
    
        with torch.profiler.record_function("train mode"):
            for layer in self.model:
                layer.train()
        
        with torch.profiler.record_function("opt - zero_grad"):
            for opt in self.opts:
                opt.zero_grad()
    
        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        with torch.profiler.record_function("Forward: Block 0"):
            x0 = X.to(self.gpus[0], non_blocking=True)
            x1 = self.model[0](x0)           # Block 0

            hat_y1 = self.model[1](x1)       # Block 0
            x2 = hat_y1.to(self.gpus[1])
    
        with torch.profiler.record_function("Forward: Block 1"):
            ## Forward: Block 1
            hat_y2 = self.model[2](x2)       # Block 1 
            x3 = hat_y2.to(self.gpus[2])
    
        with torch.profiler.record_function("Forward: Block 2"):
            ## Forward: Block 2
            hat_y3 = self.model[3](x3)       # Block 2
            x4 = hat_y3.to(self.gpus[3])

        with torch.profiler.record_function("Forward: Block 3"):
            ## Forward: Block 3
            hat_y4 = self.model[4](x4)       # Block 3
            
            x5 = hat_y4
            hat_y5 = self.model[5](x5)       # Block 3
    
        with torch.profiler.record_function("Loss,Backward,Update: Block 3"):
            ## Loss,Backward,Update: Block 3
            args = ([self.model[5]], self.opts[-1].optimizer, [hat_y5], true_y3)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 3

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
        
        with torch.profiler.record_function("wait"):
            if multi_t:
                for t in range(len(tasks)):
                    loss_all += tasks[t].get_result().item()
            else:
                for loss in gpu_losses:
                    loss_all += loss.item()
            
        return hat_y5, loss_all

    def inference(self, X, Y):
        with torch.profiler.record_function("Y to gpu"):
            true_y3 = Y.to(self.gpus[3], non_blocking=True)

        with torch.profiler.record_function("Forward: Block 0"):
            x0 = X.to(self.gpus[0], non_blocking=True)
            x1 = self.model[0](x0)           # Block 0
            hat_y1 = self.model[1](x1)       # Block 0

        with torch.profiler.record_function("Forward: Block 1"):
            x2 = hat_y1.to(self.gpus[1])
            hat_y2 = self.model[2](x2)       # Block 1

        with torch.profiler.record_function("Forward: Block 2"):
            x3 = hat_y2.to(self.gpus[2])
            hat_y3 = self.model[3](x3)       # Block 2

        with torch.profiler.record_function("Forward: Block 3"):
            x4 = hat_y3.to(self.gpus[3])
            x5 = self.model[4](x4)           # Block 3
            hat_y5 = self.model[5](x5)       # Block 3
        
        return hat_y5, true_y3

class resnet18_SCPL_m(Vision_MultiGPU):
    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.proj_type = configs['proj_type']

    def _init_model(self, configs):
        self.shape = 32
        self.in_channels = 3
        self.layer_cfg = {0:[64, 64, [1, 1]], 1:[64, 128, [2, 1]], 2:[128, 256, [2, 1]], 3:[256, 512, [2, 1]]}

        # Make Model
        self.model = [
            resnet18_Head(device=self.gpus[0]).to(self.gpus[0]),
            resnet18_SCPL_Component(cfg=self.layer_cfg[0], shape=self.shape, in_channels=self.layer_cfg[0][1], 
                                    proj_type=self.proj_type ,device=self.gpus[0]).to(self.gpus[0]),
            resnet18_SCPL_Component(cfg=self.layer_cfg[1], shape=self._shape_div_2(), in_channels=self.layer_cfg[1][1], 
                                    proj_type=self.proj_type ,device=self.gpus[1]).to(self.gpus[1]),
            resnet18_SCPL_Component(cfg=self.layer_cfg[2], shape=self._shape_div_2(), in_channels=self.layer_cfg[2][1], 
                                    proj_type=self.proj_type ,device=self.gpus[2]).to(self.gpus[2]),
            resnet18_SCPL_Component(cfg=self.layer_cfg[3], shape=1, in_channels=self.layer_cfg[3][1], 
                                    proj_type=self.proj_type ,avg_pool=torch.nn.AdaptiveAvgPool2d((1, 1)), device=self.gpus[3]).to(self.gpus[3]),
            resnet18_Tail(num_classes=self.num_classes, device=self.gpus[3]).to(self.gpus[3]),
        ]
        self.model = torch.nn.Sequential(*self.model)

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
        with torch.profiler.record_function("Init"):
            self.global_steps += 1
            tasks = list()
            gpu_losses = list()
            loss_all = 0

        with torch.profiler.record_function("Y to gpu"):
            true_y0 = Y.to(self.gpus[0], non_blocking=True)
            true_y1 = Y.to(self.gpus[1])
            true_y2 = Y.to(self.gpus[2])
            true_y3 = Y.to(self.gpus[3])
    
        with torch.profiler.record_function("train mode"):
            for layer in self.model:
                layer.train()
    
        with torch.profiler.record_function("opt - zero_grad"):
            for opt in self.opts:
                opt.zero_grad()
    
        # Forward: Block 0 ~ 3
        with torch.profiler.record_function("Forward: Block 0"):
            ## Forward: Block 0
            x0 = X.to(self.gpus[0], non_blocking=True)
            x1 = self.model[0](x0)       # Block 0

            hat_y1 = self.model[1](x1)       # Block 0
            x2 = hat_y1.detach().to(self.gpus[1])
    
        with torch.profiler.record_function("Loss,Backward,Update: Block 1"):
            ## Loss,Backward,Update: Block 0
            args = ([self.model[1]], self.opts[0].optimizer, [hat_y1], true_y0)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 0

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
        
        with torch.profiler.record_function("Forward: Block 1"):
            ## Forward: GPU 1
            hat_y2 = self.model[2](x2)       # Block 1 
            x3 = hat_y2.detach().to(self.gpus[2])
            
        with torch.profiler.record_function("Loss,Backward,Update: Block 1"):
            ## Loss,Backward,Update: Block 1
            args = ([self.model[2]], self.opts[1].optimizer, [hat_y2], true_y1)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 1

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
    
        with torch.profiler.record_function("Forward: Block 2"):
            ## Forward: Block 2
            hat_y3 = self.model[3](x3)       # Block 2
            x4 = hat_y3.detach().to(self.gpus[3])
            
        with torch.profiler.record_function("Loss,Backward,Update: Block 2"):
            ## Loss,Backward,Update: Block 2
            args = ([self.model[3]], self.opts[2].optimizer, [hat_y3], true_y2)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 2

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        with torch.profiler.record_function("Forward: Block 3"):
            ## Forward: Block 3
            hat_y4 = self.model[4](x4)       # Block 3
            
            x5 = hat_y4.detach()
            hat_y5 = self.model[5](x5)       # Block 3
    
        with torch.profiler.record_function("Loss,Backward,Update: Block 3"):
            ## Loss,Backward,Update: Block 3
            args = ([self.model[4], self.model[5]], self.opts[3].optimizer, [hat_y4, hat_y5], true_y3)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 3

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        with torch.profiler.record_function("wait"):
            if multi_t:
                for t in range(len(tasks)):
                    loss_all += tasks[t].get_result().item()
            else:
                for loss in gpu_losses:
                    loss_all += loss.item()
    
        return hat_y4, loss_all

    def inference(self, X, Y):
        with torch.profiler.record_function("Y to gpu"):
            true_y3 = Y.to(self.gpus[3], non_blocking=True)

        with torch.profiler.record_function("Forward: Block 0"):
            x0 = X.to(self.gpus[0], non_blocking=True)
            x1 = self.model[0](x0)           # Block 0
            hat_y1 = self.model[1](x1)       # Block 0

        with torch.profiler.record_function("Forward: Block 1"):
            x2 = hat_y1.to(self.gpus[1])
            hat_y2 = self.model[2](x2)       # Block 1

        with torch.profiler.record_function("Forward: Block 2"):
            x3 = hat_y2.to(self.gpus[2])
            hat_y3 = self.model[3](x3)       # Block 2

        with torch.profiler.record_function("Forward: Block 3"):
            x4 = hat_y3.to(self.gpus[3])
            x5 = self.model[4](x4)           # Block 3
            hat_y5 = self.model[5](x5)       # Block 3
        
        return hat_y5, true_y3