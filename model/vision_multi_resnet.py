import torch
import torch.nn as nn
from itertools import chain
from collections import OrderedDict
from .vision_single import resnet_Head, resnet_Block, resnet_Predictor # get_resnet_config
from .vision_multi import Vision_MultiGPU
from utils import Optimizer, CPUThread
from utils.vision import conv_1x1_bn, conv_layer_bn

class Bottleneck(nn.Module):
    """
    Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_1x1_bn(in_channels, out_channels, nn.LeakyReLU(inplace=True), 1, False)
        self.conv2 = conv_layer_bn(out_channels, out_channels, nn.LeakyReLU(inplace=True), stride, False)
        self.conv3 = conv_1x1_bn(out_channels, self.expansion * out_channels, None, 1, False)
        self.relu = nn.LeakyReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = conv_1x1_bn(in_channels, self.expansion * out_channels, None, stride, False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.relu(out + self.shortcut(x))
        return out
    

class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer_bn(in_channels, out_channels, nn.LeakyReLU(inplace=True), stride, False)
        self.conv2 = conv_layer_bn(out_channels, out_channels, None, 1, False)
        self.relu = nn.LeakyReLU(inplace=True)

        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        if stride != 1:
            # self.shortcut = conv_layer_bn(in_channels, out_channels, None, stride, False) # Original SCPL settings
            self.shortcut = conv_1x1_bn(in_channels, out_channels, None, stride, False) # New settings. Maybe this is the correct setting for ResNet18

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + self.shortcut(x))
        return out

def get_resnet_config(model_name):
    layer_cfg = None
    if "18" in model_name:
        print("[Model init] Use ResNet-18")
        layer_cfg = {
            0:[BasicBlock, 64, 64, 2, 1], 
            1:[BasicBlock, 64, 128, 2, 2], 
            2:[BasicBlock, 128, 256, 2, 2], 
            3:[BasicBlock, 256, 512, 2, 2],
            4:[512]
        }
    elif "34" in model_name:
        print("[Model init] Use ResNet-34")
        layer_cfg = {
            0:[BasicBlock, 64, 64, 3, 1], 
            1:[BasicBlock, 64, 128, 4, 2], 
            2:[BasicBlock, 128, 256, 6, 2], 
            3:[BasicBlock, 256, 512, 3, 2],
            4:[512]
        }
    elif "50" in model_name:
        print("[Model init] Use ResNet-50")
        layer_cfg = {
            0:[Bottleneck, 64, 64, 3, 1], 
            1:[Bottleneck, 256, 128, 4, 2], 
            2:[Bottleneck, 512, 256, 6, 2], 
            3:[Bottleneck, 1024, 512, 3, 2],
            4:[2048]
        }
    elif "101" in model_name:
        print("[Model init] Use ResNet-101")
        layer_cfg = {
            0:[Bottleneck, 64, 64, 3, 1], 
            1:[Bottleneck, 256, 128, 4, 2], 
            2:[Bottleneck, 512, 256, 23, 2], 
            3:[Bottleneck, 1024, 512, 3, 2],
            4:[2048]
        }
    elif "152" in model_name:
        print("[Model init] Use ResNet-152")
        layer_cfg = {
            0:[Bottleneck, 64, 64, 3, 1], 
            1:[Bottleneck, 256, 128, 8, 2], 
            2:[Bottleneck, 512, 256, 36, 2], 
            3:[Bottleneck, 1024, 512, 3, 2],
            4:[2048]
        }
    else:
        print("[Model init] Unrecognized ResNet settings. Your model is {}. Use default model: ResNet-18".format(model_name))
        layer_cfg = {
            0:[BasicBlock, 64, 64, 2, 1], 
            1:[BasicBlock, 128, 128, 2, 2], 
            2:[BasicBlock, 256, 256, 2, 2], 
            3:[BasicBlock, 512, 512, 2, 2],
            4:[512]
        }

    return layer_cfg

class resnet_BP_m(Vision_MultiGPU):
    def __init__(self, configs):
        super(resnet_BP_m, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.proj_type = None
        self.pred_type = None

    def _init_model(self, configs):
        self.shape = 32
        self.in_channels = 3

        self.layer_cfg = get_resnet_config(configs['model'])
        assert self.layer_cfg != None, "Error! layer_cfg is none"

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
                "cfg":self.layer_cfg[3], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[3][1], "proj_type":self.proj_type, "pred_type":self.pred_type, "device":self.devices[3]},
            'predictor': {
                "in_dim":self.layer_cfg[4][0], "num_classes":self.num_classes, "avg_pool":torch.nn.AdaptiveAvgPool2d((1, 1)), "device":self.devices[3]}
        }

        # Make Model
        self.model = []
        for key, values in self.model_cfg.items():
            if 'backbone' in key:
                self.model.append((key, resnet_Block(**values).to(values["device"])))
            elif 'head' in key: 
                self.model.append((key, resnet_Head(**values).to(values["device"])))
            elif 'predictor' in key:
                self.model.append((key, resnet_Predictor(**values).to(values["device"])))
        self.model = torch.nn.Sequential(OrderedDict(self.model))

    #     # Optimize model
    #     self._opt_model()

    # def _opt_model(self, zero_init_residual=False):
    #     # https://github.com/HobbitLong/SupContrast/blob/master/networks/resnet_big.py
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #     # Zero-initialize the last BN in each residual branch,
    #     # so that the residual branch starts with zeros, and each residual block behaves
    #     # like an identity. This improves the model by 0.2~0.3% according to:
    #     # https://arxiv.org/abs/1706.02677
    #     if zero_init_residual:
    #         for m in self.modules():
    #             if isinstance(m, Bottleneck):
    #                 nn.init.constant_(m.conv3[1].weight, 0)
    #             elif isinstance(m, BasicBlock):
    #                 nn.init.constant_(m.conv2[1].weight, 0)

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

class resnet_BP_p_m(resnet_BP_m):
    def __init__(self, configs):
        super(resnet_BP_p_m, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.proj_type = None
        self.pred_type = configs['pred_type']
        assert self.pred_type not in [None, ''], "Setting error, pred_type is none or empty. pred_type: {}".format(self.pred_type)

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
            [true_y0, true_y1, true_y2, true_y3, true_y3], True
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

class resnet_SCPL_m(Vision_MultiGPU):
    def __init__(self, configs):
        super(resnet_SCPL_m, self).__init__(configs)

    def _init_data(self, configs):
        # Data
        super()._init_data(configs)
        self.proj_type = configs['proj_type']
        assert self.proj_type not in [None, ''], "Setting error, proj_type is none or empty. proj_type: {}".format(self.proj_type)

    def _init_model(self, configs):
        self.shape = 32
        self.in_channels = 3

        self.layer_cfg = get_resnet_config(configs['model'])
        assert self.layer_cfg != None, "Error! layer_cfg is none"

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
                "cfg":self.layer_cfg[3], "shape":self._shape_div_2(), "num_classes":self.num_classes,
                "in_channels":self.layer_cfg[3][1], "proj_type":self.proj_type, "pred_type":self.pred_type, "device":self.devices[3]},
            'predictor': {
                "in_dim":self.layer_cfg[4][0], "num_classes":self.num_classes, "avg_pool":torch.nn.AdaptiveAvgPool2d((1, 1)), "device":self.devices[3]}
        }

        # Make Model
        self.model = []
        for key, values in self.model_cfg.items():
            if 'backbone' in key:
                self.model.append((key, resnet_Block(**values).to(values["device"])))
            elif 'head' in key: 
                self.model.append((key, resnet_Head(**values).to(values["device"])))
            elif 'predictor' in key:
                self.model.append((key, resnet_Predictor(**values).to(values["device"])))
        self.model = torch.nn.Sequential(OrderedDict(self.model))

    #     # Optimize model
    #     self._opt_model()

    # def _opt_model(self, zero_init_residual=False):
    #     # https://github.com/HobbitLong/SupContrast/blob/master/networks/resnet_big.py
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #     # Zero-initialize the last BN in each residual branch,
    #     # so that the residual branch starts with zeros, and each residual block behaves
    #     # like an identity. This improves the model by 0.2~0.3% according to:
    #     # https://arxiv.org/abs/1706.02677
    #     if zero_init_residual:
    #         for m in self.modules():
    #             if isinstance(m, Bottleneck):
    #                 nn.init.constant_(m.conv3[1].weight, 0)
    #             elif isinstance(m, BasicBlock):
    #                 nn.init.constant_(m.conv2[1].weight, 0)

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
    
class resnet_DASCPL_m(resnet_SCPL_m):
    def __init__(self, configs):
        super(resnet_DASCPL_m, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.proj_type = configs['proj_type']
        assert self.proj_type not in [None, ''], "Setting error, proj_type is none or empty. proj_type: {}".format(self.proj_type)
        self.pred_type = configs['pred_type']
        assert self.pred_type not in [None, ''], "Setting error, pred_type is none or empty. pred_type: {}".format(self.pred_type)
        self.temperature = configs['temperature']

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
