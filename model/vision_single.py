import torch
import torch.nn as nn
from utils.vision import conv_layer_bn, Flatten
from .loss_fn import VisionLocalLoss

class VGG_Block(nn.Module):
    def __init__(self, cfg, shape, in_channels, num_classes=None, proj_type=None, pred_type=None, temperature=0.1, device=None):
        super(VGG_Block, self).__init__()
        
        self.in_channels = in_channels
        self.device = device
        self.shape = shape
        self.out_channels = cfg[-2]
        self.temperature = temperature

        self.layer = self._make_layer(cfg)

        self._make_loss_layer(num_classes, proj_type, pred_type)

    def _make_loss_layer(self, num_classes, proj_type=None, pred_type=None):
        if (proj_type != None) or (pred_type != None):
            self.proj_type = proj_type
            self.pred_type = pred_type
            self.loss = VisionLocalLoss(
                temperature=self.temperature, c_in = self.out_channels, shape = self.shape, 
                num_classes=num_classes, proj_type=proj_type, pred_type=pred_type, device=self.device)

    def forward(self, x):
        output = self.layer(x)
        return output
    
    def _make_layer(self, channel_size):
        layers = []
        for dim in channel_size:
            if dim == "M":
                layers.append(nn.MaxPool2d(2, stride=2))
            else:
                layers.append(conv_layer_bn(self.in_channels, dim, nn.ReLU()))
                self.in_channels = dim
        return nn.Sequential(*layers)

class VGG_SCPL_Block(VGG_Block):
    def __init__(self, cfg, shape, in_channels, num_classes=None, proj_type="m", pred_type=None, device=None):
        super(VGG_SCPL_Block,self).__init__(cfg, shape, in_channels, num_classes, proj_type, pred_type, device)
    
    def _make_loss_layer(self, num_classes, proj_type=None, pred_type=None):
        return super()._make_loss_layer(num_classes, proj_type, pred_type=None)

    def forward(self, x, y):
        loss = 0
        output = self.layer(x)
        if self.training:
            loss += self.loss(output, y)
            output = output.detach()
        return output, loss

class VGG_Predictor(nn.Module):
    def __init__(self, num_classes, in_channels, shape, device=None):
        super(VGG_Predictor, self).__init__()
        input_neurons = int(in_channels * shape * shape)
        self.layer = nn.Sequential(Flatten(), nn.Linear(input_neurons, 2500), nn.Sigmoid(), nn.Linear(2500, num_classes)).to(device)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.layer(x)
        return output

class resnet18_Head(nn.Module):
    def __init__(self, device='cpu'):
        super(resnet18_Head,self).__init__()
        self.device = device
        self.shape = 32
        self.layer = conv_layer_bn(3, 64, nn.LeakyReLU(inplace=True), 1, False)
    def forward(self, x, y=None):
        output = self.layer(x)
        return output

class resnet18_Predictor(nn.Module):
    def __init__(self, num_classes=100, device='cpu'):
        super(resnet18_Predictor,self).__init__()
        self.device = device
        self.layer = nn.Linear(512, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        output = x.view(x.size(0), -1)
        output = self.layer(output)
        return output

class resnet18_Block(nn.Module):
    def __init__(self, cfg, shape, in_channels, avg_pool=None, proj_type=None, pred_type=None, num_classes=None, temperature=0.1, device='cpu'):
        super(resnet18_Block,self).__init__()
        self.shape = shape
        self.layer = self._make_layer(*cfg)
        self.device = device
        self.in_channels = in_channels
        self.out_channels = cfg[-2]
        self.temperature = temperature
        self.avg_pool = avg_pool #nn.AdaptiveAvgPool2d((1, 1))

        self._make_loss_layer(num_classes, proj_type, pred_type)

    def _make_loss_layer(self, num_classes, proj_type=None, pred_type=None):
        if (proj_type != None) or (pred_type != None):
            self.proj_type = proj_type
            self.pred_type = pred_type
            self.loss = VisionLocalLoss(
                temperature=self.temperature, c_in = self.out_channels, shape = self.shape, 
                num_classes=num_classes, proj_type=proj_type, pred_type=pred_type, device=self.device)

    def _make_layer(self, in_channels, out_channels, strides):
        layers = []
        cur_channels = in_channels
        for stride in strides:
            layers.append(BasicBlock(cur_channels, out_channels, stride))
            cur_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        output = self.layer(x)
        if self.avg_pool != None:
            output = self.avg_pool(output)
        return output

class resnet18_SCPL_Block(resnet18_Block):
    def __init__(self, cfg, shape, in_channels, avg_pool=None, proj_type="m", pred_type=None, num_classes=None, device='cpu'):
        super(resnet18_SCPL_Block,self).__init__(cfg, shape, in_channels, avg_pool, proj_type, pred_type, num_classes, device)

    def _make_loss_layer(self, num_classes, proj_type=None, pred_type=None):
        return super()._make_loss_layer(num_classes, proj_type, pred_type=None)

    def forward(self, x, y):
        loss = 0
        output = self.layer(x)
        if self.training:
            loss += self.loss(output, y)
            output = output.detach()
        return output, loss

""" 修改自: https://github.com/batuhan3526/ResNet50_on_Cifar_100_Without_Transfer_Learning """
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer_bn(in_channels, out_channels, nn.LeakyReLU(inplace=True), stride, False)
        self.conv2 = conv_layer_bn(out_channels, out_channels, None, 1, False)
        self.relu = nn.LeakyReLU(inplace=True)

        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        if stride != 1:
            self.shortcut = conv_layer_bn(in_channels, out_channels, None, stride, False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + self.shortcut(x))
        return out
