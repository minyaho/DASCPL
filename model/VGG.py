import torch
import torch.nn as nn
from utils import ALComponent, SingleGPUModel
from utils.vision import conv_layer_bn, Flatten
from .vision_single import VGG_SCPL_Block
from .loss_fn import ContrastiveLoss, PredSimLoss

class VGG(SingleGPUModel):
    def __init__(self, num_class):
        super(VGG, self).__init__()
        num_class = num_class
        self.in_channel = 3
        self.ce = nn.CrossEntropyLoss()

        self.layer1 = self._make_layer([128, 256, 'M'])
        self.layer2 = self._make_layer([256, 512, 'M'])
        self.layer3 = self._make_layer([512, 'M'])
        self.layer4 = self._make_layer([512, 'M'])

        self.fc = nn.Sequential(Flatten(), nn.Linear(2048, 2500), nn.Sigmoid(), nn.Linear(2500, num_class))
    
    def _make_layer(self, channel_size: list):
        layers = []
        for dim in channel_size:
            if dim == 'M':
                layers.append(nn.MaxPool2d(2, stride=2))
            else:
                layers.append(conv_layer_bn(self.in_channel, dim, nn.ReLU()))
                self.in_channel = dim
        return nn.Sequential(*layers)

    def _make_linear_layer(self, in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features, bias=True), nn.BatchNorm1d(out_features), nn.ReLU())
    
    def forward(self, x, y):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.fc(out)

        if self.training:
            loss = self.ce(out, y)

        if self.training:
            return loss
        else:
            return out.detach()

class VGG_AL_Component(ALComponent):
    def __init__(self, conv:nn.Module, flatten_size: int, hidden_size: int, out_features: int):

        g_function = nn.Sigmoid() 
        b_function = nn.Sigmoid()

        f = conv
        g = nn.Sequential(nn.Linear(out_features, hidden_size), g_function)
        b = nn.Sequential(Flatten(), nn.Linear(flatten_size, 5*hidden_size), b_function, nn.Linear(5*hidden_size, hidden_size), b_function)
        inv = nn.Sequential(nn.Linear(hidden_size, out_features), g_function)

        cf = nn.Sequential()
        cb = nn.MSELoss()
        ca = nn.MSELoss()
        super(VGG_AL_Component, self).__init__(f, g, b, inv, cf, cb, ca)

    
class VGG_AL(SingleGPUModel):
    def __init__(self, num_classes = 10):
        super(VGG_AL, self).__init__()
        layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}
        neuron_size = 500
        self.shape = 32
        self.num_classes = num_classes

        layer1 = self._make_layer([128, 256, "M"])
        self.layer1 = VGG_AL_Component(layer1, self.shape*self.shape*256, neuron_size, num_classes)

        self.shape //= 2
        layer2 = self._make_layer([256, 512, "M"])
        self.layer2 = VGG_AL_Component(layer2, self.shape*self.shape*512, neuron_size, num_classes)

        self.shape //= 2
        layer3 = self._make_layer([512, "M"])
        self.layer3 = VGG_AL_Component(layer3, self.shape*self.shape*512, neuron_size, num_classes)

        self.shape //= 2
        layer4 = self._make_layer([512, "M"])
        self.layer4 = VGG_AL_Component(layer4, self.shape*self.shape*512, neuron_size, num_classes)

    def forward(self, x, y):
        if self.training:

            y_onehot = torch.zeros([len(y), self.num_classes]).cuda(non_blocking=True)
            for i in range(len(y)):

                y_onehot[i][y[i]] = 1.

            _s = x
            _t = y_onehot
            total_loss = {'f':[], 'b':[],'ae':[]}

            _s, _t, loss_f, loss_b, loss_ae = self.layer1(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)

            _s, _t, loss_f, loss_b, loss_ae = self.layer2(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)

            _s, _t, loss_f, loss_b, loss_ae = self.layer3(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)

            _s, _t, loss_f, loss_b, loss_ae = self.layer4(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)
            return total_loss
        else:
            _s = x
            _s = self.layer1(_s, None)
            _s = self.layer2(_s, None)
            _s = self.layer3(_s, None)
            _t0 = self.layer4.bridge_forward(_s)
            _t0 = self.layer3(None, _t0)
            _t0 = self.layer2(None, _t0)
            _t0 = self.layer1(None, _t0)
            return _t0
    
    def _make_layer(self, channel_size):
        layers = []
        for dim in channel_size:
            if dim == "M":
                layers.append(nn.MaxPool2d(2, stride=2))
                self.size/=2
            else:
                layers.append(conv_layer_bn(self.features, dim, nn.ReLU()))
                self.features = dim
        return nn.Sequential(*layers)

class VGG_SCPL_REWRITE(SingleGPUModel):
    def __init__(self, num_classes = 10, device=None):
        super(VGG_SCPL_REWRITE, self).__init__()
        
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # print("VGG_SCPL_REWRITE",device)
        
        self.num_classes = num_classes

        self.shape = 32
        self.in_channels = 3
        layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}

        self.layer1 = VGG_SCPL_Block(cfg=layer_cfg[0], shape=self._shape_div_2(), in_channels=3, device=device).to(device)
        self.layer2 = VGG_SCPL_Block(cfg=layer_cfg[1], shape=self._shape_div_2(), in_channels=self.layer1.in_channels, device=device).to(device)
        self.layer3 = VGG_SCPL_Block(cfg=layer_cfg[2], shape=self._shape_div_2(), in_channels=self.layer2.in_channels, device=device).to(device)
        self.layer4 = VGG_SCPL_Block(cfg=layer_cfg[3], shape=self._shape_div_2(), in_channels=self.layer3.in_channels, device=device).to(device)

        self.fc = nn.Sequential(Flatten(), nn.Linear(2048, 2500), nn.Sigmoid(), nn.Linear(2500, num_classes)).to(device)
        self.ce = nn.CrossEntropyLoss()

        
    def forward(self, x, y):
        loss_all = 0

        output1, loss1 = self.layer1(x,y)
        loss_all += loss1

        output2, loss2 = self.layer2(output1,y)
        loss_all += loss2

        output3, loss3 = self.layer3(output2,y)
        loss_all += loss3
        
        output4, loss4 = self.layer4(output3,y)
        loss_all += loss4

        output5 = self.fc(output4)
        if self.training:
            loss_all += self.ce(output5, y)
            return loss_all
        else:
            return output5
        
    def _shape_div_2(self):
        self.shape //= 2
        return self.shape

class VGG_SCPL(SingleGPUModel):
    def __init__(self, num_classes = 10):
        super(VGG_SCPL, self).__init__()
        self.num_classes = num_classes

        self.shape = 32
        self.in_channels = 3
        layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}

        self.layer1 = self._make_layer([128, 256, "M"])
        self.shape //= 2
        self.loss1 =  ContrastiveLoss(0.1, input_neurons = 2048, c_in = 256, shape = self.shape)
        
        self.layer2 = self._make_layer([256, 512, "M"])
        self.shape //= 2
        self.loss2 =  ContrastiveLoss(0.1, input_neurons = 2048, c_in = 512, shape = self.shape)

        self.layer3 = self._make_layer([512, "M"])
        self.shape //= 2
        self.loss3 =  ContrastiveLoss(0.1, input_neurons = 2048, c_in = 512, shape = self.shape)

        self.layer4 = self._make_layer([512, "M"])
        self.shape //= 2
        self.loss4 =  ContrastiveLoss(0.1, input_neurons = 2048, c_in = 512, shape = self.shape)

        self.fc = nn.Sequential(Flatten(), nn.Linear(2048, 2500), nn.Sigmoid(), nn.Linear(2500, num_classes))
        self.ce = nn.CrossEntropyLoss()

        
    def forward(self, x, y):
        loss = 0
        output = self.layer1(x)
        if self.training:
            loss += self.loss1(output, y)
            output = output.detach()
        output = self.layer2(output)
        if self.training:
            loss += self.loss2(output, y)
            output = output.detach()
        output = self.layer3(output)
        if self.training:
            loss += self.loss3(output, y)
            output = output.detach()
        output = self.layer4(output)
        if self.training:
            loss += self.loss4(output, y)
            output = output.detach()
        
      
        output = self.fc(output)
        if self.training:
            loss += self.ce(output, y)
            return loss
        else:
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
    
class VGG_PredSim(SingleGPUModel):
    def __init__(self, num_classes = 10):
        super(VGG_PredSim, self).__init__()
        self.num_classes = num_classes

        self.num_layers = 4
        self.size = 32
        self.features = 3
        layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}
        conv_layers = []
        loss_funcs = []
        for i in range(self.num_layers):
            conv_layers.append(self._make_conv_layer(layer_cfg[i], nn.ReLU()))
            loss_funcs.append(PredSimLoss(0.1, input_neurons = 2048, c_in = layer_cfg[i][-2], shape = self.size))

        self.flatten = Flatten()

        self.fc = nn.Sequential(Flatten(), nn.Linear(2048, 2500), nn.Sigmoid(), nn.Linear(2500, num_classes))
        self.conv = nn.ModuleList(conv_layers)
        self.cl = nn.ModuleList(loss_funcs)
        self.ce = nn.CrossEntropyLoss()

        
    def forward(self, x, label):
        if self.training:
            half = x.shape[0]//2
            x1 = x

            total_loss = 0
            for i in range(self.num_layers):
                x1 = self.conv[i](x1)
                total_loss+=self.cl[i](x1, label)
                x1 = x1.detach()

            y = self.fc(x1[0:half].detach())
            total_loss += self.ce(y, label[0:half])

            return total_loss
        else:
            for i in range(self.num_layers):
                x = self.conv[i](x)
            y = self.fc(x)

            return y
    
    def _make_conv_layer(self, channel_size, activation):
        layers = []
        for dim in channel_size:
            if dim == "M":
                layers.append(nn.MaxPool2d(2, stride=2))
                self.size/=2
            else:
                layers.append(conv_layer_bn(self.features, dim, activation))
                self.features = dim
        return nn.Sequential(*layers)
    def _make_linear_layer(self, in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features, bias=True), nn.BatchNorm1d(out_features), nn.ReLU())