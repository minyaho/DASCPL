import torch
import torch.nn as nn
import random
import numpy as np
import math
import torch.nn.functional as F

class NLP_Predictor(nn.Module):
    def __init__(self, input_dim, out_dim, hid_dim=100, act_fun = nn.Tanh(), device=None):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(input_dim, hid_dim), act_fun, nn.Linear(hid_dim, out_dim))
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x):
        return self.layer(x)
    
class Vision_Predictor(nn.Module):
    def __init__(self, out_dim, input_dim, hid_dim=2500, device=None):
        super().__init__()
        self.device = device
        self.layer = nn.Sequential(Flatten(),
                                #    nn.Linear(in_channels, hid_dim, bias=False),
                                #    nn.BatchNorm1d(hid_dim),
                                #    nn.ReLU(inplace=True), # hidden layer
                                #    nn.Linear(hid_dim, num_classes)) # output layer
                                nn.Linear(input_dim, hid_dim), nn.Sigmoid(), nn.Linear(hid_dim, out_dim))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        output = self.layer(x)
        return output
    
class LocalLoss(nn.Module):
    def __init__(self, temperature=0.1, input_dim = 128, hid_dim = 512, out_dim = 1024, 
                 num_classes=None, proj_type=None, pred_type=None, device=None):
        super(LocalLoss, self).__init__()
        self.temperature = temperature
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_classes = num_classes

        if device != None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if "cpu" in self.device:
            self.tensor_type = torch.FloatTensor
        else:
            self.tensor_type = torch.cuda.FloatTensor

        self.proj_type = proj_type.replace(' ', '').split(",") if proj_type != None else None
        self.pred_type = pred_type.replace(' ', '').split(",") if pred_type != None else None
        self.projector = None
        self.predictor = None

        # if "dcl" in proj_type:
        #     self.loss_type = 'dcl'
        #     print("[CL Loss] Use dcl loss")
        # else:
        #     self.loss_type = 'infoNCE'
        #     print("[CL Loss] Use infoNCE loss")
    
    def _contrastive_loss(self, x, label):
        x = self.projector(x)
        x =  nn.functional.normalize(x)
        label = label.view(-1, 1)
        batch_size = label.shape[0]
        mask = torch.eq(label, label.T).type(self.tensor_type)
        denom_mask = torch.scatter(torch.ones_like(mask, device=x.device), 1, torch.arange(batch_size, device=x.device).view(-1, 1), 0)
        logits = torch.div(torch.matmul(x, x.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        denom = torch.exp(logits) * denom_mask
        # if self.loss_type == 'dcl':
        #     invert_make = torch.ones_like(mask, device=x.device) - mask
        #     denom = torch.exp(logits) * invert_make #denom_mask
        # else:
        #     denom = torch.exp(logits) * denom_mask
        prob = logits - torch.log(denom.sum(1, keepdim=True))
        loss = (denom_mask * mask * prob).sum(1) / mask.sum(1)
        loss = -loss
        loss = loss.view(1, batch_size).mean()
        return loss
    
    def _predict_loss(self, x, label):
        hat_y = self.predictor(x if 'non-detach' in self.pred_type else x.detach())
        loss = self.predictor.loss(hat_y, label)
        return loss

    def training_mode(self, x, label):
        loss_all = 0
        hat_y = None
        if self.projector != None:
            loss_all += self._contrastive_loss(x, label)
        if self.predictor != None:
            loss_all += self._predict_loss(x, label)
        return loss_all

    def inference_mode(self, x):
        if self.predictor != None:
            hat_y = self.predictor(x)
            return hat_y
        else:
            return None

    def forward(self, x, label=None):
        if self.training:
            loss = self.training_mode(x, label)
            return loss
        else:
            hat_y = self.inference_mode(x)
            return hat_y

class VisionLocalLoss(LocalLoss):
    def __init__(self, temperature=0.1, input_dim=128, hid_dim=512, out_dim=1024, c_in = 256, shape = 32, 
                 num_classes=None, proj_type=None, pred_type=None, device=None):

        super(VisionLocalLoss, self).__init__(temperature, input_dim, hid_dim, out_dim , num_classes, proj_type, pred_type, device)

        self.input_dim = int(c_in * shape * shape)
        if self.proj_type != None:
            self.projector = nn.Sequential(Flatten(), make_projector(self.proj_type, self.input_dim, self.hid_dim, self.out_dim, self.device, temperature=self.temperature))
        if (self.pred_type != None) and ("none" not in self.pred_type):
            self.predictor = Vision_Predictor(out_dim=self.num_classes, input_dim=self.input_dim, hid_dim=self.hid_dim, device=self.device)
            info_str = "[Predictor Loss] Use local predictor, in_dim: {}, hid_dim: {}, out_dim: {}, Device: {}".format(self.input_dim, self.hid_dim, self.num_classes, self.device)
            deatch_str = ", detach input: " + ("disable" if 'non-detach' in self.pred_type else "enable")
            print(info_str + deatch_str)

class NLPLocalLoss(LocalLoss):
    def __init__(self, temperature=0.1, input_dim=300, hid_dim=300, out_dim=300,
                 num_classes=None, proj_type=None, pred_type=None, device=None):

        super(NLPLocalLoss, self).__init__(temperature, input_dim, hid_dim, out_dim , num_classes, proj_type, pred_type, device)
        if self.proj_type != None:
            self.projector = make_projector(self.proj_type, self.input_dim, self.out_dim, self.hid_dim, self.device, temperature=self.temperature)
        if self.pred_type != None:
            self.predictor = NLP_Predictor(out_dim=self.num_classes, input_dim=self.input_dim, hid_dim=self.hid_dim ,device=self.device)
            info_str = "[Predictor Loss] Use local predictor, in_dim: {}, hid_dim: {}, out_dim: {}, Device: {}".format(self.input_dim, self.hid_dim, self.num_classes, self.device)
            deatch_str = ", detach input: " + ("disable" if 'non-detach' in self.pred_type else "enable")
            print(info_str + deatch_str)

class NLP_Tail(nn.Module):
    def __init__(self, inp_dim, out_dim, hid_dim=100, act_fun = nn.Tanh(), device=None):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(inp_dim, hid_dim), act_fun, nn.Linear(hid_dim, out_dim))
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x):
        return self.layer(x)
    
class vision_Tail(nn.Module):
    def __init__(self, num_classes, in_channels, shape,  hid_dim=500, device=None):
        super().__init__()
        self.device = device
        self.layer = nn.Sequential(Flatten(),
                                #    nn.Linear(in_channels, hid_dim, bias=False),
                                #    nn.BatchNorm1d(hid_dim),
                                #    nn.ReLU(inplace=True), # hidden layer
                                #    nn.Linear(hid_dim, num_classes)) # output layer
                                nn.Linear(in_channels, hid_dim), nn.Sigmoid(), nn.Linear(hid_dim, num_classes))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        output = self.layer(x)
        return output

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, input_neurons = 128, mid_neurons = 512, out_neurons = 1024, 
                 c_in = 256, shape = 32, device=None, num_classes=None, proj_type='m'):
        super(ContrastiveLoss, self).__init__()
        
        self.temperature = temperature

        if device != None:
            self.device = device
        else:
            # self.device = "cpu"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if "cpu" in self.device:
            self.tensor_type = torch.FloatTensor
        else:
            self.tensor_type = torch.cuda.FloatTensor

        print(c_in, shape, c_in * shape * shape)
        
        input_neurons = int(c_in * shape * shape)

        self.proj_type = proj_type.split(",")
        self.projector = nn.Sequential(Flatten(), make_projector(proj_type, input_neurons, mid_neurons, out_neurons, self.device))

        # if "dcl" in proj_type:
        #     self.loss_type = 'dcl'
        #     print("[CL Loss] Use dcl loss")
        # else:
        #     self.loss_type = 'infoNCE'
        #     print("[CL Loss] Use infoNCE loss")

        if "predict" in self.proj_type:
            self.classifier = vision_Tail(num_classes=num_classes, in_channels=input_neurons, shape=shape, device=self.device)
            print("[CL Loss] Use local classifier, in_dim: {}, out_dim: {}, Device: {}".format(input_neurons, num_classes, self.device))

        # if out_neurons == 0:
        #     self.linear = nn.Sequential(Flatten())#.to(self.device)
        # elif mid_neurons == 0:
        #     self.linear = nn.Sequential(Flatten(), nn.Linear(input_neurons, out_neurons))#.to(self.device)
        # else:
        #     self.linear = nn.Sequential(Flatten(), nn.Linear(input_neurons, mid_neurons), nn.ReLU(), nn.Linear(mid_neurons, out_neurons))#.to(self.device)

    
    def train_mode(self, x, label):
        label2 = label.clone()
        x1 = self.projector(x)
        x1 =  nn.functional.normalize(x1)
        label = label.view(-1, 1)
        batch_size = label.shape[0]
        mask = torch.eq(label, label.T).type(self.tensor_type)
        denom_mask = torch.scatter(torch.ones_like(mask, device=x.device), 1, torch.arange(batch_size, device=x.device).view(-1, 1), 0)
        logits = torch.div(torch.matmul(x1, x1.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        denom = torch.exp(logits) * denom_mask
        # if self.loss_type == 'dcl':
        #     invert_make = torch.ones_like(mask, device=x.device) - mask
        #     denom = torch.exp(logits) * invert_make #denom_mask
        # else:
        #     denom = torch.exp(logits) * denom_mask
        prob = logits - torch.log(denom.sum(1, keepdim=True))
        loss = (denom_mask * mask * prob).sum(1) / mask.sum(1)
        loss = -loss
        loss = loss.view(1, batch_size).mean()

        if "predict" in self.proj_type:
            y = self.classifier(x.detach() if "detach" in self.proj_type else x)
            label_loss = self.classifier.loss(y, label2)
            loss = loss + label_loss

        return loss
    
    def inference(self, x, label):
        if "predict" in self.proj_type:
            y = self.classifier(x)
            return y
        else:
            return None

    def forward(self, x, label=None):
        if self.training:
            return self.train_mode(x, label)
        else:
            return self.inference(x, label)

class PredSimLoss(nn.Module):
    def __init__(self, temperature = 0.1, input_neurons = 2048, c_in = 256, shape = 32):
        super().__init__()
        num_classes = 200
        self.conv_loss = nn.Conv2d(c_in, c_in, 3, stride=1, padding=1, bias=False)
        self.decoder_y = nn.Linear(input_neurons, num_classes)
        # Resolve average-pooling kernel size in order for flattened dim to match args.dim_in_decoder
        ks_h, ks_w = 1, 1
        dim_out_h, dim_out_w = shape, shape
        dim_in_decoder = c_in*dim_out_h*dim_out_w
        while dim_in_decoder > input_neurons and ks_h < shape:
            ks_h*=2
            dim_out_h = math.ceil(shape / ks_h)
            dim_in_decoder = c_in*dim_out_h*dim_out_w
            if dim_in_decoder > input_neurons:
               ks_w*=2
               dim_out_w = math.ceil(shape / ks_w)
               dim_in_decoder = c_in*dim_out_h*dim_out_w 
        if ks_h > 1 or ks_w > 1:
            pad_h = (ks_h * (dim_out_h - shape // ks_h)) // 2
            pad_w = (ks_w * (dim_out_w - shape // ks_w)) // 2
            self.avg_pool = nn.AvgPool2d((ks_h,ks_w), padding=(0, 0))
        else:
            self.avg_pool = None
    def forward(self, h, y):
        y_onehot = nn.functional.one_hot(y, num_classes=200).float()
        h_loss = self.conv_loss(h)
        Rh = similarity_matrix(h_loss)
        
        if self.avg_pool is not None:
            h = self.avg_pool(h)
        y_hat_local = self.decoder_y(h.view(h.size(0), -1))
        
        Ry = similarity_matrix(y_onehot).detach()
        loss_pred = (1-0.99) * F.cross_entropy(y_hat_local,  y.detach())
        loss_sim = 0.99 * F.mse_loss(Rh, Ry)
        loss = loss_pred + loss_sim
        
        return loss

def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return R

class NLPContrastiveLoss(nn.Module):
    def __init__(self,  inp_dim, out_dim, hid_dim=100, temperature=0.1, proj_type="i", class_num=None, device=None):
        super(NLPContrastiveLoss, self).__init__()
        self.temperature = temperature
        
        if device != None:
            self.device = device
        else:
            # self.device = "cpu"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if "cpu" in self.device:
            self.tensor_type = torch.FloatTensor
        else:
            self.tensor_type = torch.cuda.FloatTensor

        self.proj_type = proj_type.split(",")
        self.projector = make_projector(self.proj_type, inp_dim, out_dim, hid_dim, self.device)

        # if "dcl" in proj_type:
        #     self.loss_type = 'dcl'
        #     print("[CL Loss] Use dcl loss")
        # else:
        #     self.loss_type = 'infoNCE'
        #     print("[CL Loss] Use infoNCE loss")

        if "predict" in self.proj_type:
            self.classifier = NLP_Tail(inp_dim, class_num, hid_dim, act_fun = nn.Tanh(), device=self.device)
            print("[CL Loss] Use local classifier, in_dim: {}, out_dim: {}, h_dim: {}, Device: {}".format(inp_dim, class_num, hid_dim, device)
                + ", detach: " + "use" if "detach" in self.proj_type else "non")

    def train_mode(self, x, label):
        label2 = label.clone()
        x1 = self.projector(x)
        x1 =  nn.functional.normalize(x1)
        label = label.view(-1, 1)
        batch_size = label.shape[0]
        mask = torch.eq(label, label.T).type(self.tensor_type)
        denom_mask = torch.scatter(torch.ones_like(mask, device=x.device), 1, torch.arange(batch_size, device=x.device).view(-1, 1), 0)
        logits = torch.div(torch.matmul(x1, x1.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        # if self.loss_type == 'dcl':
        #     invert_make = torch.ones_like(mask, device=x.device) - mask
        #     denom = torch.exp(logits) * invert_make #denom_mask
        # else:
        #     denom = torch.exp(logits) * denom_mask
        denom = torch.exp(logits) * denom_mask
        prob = logits - torch.log(denom.sum(1, keepdim=True))
        loss = (denom_mask * mask * prob).sum(1) / mask.sum(1)
        loss = -loss
        loss = loss.view(1, batch_size).mean()

        if "predict" in self.proj_type:
            y = self.classifier(x.detach() if "detach" in self.proj_type else x)
            label_loss = self.classifier.loss(y, label2)
            loss = loss + label_loss

        return loss
    
    def inference(self, x, label):
        if "predict" in self.proj_type:
            y = self.classifier(x)
            return y
        else:
            return None

    def forward(self, x, label=None):

        if self.training:
            return self.train_mode(x, label)
        else:
            return self.inference(x, label)

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def make_projector(proj_type, inp_dim, hid_dim, out_dim, device, temperature=None):
    if "i" in proj_type:
        # Identity function
        print("[CL Loss] Type: Identity function, Device: {}".format(device))
        return nn.Identity()
    elif "l" in proj_type:
        # Linear
        print("[CL Loss] Type: Linear, in_dim: {}, out_dim: {}, Device: {}".format(inp_dim, out_dim, device), end="")
        print(", Temperature: {}".format(temperature) if temperature != None else "\n")
        return nn.Sequential(nn.Linear(inp_dim, out_dim))
    elif "m" in proj_type:
        # MLP
        print("[CL Loss] Type: MLP, in_dim: {}, out_dim: {}, h_dim: {}, Device: {}".format(inp_dim, out_dim, hid_dim, device), end="")
        print(", Temperature: {}".format(temperature) if temperature != None else "\n")
        return nn.Sequential(nn.Linear(inp_dim, hid_dim), 
                             nn.ReLU(), 
                             nn.Linear(hid_dim, out_dim))
    elif 'mb' in proj_type:
        print("[CL Loss] Type: MLP with BN, in_dim: {}, out_dim: {}, h_dim: {}, Device: {}".format(inp_dim, out_dim, hid_dim, device), end="")
        print(", Temperature: {}".format(temperature) if temperature != None else "\n")
        return nn.Sequential(nn.Linear(inp_dim, hid_dim, bias=False),
                             nn.BatchNorm1d(hid_dim),
                             nn.ReLU(inplace=True), # hidden layer
                             nn.Linear(hid_dim, out_dim)) # output layer
    elif 'SimSiamMLP' in proj_type:
        print("[CL Loss] Type: SimSiamMLP, in_dim: {}, out_dim: {}, h_dim: {}, Device: {}".format(inp_dim, out_dim, hid_dim, device), end="")
        print(", Temperature: {}".format(temperature) if temperature != None else "\n")
        return nn.Sequential(
            nn.Linear(inp_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
            )
    else:
        raise RuntimeError("ContrastiveLoss: Error setting of the projective head")