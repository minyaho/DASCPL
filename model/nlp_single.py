import torch
import torch.nn as nn
from utils import SingleGPUModel
from .loss_fn import NLPContrastiveLoss, NLPLocalLoss
from .transformer.encoder import TransformerEncoder

class EncoderLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, f=None, n_heads=4, word_vec=None):
        super().__init__()

        self.mode = f
        print('[Layer] Enc mode:', self.mode)
        if f == 'emb':
            self.f = nn.Embedding(inp_dim, out_dim)
            if word_vec is not None:
                print("[Layer] Use pretrained embedding")
                self.f = nn.Embedding.from_pretrained(word_vec, freeze=False)
        elif f == 'lstm':
            self.f = nn.LSTM(inp_dim, out_dim, bidirectional=True, batch_first=True)
        elif f == 'trans':
            # dropout = 0.1
            # print("dropout:", dropout)
            self.f = TransformerEncoder(d_model=inp_dim, d_ff=out_dim, n_heads=n_heads,) #dropout=dropout)
        else:
            raise ValueError("EncoderLayer format setting error!")

    def forward(self, x, hidden=None, mask=None):
        if self.mode == 'emb':
            enc_x = self.f(x.long())
        elif self.mode == 'lstm':
            enc_x, (h, c) = self.f(x, hidden)
            # h = h.reshape(2, x.size(0), -1)
            hidden = (h, c)
        elif self.mode == 'trans':
            enc_x = self.f(x, mask=mask)
        return enc_x, hidden, mask

    def reduction(self, x, h=None, mask=None):
        # to match bridge function
        if self.mode == 'emb':
            return x.mean(1)
        elif self.mode == 'lstm':
            _h = h[0] + h[1]
            return _h
        elif self.mode == 'trans':
            denom = torch.sum(mask, -1, keepdim=True)
            feat = torch.sum(x * mask.unsqueeze(-1), dim=1) / denom
            return feat

class PredictLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, hid_dim=100, act_fun = nn.Tanh()):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(inp_dim, hid_dim), act_fun, nn.Linear(hid_dim, out_dim))

    def forward(self, x):
        return self.layer(x)

class LSTM_SCPL_3(SingleGPUModel):
    def __init__(self, configs):
        print("LSTM_SCPL_ - Use Embedding loss ver.")
        super().__init__()

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim'] 
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        n_heads = configs['head']
        word_vec = configs['word_vec']
        proj_type = configs['proj_type']

        self.layer0 = EncoderLayer(inp_dim=vocab_size, out_dim=emb_dim, f='emb', word_vec=word_vec) # embedding
        self.layer1 = EncoderLayer(inp_dim=emb_dim, out_dim=h_dim, f='lstm')                       # LSTM layer
        self.layer2 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
#         self.layer3 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
        self.layer4 = PredictLayer(inp_dim=h_dim, hid_dim=h_dim, out_dim=num_classes, act_fun=nn.Tanh())
        
        self.loss0 = NLPContrastiveLoss(inp_dim=emb_dim, hid_dim=emb_dim, out_dim=emb_dim, proj_type=proj_type)
        self.loss1 = NLPContrastiveLoss(inp_dim=h_dim, hid_dim=h_dim, out_dim=h_dim, proj_type=proj_type)
        self.loss2 = NLPContrastiveLoss(inp_dim=h_dim, hid_dim=h_dim, out_dim=h_dim, proj_type=proj_type)
#         self.loss3 = NLPContrastiveLoss(inp_dim=h_dim, hid_dim=h_dim, out_dim=h_dim)
        self.loss4 = nn.CrossEntropyLoss()

    def forward(self, x, y):
        loss = 0

        out0, _, _ = self.layer0(x)
        if self.training:
            loss += self.loss0(x=out0.mean(1), label=y)#x=out1[:, -1, :], label=y)
        
        out1, (h1, c1), _ = self.layer1(out0.detach())
        h_state1 = (h1[0] + h1[1])/2
        if self.training:
            loss += self.loss1(x=h_state1, label=y)#x=out1[:, -1, :], label=y)

        out2, (h2, c2), _ = self.layer2(out1.detach(), hidden=(h1.detach(), c1.detach()))
        h_state2 = (h2[0] + h2[1])/2
        if self.training:
            loss += self.loss2(x=h_state2, label=y)#x=out2[:, -1, :], label=y)

#         out3, (h3, c3), _ = self.layer3(out2.detach(), hidden=(h2.detach(), c2.detach()))
#         h_state3 = (h3[0] + h3[1])/2
#         if self.training:
#             loss += self.loss3(x=h_state3, label=y)

        out4 = self.layer4(h_state2.detach())
        if self.training:
            loss += self.loss4(out4, y)
        
        return out4, loss

class LSTM_BP_3(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim'] 
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        word_vec = configs['word_vec']

        self.layer0 = EncoderLayer(inp_dim=vocab_size, out_dim=emb_dim, f='emb', word_vec=word_vec) # embedding
        self.layer1 = EncoderLayer(inp_dim=emb_dim, out_dim=h_dim, f='lstm')                       # LSTM layer
        self.layer2 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
        # self.layer3 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
        self.layer4 = PredictLayer(inp_dim=h_dim, hid_dim=h_dim, out_dim=num_classes, act_fun=nn.Tanh())

        self.loss4 = nn.CrossEntropyLoss()

    def forward(self, x, y):
        loss = 0

        out0, _, _ = self.layer0(x)
        out1, (h1, c1), _ = self.layer1(out0)
        h_state1 = (h1[0] + h1[1])/2

        out2, (h2, c2), _ = self.layer2(out1, hidden=(h1, c1))
        h_state2 = (h2[0] + h2[1])/2

        # out3, hidden3 = self.layer3(out2, hidden=(h2, c2))
        # h_state3 = (h3[0] + h3[1])/2

        out4 = self.layer4(h_state2)
        if self.training:
            loss += self.loss4(out4, y)
        
        return out4, loss

class LSTM_BP_4(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim']
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        word_vec = configs['word_vec']
        
        self.layer0 = EncoderLayer(inp_dim=vocab_size, out_dim=emb_dim, f='emb', word_vec=word_vec) # embedding
        self.layer1 = EncoderLayer(inp_dim=emb_dim, out_dim=h_dim, f='lstm')                       # LSTM layer
        self.layer2 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
        self.layer3 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
        self.layer4 = PredictLayer(inp_dim=h_dim, hid_dim=h_dim, out_dim=num_classes, act_fun=nn.Tanh())

        self.loss4 = nn.CrossEntropyLoss()

    def forward(self, x, y):
        loss = 0

        out0, _, _ = self.layer0(x)
        out1, (h1, c1), _ = self.layer1(out0)
        h_state1 = (h1[0] + h1[1])/2

        out2, (h2, c2), _ = self.layer2(out1, hidden=(h1, c1))
        h_state2 = (h2[0] + h2[1])/2

        out3, (h3, c3), _ = self.layer3(out2, hidden=(h2, c2))
        h_state3 = (h3[0] + h3[1])/2

        out4 = self.layer4(h_state3)
        if self.training:
            loss += self.loss4(out4, y)
        
        return out4, loss

class LSTM_BP_d(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim']
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        word_vec = configs['word_vec']
        layer = configs['layers']

        assert (layer >= 2), 'The number of layers needs to be greater than 2. 1 is Embedding, the other is LSTM.'
        # assert (layer >= 1), 'The number of layers needs to be greater than 1.'

        self.layers = list()
        self.loss = list()
        self.num_layers = layer

        # Embedding layer
        self.layers.append(EncoderLayer(inp_dim=vocab_size, out_dim=emb_dim, f='emb', word_vec=word_vec))

        # Lstm layer
        for i in range(1, self.num_layers):
            if i == 1:
                self.layers.append(EncoderLayer(inp_dim=emb_dim, out_dim=h_dim, f='lstm'))
            else:
                self.layers.append(EncoderLayer(h_dim*2, h_dim, f='lstm'))

        # Predict layer
        self.layers.append(PredictLayer(inp_dim=h_dim, hid_dim=h_dim, out_dim=num_classes, act_fun=nn.Tanh()))

        # loss fn for predict layer
        self.loss.append(nn.CrossEntropyLoss())

        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x, y):
        loss = 0
        features = list()

        # Embedding layer
        features.append(self.layers[0](x))

        # LSTM layer
        for i in range(1, self.num_layers):
            if i == 1:
                features.append(self.layers[i](features[-1][0]))
            else:
                features.append(self.layers[i](features[-1][0], features[-1][1]))

        # Predict layer
        final_h = features[-1][1][0]
        h_state = (final_h[0] + final_h[1])/2

        out = self.layers[-1](h_state)
        if self.training:
            loss += self.loss[-1](out, y)
        
        return out, loss

class NLP_Block(nn.Module):
    def __init__(self, inp_dim, out_dim, f, h_dim=300, word_vec=None, n_heads=None, proj_type=None, pred_type=None, num_classes=None, device=None):
        super(NLP_Block, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.device = device

        self.layer = EncoderLayer(inp_dim=inp_dim, out_dim=out_dim, f=f, n_heads=n_heads, word_vec=word_vec) # embedding
        self._make_loss_layer(num_classes, proj_type, pred_type)

    def _make_loss_layer(self, num_classes, proj_type=None, pred_type=None):
        if (proj_type != None) or (pred_type != None):
            self.proj_type = proj_type
            self.pred_type = pred_type
            self.loss = NLPLocalLoss(temperature=0.1, input_dim=self.h_dim, hid_dim=self.h_dim, out_dim=self.h_dim,
                    num_classes=num_classes, proj_type=proj_type, pred_type=pred_type, device=self.device)

    def forward(self, x, hidden=None, mask=None):
        return self.layer(x, hidden, mask)

    def reduction(self, x, h=None, mask=None):
        return self.layer.reduction(x, h, mask)

class NLP_BP_Block(NLP_Block):
    def __init__(self, inp_dim, out_dim, f, h_dim=300, word_vec=None, n_heads=None, proj_type=None, pred_type=None, num_classes=None, device=None):
        super(NLP_BP_Block, self).__init__(inp_dim, out_dim, f, h_dim, word_vec, n_heads, proj_type, pred_type, num_classes, device)

class NLP_SCPL_Block(NLP_Block):
    def __init__(self, inp_dim, out_dim, f, h_dim=300, word_vec=None, n_heads=None, proj_type="i", pred_type=None, num_classes=None, device=None):
        super(NLP_SCPL_Block, self).__init__(inp_dim, out_dim, f, h_dim, word_vec, n_heads, proj_type, pred_type, num_classes, device)

class NLP_Predictor(nn.Module):
    def __init__(self, inp_dim, out_dim, hid_dim=100, act_fun = nn.Tanh(), device=None):
        super(NLP_Predictor, self).__init__()
        self.layer = nn.Sequential(nn.Linear(inp_dim, hid_dim), act_fun, nn.Linear(hid_dim, out_dim))
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x):
        return self.layer(x)

class LSTM_SCPL_REWRITE(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim'] 
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        word_vec = configs['word_vec']
        proj_type = configs['proj_type']

        layer_cfg = {
            0:{"inp_dim":vocab_size, "out_dim":emb_dim, "f":"emb", "h_dim":h_dim, "word_vec":word_vec, "proj_type":proj_type}, 
            1:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"lstm", "h_dim":h_dim, "word_vec":None, "proj_type":proj_type},
            2:{"inp_dim":h_dim*2, "out_dim":h_dim, "f":"lstm", "h_dim":h_dim, "word_vec":None, "proj_type":proj_type},
            4:{"inp_dim":h_dim, "out_dim":num_classes, "hid_dim":h_dim, "act_fun":nn.Tanh()}}
        self.layer0 = NLP_SCPL_Block(**layer_cfg[0])
        self.layer1 = NLP_SCPL_Block(**layer_cfg[1])
        self.layer2 = NLP_SCPL_Block(**layer_cfg[2])
        self.layer4 = NLP_Predictor(**layer_cfg[4])

    def forward(self, x, y):
        loss = 0

        out0, _, _ = self.layer0(x)
        out1, (h1, c1), _ = self.layer1(out0)
        h_state1 = (h1[0] + h1[1])/2
        if self.training:
            loss += self.layer1.loss(x=h_state1, label=y)#x=out1[:, -1, :], label=y)

        out2, (h2, c2), _ = self.layer2(out1.detach(), hidden=(h1.detach(), c1.detach()))
        h_state2 = (h2[0] + h2[1])/2
        if self.training:
            loss += self.layer2.loss(x=h_state2, label=y)#x=out2[:, -1, :], label=y)

#         out3, (h3, c3) = self.layer3(out2.detach(), hidden=(h2.detach(), c2.detach()))
#         h_state3 = (h3[0] + h3[1])/2
#         if self.training:
#             loss += self.loss3(x=h_state3, label=y)

        out4 = self.layer4(h_state2.detach())
        if self.training:
            loss += self.layer4.loss(out4, y)
        
        return out4, loss

class LSTM_SCPL_4(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()
        print("LSTM_SCPL_ - Use Embedding loss ver.")
        print("This is four layers .")

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim'] 
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        word_vec = configs['word_vec']
        proj_type = configs['proj_type']

        self.layer0 = EncoderLayer(inp_dim=vocab_size, out_dim=emb_dim, f='emb', word_vec=word_vec) # embedding
        self.layer1 = EncoderLayer(inp_dim=emb_dim, out_dim=h_dim, f='lstm')                       # LSTM layer
        self.layer2 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
        self.layer3 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
        self.layer4 = PredictLayer(inp_dim=h_dim, hid_dim=h_dim, out_dim=num_classes, act_fun=nn.Tanh())
        
        self.loss0 = NLPContrastiveLoss(inp_dim=emb_dim, hid_dim=emb_dim, out_dim=emb_dim, proj_type=proj_type)
        self.loss1 = NLPContrastiveLoss(inp_dim=h_dim, hid_dim=h_dim, out_dim=h_dim, proj_type=proj_type)
        self.loss2 = NLPContrastiveLoss(inp_dim=h_dim, hid_dim=h_dim, out_dim=h_dim, proj_type=proj_type)
        self.loss3 = NLPContrastiveLoss(inp_dim=h_dim, hid_dim=h_dim, out_dim=h_dim, proj_type=proj_type)
        self.loss4 = nn.CrossEntropyLoss()

    def forward(self, x, y):
        loss = 0

        # with torch.profiler.record_function("Forward - layer 0"):
        out0, _, _ = self.layer0(x)

        # with torch.profiler.record_function("Loss - layer 0"):
        if self.training:
            loss += self.loss0(x=out0.mean(1), label=y)#x=out1[:, -1, :], label=y)
    
        # with torch.profiler.record_function("Forward - layer 1"):
        out1, (h1, c1), _ = self.layer1(out0.detach())
        h_state1 = (h1[0] + h1[1])/2

        # with torch.profiler.record_function("Loss - layer 1"):
        if self.training:
            loss += self.loss1(x=h_state1, label=y)#x=out1[:, -1, :], label=y)

        # with torch.profiler.record_function("Forward - layer 2"):
        out2, (h2, c2), _ = self.layer2(out1.detach(), hidden=(h1.detach(), c1.detach()))
        h_state2 = (h2[0] + h2[1])/2

        # with torch.profiler.record_function("Loss - layer 2"):
        if self.training:
            loss += self.loss2(x=h_state2, label=y)#x=out2[:, -1, :], label=y)

        # with torch.profiler.record_function("Forward - layer 3"):
        out3, (h3, c3), _ = self.layer3(out2.detach(), hidden=(h2.detach(), c2.detach()))
        h_state3 = (h3[0] + h3[1])/2

        # with torch.profiler.record_function("Loss - layer 3"):
        if self.training:
            loss += self.loss3(x=h_state3, label=y)

        # with torch.profiler.record_function("Forward - layer 4"):
        out4 = self.layer4(h_state3.detach())

        # with torch.profiler.record_function("Loss - layer 4"):
        if self.training:
            loss += self.loss4(out4, y)
        
        return out4, loss

class LSTM_SCPL_5(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()

        print("LSTM_SCPL_ - Use Embedding loss ver.")
        print("This is 5 layers .")

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim'] 
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        word_vec = configs['word_vec']
        proj_type = configs['proj_type']
        
        self.layer0 = EncoderLayer(inp_dim=vocab_size, out_dim=emb_dim, f='emb', word_vec=word_vec) # embedding
        self.layer1 = EncoderLayer(inp_dim=emb_dim, out_dim=h_dim, f='lstm')                       # LSTM layer
        self.layer2 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
        self.layer3 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
        self.layer4 = EncoderLayer(h_dim*2, h_dim, f='lstm')                                      # LSTM layer
        self.layer5 = PredictLayer(inp_dim=h_dim, hid_dim=h_dim, out_dim=num_classes, act_fun=nn.Tanh())
        
        self.loss0 = NLPContrastiveLoss(inp_dim=emb_dim, hid_dim=emb_dim, out_dim=emb_dim, proj_type=proj_type)
        self.loss1 = NLPContrastiveLoss(inp_dim=h_dim, hid_dim=h_dim, out_dim=h_dim, proj_type=proj_type)
        self.loss2 = NLPContrastiveLoss(inp_dim=h_dim, hid_dim=h_dim, out_dim=h_dim, proj_type=proj_type)
        self.loss3 = NLPContrastiveLoss(inp_dim=h_dim, hid_dim=h_dim, out_dim=h_dim, proj_type=proj_type)
        self.loss4 = NLPContrastiveLoss(inp_dim=h_dim, hid_dim=h_dim, out_dim=h_dim, proj_type=proj_type)
        self.loss5 = nn.CrossEntropyLoss()

    def forward(self, x, y):
        loss = 0

        out0, _, _ = self.layer0(x)
        if self.training:
            loss += self.loss0(x=out0.mean(1), label=y)#x=out1[:, -1, :], label=y)
        
        out1, (h1, c1), _ = self.layer1(out0.detach())
        h_state1 = (h1[0] + h1[1])/2
        if self.training:
            loss += self.loss1(x=h_state1, label=y)#x=out1[:, -1, :], label=y)

        out2, (h2, c2), _ = self.layer2(out1.detach(), hidden=(h1.detach(), c1.detach()))
        h_state2 = (h2[0] + h2[1])/2
        if self.training:
            loss += self.loss2(x=h_state2, label=y)#x=out2[:, -1, :], label=y)

        out3, (h3, c3), _ = self.layer3(out2.detach(), hidden=(h2.detach(), c2.detach()))
        h_state3 = (h3[0] + h3[1])/2
        if self.training:
            loss += self.loss3(x=h_state3, label=y)

        out4, (h4, c4), _ = self.layer4(out3.detach(), hidden=(h3.detach(), c3.detach()))
        h_state4 = (h4[0] + h4[1])/2
        if self.training:
            loss += self.loss4(x=h_state4, label=y)

        out5 = self.layer5(h_state4.detach())
        if self.training:
            loss += self.loss5(out5, y)
        
        return out5, loss

class Trans_BP_3(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()
        print("Trans_ - 2 layers")

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim'] 
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        n_heads = configs['head']
        word_vec = configs['word_vec']

        layer_cfg = {
            0:{"inp_dim":vocab_size, "out_dim":emb_dim, "f":"emb", "h_dim":h_dim, "word_vec":word_vec}, 
            1:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":h_dim, "n_heads":n_heads, "word_vec":None},
            2:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":h_dim, "n_heads":n_heads, "word_vec":None},
            4:{"inp_dim":emb_dim, "out_dim":num_classes, "hid_dim":h_dim, "act_fun":nn.Tanh()}}
        self.layer0 = NLP_BP_Block(**layer_cfg[0])
        self.layer1 = NLP_BP_Block(**layer_cfg[1])
        self.layer2 = NLP_BP_Block(**layer_cfg[2])
        self.layer4 = NLP_Predictor(**layer_cfg[4])

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()
        
    def forward(self, x, y):
        loss = 0
        mask = self.get_mask(x)
        # print("mask:", mask.shape)
        # print("x:", x.shape)

        # Layer 0
        out0, hidden0, mask0 = self.layer0(x, mask=mask)
        # feature0 = self.layer0.reduction(out0, hidden0, mask0)
        # if self.training:
        #     loss += self.layer0.loss(x=feature0, label=y)

        # print(out0.shape, feature0.shape)

        # Layer 1
        out1, hidden1, mask1 = self.layer1(out0, hidden0, mask0)
        # feature1 = self.layer1.reduction(out1, hidden1, mask1)
        # if self.training:
        #     loss += self.layer1.loss(x=feature1, label=y)

        # print(out1.shape,feature1.shape)

        # Layer 2
        out2, hidden2, mask2 = self.layer2(out1, hidden1, mask1)
        feature2 = self.layer2.reduction(out2, hidden2, mask2)
        # if self.training:
        #     loss += self.layer2.loss(x=feature2, label=y)#x=out2[:, -1, :], label=y)

#         out3, (h3, c3) = self.layer3(out2.detach(), hidden=(h2.detach(), c2.detach()))
#         h_state3 = (h3[0] + h3[1])/2
#         if self.training:
#             loss += self.loss3(x=h_state3, label=y)


        # print(out2.shape,feature2.shape)
        out4 = self.layer4(feature2)
        # print()
        if self.training:
            loss += self.layer4.loss(out4, y)
        
        return out4, loss

class Trans_BP_4(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim'] 
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        n_heads = configs['head']
        word_vec = configs['word_vec']

        layer_cfg = {
            0:{"inp_dim":vocab_size, "out_dim":emb_dim, "f":"emb", "h_dim":h_dim, "word_vec":word_vec}, 
            1:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":h_dim, "n_heads":n_heads, "word_vec":None},
            2:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":h_dim, "n_heads":n_heads, "word_vec":None},
            3:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":h_dim, "n_heads":n_heads, "word_vec":None},
            4:{"inp_dim":emb_dim, "out_dim":num_classes, "hid_dim":h_dim, "act_fun":nn.Tanh()}}
        self.layer0 = NLP_BP_Block(**layer_cfg[0])
        self.layer1 = NLP_BP_Block(**layer_cfg[1])
        self.layer2 = NLP_BP_Block(**layer_cfg[2])
        self.layer3 = NLP_BP_Block(**layer_cfg[3])
        self.layer4 = NLP_Predictor(**layer_cfg[4])

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()
        
    def forward(self, x, y):
        loss = 0
        mask = self.get_mask(x)

        out0, hidden0, mask0 = self.layer0(x, mask=mask)

        # Layer 1
        out1, hidden1, mask1 = self.layer1(out0, hidden0, mask0)
        # feature1 = self.layer1.reduction(out1, hidden1, mask1)

        # Layer 2
        out2, hidden2, mask2 = self.layer2(out1, hidden1, mask1)
        # feature2 = self.layer2.reduction(out2, hidden2, mask2)

        # Layer 3
        out3, hidden3, mask3 = self.layer3(out2, hidden2, mask2)
        feature3 = self.layer3.reduction(out3, hidden3, mask3)

        out4 = self.layer4(feature3)

        if self.training:
            loss += self.layer4.loss(out4, y)
        
        return out4, loss

class Trans_BP_d(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim'] 
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        n_heads = configs['head']
        word_vec = configs['word_vec']
        layer = configs['layers']

        assert (layer >= 2), 'The number of layers needs to be greater than 2.'

        self.num_layers = layer
        self.layers = list()
        self.layer_cfg = dict()
        
        #  config 
        # Embedding layer
        self.layer_cfg[0] = {"inp_dim":vocab_size, "out_dim":emb_dim, "f":"emb", "h_dim":h_dim, "word_vec":word_vec}

        # Transformer layer
        for i in range(1, self.num_layers):
            self.layer_cfg[i] = {"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":h_dim, "n_heads":n_heads, "word_vec":None}

        # Predict layer
        self.layer_cfg[self.num_layers] = {"inp_dim":emb_dim, "out_dim":num_classes, "hid_dim":h_dim, "act_fun":nn.Tanh()}

        # Make 
        # Embedding layer
        self.layers.append(NLP_SCPL_Block(**self.layer_cfg[0]))

        # Transformer layer
        for i in range(1, self.num_layers):
            self.layers.append(NLP_SCPL_Block(**self.layer_cfg[i]))
        
        # Predict layer
        self.layers.append(NLP_Predictor(**self.layer_cfg[self.num_layers]))

        self.layers = torch.nn.Sequential(*self.layers)

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()
        
    def forward(self, x, y):
        loss = 0
        mask = self.get_mask(x)

        features = list()

        # Embedding layer
        features.append(self.layers[0](x, mask=mask))

        # Transformer layer
        for i in range(1, self.num_layers):
            features.append(self.layers[i](*features[-1]))

        # Predict layer
        final_feature = self.layers[-2].reduction(*features[-1])

        out = self.layers[-1](final_feature)

        if self.training:
            loss += self.layers[-1].loss(out, y)
        
        return out, loss

class Trans_SCPL_3(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()
        print("Trans_SCPL_ - 3 layers - Use Embedding loss ver.")

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim'] 
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        n_heads = configs['head']
        word_vec = configs['word_vec']
        proj_type = configs['proj_type']

        layer_cfg = {
            0:{"inp_dim":vocab_size, "out_dim":emb_dim, "f":"emb", "h_dim":emb_dim, "word_vec":word_vec, "proj_type":proj_type}, 
            1:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":emb_dim, "n_heads":n_heads, "word_vec":None, "proj_type":proj_type},
            2:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":emb_dim, "n_heads":n_heads, "word_vec":None, "proj_type":proj_type},
            4:{"inp_dim":emb_dim, "out_dim":num_classes, "hid_dim":h_dim, "act_fun":nn.Tanh()}}
        self.layer0 = NLP_SCPL_Block(**layer_cfg[0])
        self.layer1 = NLP_SCPL_Block(**layer_cfg[1])
        self.layer2 = NLP_SCPL_Block(**layer_cfg[2])
        self.layer4 = NLP_Predictor(**layer_cfg[4])

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()
        
    def forward(self, x, y):
        loss = 0
        mask = self.get_mask(x)
        # print("mask:", mask.shape)
        # print("x:", x.shape)

        # Layer 0
        out0, hidden0, mask0 = self.layer0(x, mask=mask)
        feature0 = self.layer0.reduction(out0, hidden0, mask0)
        if self.training:
            loss += self.layer0.loss(x=feature0, label=y)

        # print(out0.shape, feature0.shape)

        # Layer 1
        out1, hidden1, mask1 = self.layer1(out0.detach(), hidden0, mask0)
        feature1 = self.layer1.reduction(out1, hidden1, mask1)
        if self.training:
            loss += self.layer1.loss(x=feature1, label=y)

        # print(out1.shape,feature1.shape)

        # Layer 2
        out2, hidden2, mask2 = self.layer2(out1.detach(), hidden1, mask1)
        feature2 = self.layer2.reduction(out2, hidden2, mask2)
        if self.training:
            loss += self.layer2.loss(x=feature2, label=y)#x=out2[:, -1, :], label=y)

#         out3, (h3, c3) = self.layer3(out2.detach(), hidden=(h2.detach(), c2.detach()))
#         h_state3 = (h3[0] + h3[1])/2
#         if self.training:
#             loss += self.loss3(x=h_state3, label=y)


        # print(out2.shape,feature2.shape)
        out4 = self.layer4(feature2.detach())
        # print()
        if self.training:
            loss += self.layer4.loss(out4, y)
        
        return out4, loss

class Trans_SCPL_4(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()
        print("Trans_SCPL_ - 4 layers Transformer - Use Embedding loss ver.")

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim']
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        n_heads = configs['head']
        word_vec = configs['word_vec']
        proj_type = configs['proj_type']

        layer_cfg = {
            0:{"inp_dim":vocab_size, "out_dim":emb_dim, "f":"emb", "h_dim":emb_dim, "word_vec":word_vec, "proj_type":proj_type}, 
            1:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":emb_dim, "n_heads":n_heads, "word_vec":None, "proj_type":proj_type},
            2:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":emb_dim, "n_heads":n_heads, "word_vec":None, "proj_type":proj_type},
            3:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":emb_dim, "n_heads":n_heads, "word_vec":None, "proj_type":proj_type},
            4:{"inp_dim":emb_dim, "out_dim":num_classes, "hid_dim":h_dim, "act_fun":nn.Tanh()}}
        
        self.layer0 = NLP_SCPL_Block(**layer_cfg[0])
        self.layer1 = NLP_SCPL_Block(**layer_cfg[1])
        self.layer2 = NLP_SCPL_Block(**layer_cfg[2])
        self.layer3 = NLP_SCPL_Block(**layer_cfg[3])
        self.layer4 = NLP_Predictor(**layer_cfg[4])

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask
        
    def forward(self, x, y):
        loss = 0
        mask = self.get_mask(x)

        # Layer 0
        out0, hidden0, mask0 = self.layer0(x, mask=mask)
        feature0 = self.layer0.reduction(out0, hidden0, mask0)
        if self.training:
            loss += self.layer0.loss(x=feature0, label=y)

        # Layer 1
        out1, hidden1, mask1 = self.layer1(out0.detach(), hidden0, mask0)
        feature1 = self.layer1.reduction(out1, hidden1, mask1)
        if self.training:
            loss += self.layer1.loss(x=feature1, label=y)

        # Layer 2
        out2, hidden2, mask2 = self.layer2(out1.detach(), hidden1, mask1)
        feature2 = self.layer2.reduction(out2, hidden2, mask2)
        if self.training:
            loss += self.layer2.loss(x=feature2, label=y)

        # Layer 3
        out3, hidden3, mask3 = self.layer3(out2.detach(), hidden2, mask2)
        feature3 = self.layer3.reduction(out3, hidden3, mask3)
        if self.training:
            loss += self.layer3.loss(x=feature3, label=y)

        # Layer 4
        out4 = self.layer4(feature3.detach())
        if self.training:
            loss += self.layer4.loss(out4, y)
        
        return out4, loss

class Trans_SCPL_5(SingleGPUModel):
    def __init__(self, configs):
        super().__init__()
        print("Trans_SCPL_ - 5 layers Transformer - Use Embedding loss ver.")

        vocab_size = configs["vocab_size"]
        emb_dim = configs['emb_dim']
        h_dim = configs['h_dim']
        num_classes = configs['n_classes']
        n_heads = configs['head']
        word_vec = configs['word_vec']
        proj_type = configs['proj_type']

        layer_cfg = {
            0:{"inp_dim":vocab_size, "out_dim":emb_dim, "f":"emb", "h_dim":emb_dim, "word_vec":word_vec, "proj_type":proj_type}, 
            1:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":emb_dim, "n_heads":n_heads, "word_vec":None, "proj_type":proj_type},
            2:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":emb_dim, "n_heads":n_heads, "word_vec":None, "proj_type":proj_type},
            3:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":emb_dim, "n_heads":n_heads, "word_vec":None, "proj_type":proj_type},
            4:{"inp_dim":emb_dim, "out_dim":h_dim, "f":"trans", "h_dim":emb_dim, "n_heads":n_heads, "word_vec":None, "proj_type":proj_type},
            5:{"inp_dim":h_dim, "out_dim":num_classes, "hid_dim":h_dim, "act_fun":nn.Tanh()}}
        self.layer0 = NLP_SCPL_Block(**layer_cfg[0])
        self.layer1 = NLP_SCPL_Block(**layer_cfg[1])
        self.layer2 = NLP_SCPL_Block(**layer_cfg[2])
        self.layer3 = NLP_SCPL_Block(**layer_cfg[3])
        self.layer4 = NLP_SCPL_Block(**layer_cfg[4])
        self.layer5 = NLP_Predictor(**layer_cfg[5])

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()
        
    def forward(self, x, y):
        loss = 0
        mask = self.get_mask(x)

        # Layer 0
        out0, hidden0, mask0 = self.layer0(x, mask=mask)
        feature0 = self.layer0.reduction(out0, hidden0, mask0)
        if self.training:
            loss += self.layer0.loss(x=feature0, label=y)

        # Layer 1
        out1, hidden1, mask1 = self.layer1(out0.detach(), mask=mask0.detach())
        feature1 = self.layer1.reduction(out1, hidden1, mask1)
        if self.training:
            loss += self.layer1.loss(x=feature1, label=y)

        # Layer 2
        out2, hidden2, mask2 = self.layer2(out1.detach(), mask=mask1.detach())
        feature2 = self.layer2.reduction(out2, hidden2, mask2)
        if self.training:
            loss += self.layer2.loss(x=feature2, label=y)

        # Layer 3
        out3, hidden3, mask3 = self.layer3(out2.detach(), mask=mask2.detach())
        feature3 = self.layer3.reduction(out3, hidden3, mask3)
        if self.training:
            loss += self.layer3.loss(x=feature3, label=y)

        # Layer 4
        out4, hidden4, mask4 = self.layer4(out3.detach(), mask=mask3.detach())
        feature4 = self.layer4.reduction(out4, hidden4, mask4)
        if self.training:
            loss += self.layer4.loss(x=feature4, label=y)

        # Layer 5
        out5 = self.layer5(feature3.detach())
        if self.training:
            loss += self.layer5.loss(out5, y)
        
        return out5, loss