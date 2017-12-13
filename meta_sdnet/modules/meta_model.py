import os
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from layers import *
import scipy.io

class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x


class MetaSDNetOnline(nn.Module):
    def __init__(self, model_path=None):
        super(MetaSDNetOnline, self).__init__()
        self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 96, kernel_size=7, stride=2)),
                ('relu1', nn.ReLU()),
                ('lrn1', LRN()),
                ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv2', nn.Conv2d(96, 256, kernel_size=5, stride=2)),
                ('relu2', nn.ReLU()),
                ('lrn2', LRN()),
                ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv3', nn.Conv2d(256, 512, kernel_size=3, stride=1)),
                ('relu3', nn.ReLU())
        ]))
        self.fc = nn.Sequential(OrderedDict([
                ('fc4',   nn.Linear(512 * 3 * 3, 512)),
                ('relu4', nn.ReLU()),
                ('fc5',   nn.Linear(512, 512)),
                ('relu5', nn.ReLU()),
                ('fc6',   nn.Linear(512, 2))
        ]))
        self.lrn1_f = LRN()
        self.lrn2_f = LRN()
    
    def forward(self, x, out_layer=None, in_layer=None):
        if (out_layer == None) and (in_layer == None):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif out_layer == 'features':
            x = self.features(x)
            x = x.view(x.size(0), -1)
        elif in_layer == 'fc':
            x = self.fc(x)
        return x
    
    def set_learnable_params(self, layers):
        for k,p in self.named_parameters():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if p.requires_grad:
                params[k] = p
        return params

    def copy_meta_weights(self, meta_weights):
        for k, p in self.named_parameters():
            if p.requires_grad and k in meta_weights.keys():
                p.data = meta_weights[k].data.clone()
        
class MetaSDNet(nn.Module):
    def __init__(self, model_path=None):
        super(MetaSDNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 96, kernel_size=7, stride=2)),
                ('relu1', nn.ReLU()),
                ('lrn1', LRN()),
                ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv2', nn.Conv2d(96, 256, kernel_size=5, stride=2)),
                ('relu2', nn.ReLU()),
                ('lrn2', LRN()),
                ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv3', nn.Conv2d(256, 512, kernel_size=3, stride=1)),
                ('relu3', nn.ReLU())
        ]))
        self.fc = nn.Sequential(OrderedDict([
                ('fc4',   nn.Linear(512 * 3 * 3, 512)),
                ('relu4', nn.ReLU()),
                ('fc5',   nn.Linear(512, 512)),
                ('relu5', nn.ReLU()),
                ('fc6',   nn.Linear(512, 2))
        ]))
        self.lrn1_f = LRN()
        self.lrn2_f = LRN()
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
    
    def forward(self, x, weights=None, out_layer=None, in_layer=None):
        if weights == None:
            if (out_layer == None) and (in_layer == None):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
            elif out_layer == 'features':
                x = self.features(x)
                x = x.view(x.size(0), -1)
            elif in_layer == 'fc':
                x = self.fc(x)
        else:
            if in_layer =='fc':
                x = linear(x, weights['fc.fc4.weight'], weights['fc.fc4.bias'])
                x = relu(x)
                x = linear(x, weights['fc.fc5.weight'], weights['fc.fc5.bias'])
                x = relu(x)
                x = linear(x, weights['fc.fc6.weight'], weights['fc.fc6.bias'])
            else:
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = linear(x, weights['fc.fc4.weight'], weights['fc.fc4.bias'])
                x = relu(x)
                x = linear(x, weights['fc.fc5.weight'], weights['fc.fc5.bias'])
                x = relu(x)
                x = linear(x, weights['fc.fc6.weight'], weights['fc.fc6.bias'])
        return x
    
    def set_learnable_params(self, layers):
        for k,p in self.named_parameters():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if p.requires_grad:
                params[k] = p
        return params

    def load_model(self, model_path):
        states = torch.load(model_path)
        features = states['features']
        fc = states['fc']
        self.features.load_state_dict(features)
        self.fc.load_state_dict(fc)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_features = list(mat['layers'])[0]
        # copy conv weights
        for i in range(3):
            weight, bias = mat_features[i*4]['weights'].item()[0]
            self.features[i*4].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.features[i*4].bias.data = torch.from_numpy(bias[:,0])
    
    def copy_meta_weights(self, meta_weights):
        for k, p in self.named_parameters():
            if p.requires_grad and k in meta_weights.keys():
                p.data = meta_weights[k].data.clone()
        
    def copy_init_weights(self, net_from):
        states_fc_from = net_from.fc.state_dict()
        states_features_from = net_from.features.state_dict()
        self.fc.load_state_dict(states_fc_from)
        self.features.load_state_dict(states_features_from)

class MetaSDNetFull(nn.Module):
    def __init__(self, model_path=None):
        super(MetaSDNetFull, self).__init__()
        self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 96, kernel_size=7, stride=2)),
                ('relu1', nn.ReLU()),
                ('lrn1', LRN()),
                ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv2', nn.Conv2d(96, 256, kernel_size=5, stride=2)),
                ('relu2', nn.ReLU()),
                ('lrn2', LRN()),
                ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv3', nn.Conv2d(256, 512, kernel_size=3, stride=1)),
                ('relu3', nn.ReLU())
        ]))
        self.fc = nn.Sequential(OrderedDict([
                ('fc4',   nn.Linear(512 * 3 * 3, 512)),
                ('relu4', nn.ReLU()),
                ('fc5',   nn.Linear(512, 512)),
                ('relu5', nn.ReLU()),
                ('fc6',   nn.Linear(512, 2))
        ]))
        self.lrn1_f = LRN()
        self.lrn2_f = LRN()
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
    
    def forward(self, x, weights=None, out_layer=None, in_layer=None):
        if weights == None:
            if (out_layer == None) and (in_layer == None):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
            elif out_layer == 'features':
                x = self.features(x)
                x = x.view(x.size(0), -1)
            elif in_layer == 'fc':
                x = self.fc(x)
        else:
            if in_layer =='fc':
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = linear(x, weights['fc.fc4.weight'], weights['fc.fc4.bias'])
                x = relu(x)
                x = linear(x, weights['fc.fc5.weight'], weights['fc.fc5.bias'])
                x = relu(x)
                x = linear(x, weights['fc.fc6.weight'], weights['fc.fc6.bias'])
            else:
                x = conv2d(x, weights['features.conv1.weight'], weights['features.conv1.bias'],stride=2)
                x = relu(x)
                x = self.lrn1_f(x)
                x = maxpool(x, kernel_size=3, stride=2)
                x = conv2d(x, weights['features.conv2.weight'], weights['features.conv2.bias'],stride=2)
                x = relu(x)
                x = self.lrn1_f(x)
                x = maxpool(x, kernel_size=3, stride=2)
                x = conv2d(x, weights['features.conv3.weight'], weights['features.conv3.bias'],stride=1)
                x = relu(x)
                x = x.view(x.size(0), -1)
                x = linear(x, weights['fc.fc4.weight'], weights['fc.fc4.bias'])
                x = relu(x)
                x = linear(x, weights['fc.fc5.weight'], weights['fc.fc5.bias'])
                x = relu(x)
                x = linear(x, weights['fc.fc6.weight'], weights['fc.fc6.bias'])
        return x
    
    def set_learnable_params(self, layers):
        for k,p in self.named_parameters():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if p.requires_grad:
                params[k] = p
        return params

    def load_model(self, model_path):
        states = torch.load(model_path)
        features = states['features']
        fc = states['fc']
        self.features.load_state_dict(features)
        self.fc.load_state_dict(fc)
    
    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_features = list(mat['layers'])[0]
        # copy conv weights
        for i in range(3):
            weight, bias = mat_features[i*4]['weights'].item()[0]
            self.features[i*4].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.features[i*4].bias.data = torch.from_numpy(bias[:,0])
    
    def copy_meta_weights(self, meta_weights):
        for k, p in self.named_parameters():
            if p.requires_grad and k in meta_weights.keys():
                p.data = meta_weights[k].data.clone()
        
class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
 
    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:,1]
        neg_loss = -F.log_softmax(neg_score)[:,0]
        
        loss = pos_loss.sum() + neg_loss.sum()
        return loss

class BinaryFocalLoss(nn.Module):
    def __init__(self):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = 2
 
    def forward(self, pos_score, neg_score):
        pos_p = F.softmax(pos_score)[:,1]
        neg_p = F.softmax(neg_score)[:,0]
        pos_logp = F.log_softmax(pos_score)[:,1]
        neg_logp = F.log_softmax(neg_score)[:,0]
#        pos_logp = pos_p.log()
#        neg_logp = neg_p.log()

        pos_loss = -torch.pow((1-pos_p),self.gamma)*pos_logp
        neg_loss = -torch.pow((1-neg_p),self.gamma)*neg_logp

        #pos_loss = -F.log_softmax(pos_score)[:,1]
        #neg_loss = -F.log_softmax(neg_score)[:,0]
        
        loss = pos_loss.sum() + neg_loss.sum()
        return loss

class Accuracy():
    def __call__(self, pos_score, neg_score):
        
        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()
        
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)
        tot_acc = (pos_correct+neg_correct) / (pos_score.size(0)+neg_score.size(0))

        return tot_acc.data[0], pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):
        
        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)
        
        return prec.data[0]
