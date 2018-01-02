import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

from matplotlib import pyplot as plt
from layers import *

class CFConv1NetOnline(nn.Module):
    def __init__(self, fh, fw, rh, rw, in_num_channels, out_num_channels):
        super(CFConv1NetOnline, self).__init__()
        self.in_num_channels = in_num_channels
        self.out_num_channels = out_num_channels
        self.rw = rw
        self.rh = rh
        self.fw = fw
        self.fh = fh
        self.conv1 = nn.Conv2d(self.in_num_channels, self.out_num_channels, kernel_size=1, stride=1, padding=0)
        self.cf = nn.Conv2d(self.out_num_channels, 1, kernel_size=(self.fh,self.fw), 
                        stride=1, padding=(self.rh,self.rw))

    def forward(self, x, weights=None):
        x = self.conv1(x)
        x = self.cf(x)
        return x
    
    def copy_meta_weights(self, meta_weights):
        for k, p in self.named_parameters():
            if p.requires_grad and k in meta_weights.keys():
                p.data = meta_weights[k].data.clone()

class CFConv1Net(nn.Module):
    def __init__(self, target_sz, in_num_channels, out_num_channels):
        super(CFConv1Net, self).__init__()
        self.in_num_channels = in_num_channels
        self.out_num_channels = out_num_channels
        self.rw = int(np.ceil(target_sz[0]/2))
        self.rh = int(np.ceil(target_sz[1]/2))
        self.fw = int(self.rw*2 + 1)
        self.fh = int(self.rh*2 + 1)
        self.conv1 = nn.Conv2d(self.in_num_channels, self.out_num_channels, kernel_size=1, stride=1, padding=0)
        self.cf = nn.Conv2d(self.out_num_channels, 1, kernel_size=(self.fh,self.fw), 
                        stride=1, padding=(self.rh,self.rw))
        self.weight_init()

    def forward(self, x):
        x = conv2d(x, weights['conv1.weight'], weights['conv1.bias'],stride=1,padding=0)
        x = conv2d(x, weights['cf.weight'], weights['cf.bias'],stride=1,padding=(self.rh,self.rw))
        return x

    def weight_init(self):
        self.conv1.bias.data.zero_()
        self.cf.bias.data.zero_()
        self.conv1.weight.data = (self.conv1.weight.data.normal_()/np.sqrt(self.conv1.in_channels))/1e5
        self.cf.weight.data = (self.cf.weight.data.normal_()/np.sqrt(self.cf.in_channels*self.rh*self.rw))/1e5
    
    def copy_meta_weights(self, meta_weights):
        for k, p in self.named_parameters():
            if p.requires_grad and k in meta_weights.keys():
                p.data = meta_weights[k].clone()

class FeatureExtractor(nn.Module):
    def __init__(self, model_path=None):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(OrderedDict([
                ('conv1_1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
                ('relu1_1', nn.ReLU()),
                ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
                ('relu1_2', nn.ReLU()),
                ('pad1_2', nn.ZeroPad2d((0,1,0,1))),
                ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),

                ('conv2_1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
                ('relu2_1', nn.ReLU()),
                ('conv2_2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
                ('relu2_2', nn.ReLU()),
                ('pad2_2', nn.ZeroPad2d((0,1,0,1))),
                ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),

                ('conv3_1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
                ('relu3_1', nn.ReLU()),
                ('conv3_2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
                ('relu3_2', nn.ReLU()),
                ('conv3_3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
                ('relu3_3', nn.ReLU()),

                ('conv4_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
                ('relu4_1', nn.ReLU()),
                ('conv4_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
                ('relu4_2', nn.ReLU()),
                ('conv4_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
                ('relu4_3', nn.ReLU()),
        ]))
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
    
    def forward(self, x):
        x = self.features(x)
        return x

    def load_model(self, model_path):
        states = torch.load(model_path)
        features = states['features']
        fc = states['fc']
        self.features.load_state_dict(features)
        self.fc.load_state_dict(fc)

    def load_mat_model(self, matfile):
        from_indices = [0,2,5,7,10,12,14,17,19,21]
        to_indices = [0,2,6,8,12,14,16,18,20,22]
        mat = scipy.io.loadmat(matfile)
        mat_features = list(mat['layers'])[0]
        # copy conv weights
        for i in range(len(from_indices)):
            from_i = from_indices[i]
            to_i = to_indices[i]
            weight, bias = mat_features[from_i]['weights'].item()[0]
            self.features[to_i].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.features[to_i].bias.data = torch.from_numpy(bias[:,0])


class L2normLoss(nn.Module):
    def __init__(self):
        super(L2normLoss, self).__init__()
 
    def forward(self, X, C):
        abs_diff = (X - C).abs()
        mask = (abs_diff>=0.1).float()
        n_effect = mask.sum()+1 #1 for divide-by-zero error
        loss = (C.exp()*mask*(X-C)).pow(2).sum()/n_effect
        return loss
