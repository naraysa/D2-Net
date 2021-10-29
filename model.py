import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import copy
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0,0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1: 
        # m.weight.data.normal_(0,0.01)
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


class Model(torch.nn.Module):
    def __init__(self, n_feature, n_class, args, labels101to20=None):
        super(Model, self).__init__()
        self.labels20 = None
        self.activitynet = args.activity_net
        if labels101to20 is not None:
            self.labels20 = labels101to20
        self.n_class = n_class
        self.n_feature = n_feature
        n_featureby2 = int(n_feature/2)
        
        if self.activitynet:
            ksz, dil = 5, 2
        else:
            ksz, dil = 3, 1    
        pad = int((ksz-1)/2 * dil)
        
        self.conv = nn.Conv1d(n_feature, n_feature, kernel_size=ksz, padding=pad, dilation=dil, bias=True, groups=2)
        self.conv1 = nn.Conv1d(n_feature, n_feature, kernel_size=ksz, padding=pad, dilation=dil, bias=True, groups=2)
        
        self.relu = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        if self.activitynet:
            self.classifier = nn.Conv1d(n_feature, 2*n_class, kernel_size=5, padding=4, dilation=2, bias=True,groups=2)
        else:
            self.classifier = nn.Conv1d(n_feature, 2*n_class, kernel_size=1, padding=0, bias=True,groups=2)

        self.apply(weights_init)
        self.running_bg = nn.Parameter(data=torch.zeros(1,n_featureby2))
        self.dropout = nn.Dropout(0.7)
        

    def forward(self,inputs,device,is_training=True):
        #inputs - batch x seq_len x featSize
        inputs = inputs.permute([0,2,1])
        
        
        x1 = self.relu(self.conv(inputs)) + inputs
        x2 = self.relu(self.conv1(x1)) + x1
        x = self.dropout(x2) if is_training else x2
        cls_x = self.classifier(x)

        x = x.permute([0,2,1])
        cls_x = cls_x.permute([0,2,1])
        
        return x[:,:,1024:], cls_x[:,:,self.n_class:], x[:,:,:1024], cls_x[:,:,:self.n_class]

    
