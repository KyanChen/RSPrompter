import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import register
import numpy as np

class ExpansionNet(nn.Module):
    def __init__(self, args):
        super(ExpansionNet, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.hidden_list = args.hidden_list
        layers = []
        lastv = self.in_dim
        hidden_list = self.hidden_list
        out_dim = self.out_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        b, _, c = x.shape
        x = x.view(-1, c)
        logits = self.layers(x)
        out = nn.functional.normalize(logits, dim=1)
        return out.view(b,_,self.out_dim)


@register('ExpansionNet')
def make_ExpansionNet(in_dim=580,out_dim=10,hidden_list=None):
    args = Namespace()
    args.in_dim = in_dim
    args.out_dim = out_dim
    args.hidden_list = hidden_list
    return ExpansionNet(args)
