from argparse import Namespace

import torch
import torch.nn as nn
from models import register
import torch.nn.functional as F

def make_model(args, parent=False):
    return CNN7(args)


@register('LGCNET')
def LGCNET(scale_ratio, rgb_range=1):
    args = Namespace()
    args.scale = [scale_ratio]
    args.n_colors = 3
    args.rgb_range = rgb_range
    return LGCNET(args)


class LGCNET(nn.Module):
    def __init__(self, args, nfeats = 32):
        super(LGCNET, self).__init__()
        self.conv1 = nn.Conv2d(args.n_colors, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(nfeats*3, nfeats*2, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv7 = nn.Conv2d(nfeats*2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu =  nn.ReLU()

    def forward(self, x, out_size):
        x = F.interpolate(x, out_size, mode='bicubic')
        residual = x
        im1 = self.relu(self.conv1(x))
        im2 = self.relu(self.conv2(im1))
        im3 = self.relu(self.conv3(im2))
        im4 = self.relu(self.conv4(im3))
        im5 = self.relu(self.conv5(im4))
        out = self.relu(self.conv6(torch.cat((im3, im4, im5), dim = 1)))
        out = self.conv7(out) + residual
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class CNN7(nn.Module):
    def __init__(self, args, nfeats = 32):
        super(CNN7, self).__init__()
        self.conv1 = nn.Conv2d(args.n_colors, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(nfeats, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu =  nn.ReLU()

    def forward(self, x):
        residual = x
        im1 = self.relu(self.conv1(x))
        im2 = self.relu(self.conv2(im1))
        im3 = self.relu(self.conv3(im2))
        im4 = self.relu(self.conv4(im3))
        im5 = self.relu(self.conv5(im4))
        im6 = self.relu(self.conv6(im5))
        out = self.conv7(im6) + residual
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))