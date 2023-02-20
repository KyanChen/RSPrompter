from . import common

import torch
import torch.nn as nn
from models import register
import torch.nn.functional as F
from argparse import Namespace

def make_model(args, parent=False):
    return VDSR(args)

@register('VDSR')
def VDSR(scale_ratio, rgb_range=1):
    args = Namespace()
    args.scale = [scale_ratio]
    args.n_colors = 3
    args.rgb_range = rgb_range
    return VDSR(args)


class VDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(VDSR, self).__init__()

        n_feats = 64
        kernel_size = 3

        m_head = [common.BasicBlock(conv, args.n_colors, n_feats, kernel_size, bias=True, bn=True)]

        layer_nums = 18
        m_body = [
            common.BasicBlock(conv, n_feats, n_feats, kernel_size, bias=True, bn=True)
            for _ in range(layer_nums)
        ]

        m_tail = [conv(n_feats, args.n_colors, kernel_size, bias=True)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, out_size):
        x = F.interpolate(x, size=out_size, mode='bicubic')
        residual = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        out = x + residual
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