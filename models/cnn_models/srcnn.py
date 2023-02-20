from argparse import Namespace
import torch.nn as nn
from models import register
import torch.nn.functional as F


def make_model(args, parent=False):
    return SRCNN(args)


@register('SRCNN')
def SRCNN(scale_ratio=1, rgb_range=1):
    args = Namespace()
    args.scale = scale_ratio
    args.rgb_range = rgb_range
    args.n_colors = 3
    return SRCNN(args)


class SRCNN(nn.Module):
    def __init__(self, args):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(args.n_colors,  64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, args.n_colors, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.scale = args.scale

    def forward(self, x, out_size):
        x = F.interpolate(x, out_size, mode='bicubic')
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

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