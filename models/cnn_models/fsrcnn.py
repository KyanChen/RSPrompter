
# code ref: https://github.com/yjn870/FSRCNN-pytorch/blob/master/models.py

from . import common
import math

from argparse import Namespace
import torch.nn as nn
from models import register
import torch.nn.functional as F


def make_model(args, parent=False):
    return FSRCNN(args)


@register('FSRCNN')
def FSRCNN(scale_ratio, rgb_range=1):
    args = Namespace()
    args.scale = [scale_ratio]
    args.n_colors = 3
    args.rgb_range = rgb_range
    return FSRCNN(args)

class FSRCNN(nn.Module):
    def __init__(self, args,  conv=common.default_conv, d=56, s=12 * 3, m=8):
        super(FSRCNN, self).__init__()

        scale = args.scale[0]
        act = nn.PReLU()

        m_first_part = []
        m_first_part.append(conv(args.n_colors, d, kernel_size=5))
        m_first_part.append(act)
        self.first_part = nn.Sequential(*m_first_part)

        m_mid_part = []
        m_mid_part.append(conv(d, s, kernel_size=1))
        m_mid_part.append(act)
        for _ in range(m):
            m_mid_part.append(conv(s, s, kernel_size=3))
            m_mid_part.append(act)
        m_mid_part.append(conv(s, d, kernel_size=1))
        m_mid_part.append(act)
        self.mid_part = nn.Sequential(*m_mid_part)

        self.last_part = nn.ConvTranspose2d(d, args.n_colors, kernel_size=9, stride=scale, padding=9//2,
                                            output_padding=scale-1)

        # self._initialize_weights()


    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x, out_size=None):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
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