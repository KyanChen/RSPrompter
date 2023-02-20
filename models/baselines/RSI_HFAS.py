import time
from collections import OrderedDict

import torch
import torch.nn as nn
import math
import torchvision.utils as SI

def make_model(args, parent=False):
    return metafpn(args)


class Pos2Weight(nn.Module):
    def __init__(self, inC, kernel_size=3, outC=3):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.kernel_size * self.kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return out


class FPN(nn.Module):
    def __init__(self, G0, kSize=3):
        super(FPN, self).__init__()

        kSize1 = 1
        self.conv1 = RDB_Conv(G0, G0, kSize)
        self.conv2 = RDB_Conv(G0, G0, kSize)
        self.conv3 = RDB_Conv(G0, G0, kSize)
        self.conv4 = RDB_Conv(G0, G0, kSize)
        self.conv5 = RDB_Conv(G0, G0, kSize)
        self.conv6 = RDB_Conv(G0, G0, kSize)
        self.conv7 = RDB_Conv(G0, G0, kSize)
        self.conv8 = RDB_Conv(G0, G0, kSize)
        self.conv9 = RDB_Conv(G0, G0, kSize)
        self.conv10 = RDB_Conv(G0, G0, kSize)
        self.compress_in1 = nn.Conv2d(4 * G0, G0, kSize1, padding=(kSize1 - 1) // 2, stride=1)
        self.compress_in2 = nn.Conv2d(3 * G0, G0, kSize1, padding=(kSize1 - 1) // 2, stride=1)
        self.compress_in3 = nn.Conv2d(2 * G0, G0, kSize1, padding=(kSize1 - 1) // 2, stride=1)
        self.compress_in4 = nn.Conv2d(2 * G0, G0, kSize1, padding=(kSize1 - 1) // 2, stride=1)
        self.compress_out = nn.Conv2d(4 * G0, G0, kSize1, padding=(kSize1 - 1) // 2, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x11 = x + x4
        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x5_res = self.compress_in1(x5)
        x5 = self.conv5(x5_res)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x12 = x5_res + x7
        x8 = torch.cat((x5, x6, x7), dim=1)
        x8_res = self.compress_in2(x8)
        x8 = self.conv8(x8_res)
        x9 = self.conv9(x8)
        x13 = x8_res + x9
        x10 = torch.cat((x8, x9), dim=1)
        x10_res = self.compress_in3(x10)
        x10 = self.conv10(x10_res)
        x14 = x10_res + x10
        output = torch.cat((x11, x12, x13, x14), dim=1)
        output = self.compress_out(output)
        output = output + x
        return output


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResidualDenseBlock_8C(nn.Module):
    '''
    Residual Dense Block
    style: 8 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='relu',
                 mode='CNA'):
        super(ResidualDenseBlock_8C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv2 = ConvBlock(nc + gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv3 = ConvBlock(nc + 2 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv4 = ConvBlock(nc + 3 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv5 = ConvBlock(nc + 4 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv6 = ConvBlock(nc + 5 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv7 = ConvBlock(nc + 6 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv8 = ConvBlock(nc + 7 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv9 = ConvBlock(nc + 8 * gc, nc, 1, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                               act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1))
        x8 = self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1))
        x9 = self.conv9(torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8), 1))
        return x9.mul(0.2) + x


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0, \
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!' % sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)


def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, \
                act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!' % sys.modules[__name__]

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)


def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!' % pad_type)
    return layer


def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!' % norm_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        kSize = 3
        kSize1 = 1

        self.fpn1 = FPN(num_features)
        self.fpn2 = FPN(num_features)
        self.fpn3 = FPN(num_features)
        self.fpn4 = FPN(num_features)
        self.compress_in = nn.Conv2d(2 * num_features, num_features, kSize1, padding=(kSize1 - 1) // 2, stride=1)
        self.compress_out = nn.Conv2d(4 * num_features, num_features, kSize1, padding=(kSize1 - 1) // 2, stride=1)

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)  # tense拼接
        x = self.compress_in(x)

        fpn1 = self.fpn1(x)
        fpn2 = self.fpn2(fpn1)
        fpn3 = self.fpn3(fpn2)
        fpn4 = self.fpn4(fpn3)
        output = torch.cat((fpn1, fpn2, fpn3, fpn4), dim=1)
        output = self.compress_out(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True


class metafpn(nn.Module):
    def __init__(self,
                 RDNkSize=3,
                 G0=64,
                 n_colors=3,
                 act_type='prelu',
                 norm_type=None
                 ):
        super(metafpn, self).__init__()  # 第一句话，调用父类的构造函数，这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化。当然，如果初始化的逻辑与父类的不同，不使用父类的方法，自己重新初始化也是可以的。

        kernel_size = RDNkSize
        self.num_steps = 4
        self.num_features = G0
        self.scale_idx = 0
        self.scale = 1
        in_channels = n_colors
        num_groups = 6

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # LR feature extraction block
        self.conv_in = ConvBlock(in_channels, 4 * self.num_features,
                                        # 3×3Conv      一个卷积核产生一个feature map就是num_features
                                        kernel_size=3,
                                        act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * self.num_features, self.num_features,
                                        kernel_size=1,
                                        act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = FeedbackBlock(self.num_features, num_groups, self.scale, act_type, norm_type)

        # reconstruction block
        # uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        # self.out = DeconvBlock(num_features, num_features,
        #                        kernel_size=kernel_size, stride=stride, padding=padding,
        #                        act_type='prelu', norm_type=norm_type)
        self.P2W = Pos2Weight(inC=self.num_features)

    def repeat_x(self, x):
        scale_int = math.ceil(self.scale)
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)

        x = torch.cat([x] * scale_int, 3)
        x = torch.cat([x] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, pos_mat):
        self._reset_state()

        # x = self.sub_mean(x)
        scale_int = math.ceil(self.scale)
        # uncomment for pytorch 0.4.0
        # inter_res = self.upsample(x)

        # comment for pytorch 0.4.0
        inter_res = nn.functional.interpolate(x, scale_factor=scale_int, mode='bilinear', align_corners=False)

        x = self.conv_in(x)
        x = self.feat_in(x)

        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)
            
            #output1 = h.clone()
           # for i in range(60):
             #   output2 = output1[:,i:i+3,:,:]
              #  SI.save_image(output2,"results/result"+str(i)+".png")
            
            # meta###########################################
            local_weight = self.P2W(
                pos_mat.view(pos_mat.size(1), -1))  ###   (outH*outW, outC*inC*kernel_size*kernel_size)
            up_x = self.repeat_x(h)  ### the output is (N*r*r,inC,inH,inW)

            # N*r^2 x [inC * kH * kW] x [inH * inW]
            cols = nn.functional.unfold(up_x, 3, padding=1)
            scale_int = math.ceil(self.scale)

            cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                          1).permute(0, 1, 3, 4, 2).contiguous()

            local_weight = local_weight.contiguous().view(x.size(2), scale_int, x.size(3), scale_int, -1, 3).permute(1,
                                                                                                                     3,
                                                                                                                     0,
                                                                                                                     2,
                                                                                                                     4,
                                                                                                                     5).contiguous()
            local_weight = local_weight.contiguous().view(scale_int ** 2, x.size(2) * x.size(3), -1, 3)

            out = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
            out = out.contiguous().view(x.size(0), scale_int, scale_int, 3, x.size(2), x.size(3)).permute(0, 3, 4, 1, 5,
                                                                                                          2)
            out = out.contiguous().view(x.size(0), 3, scale_int * x.size(2), scale_int * x.size(3))

            h = torch.add(inter_res, out)
            # h = self.add_mean(h)
                 
            outs.append(h)

        return outs  # return output of every timesteps

    def _reset_state(self):
        self.block.reset_state()

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        self.scale = self.args.scale[scale_idx]