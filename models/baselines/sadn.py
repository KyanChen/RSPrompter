import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.utils import weight_norm
import time
from os.path import exists
import os

from . import upsampler
from .dynamic_layers import ScaleAwareDynamicConv2d
from easydict import EasyDict
import models
from models import register


def spatial_fold(input, fold):
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.view(batch, channel, h_fold, fold, w_fold, fold)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, -1, h_fold, w_fold)
    )


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, -1, h_unfold, w_unfold)
    )


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    # logger.warning("The module is deprecated, and will be removed in the future! ")
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class WeightNormedConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        act=nn.ReLU(True),
    ):
        conv = weight_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                bias=bias,
            )
        )
        m = [conv]
        if act:
            m.append(act)
        super().__init__(*m)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        if len(rgb_std) != len(rgb_mean):
            assert len(rgb_std) == 1
            rgb_std = rgb_std * len(rgb_mean)
        channel = len(rgb_mean)
        super(MeanShift, self).__init__(channel, channel, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(channel).view(channel, channel, 1, 1)
        self.weight.data.div_(std.view(channel, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):

        m = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size // 2),
                stride=stride,
                bias=bias,
            )
        ]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WideConvBlock(nn.Module):
    def __init__(self, num_features, kernel_size, width_multiplier=4, reduction=4):
        super().__init__()

        self.body = nn.Sequential(
            *[
                WeightNormedConv(
                    num_features, int(num_features * width_multiplier), 3
                ),
                WeightNormedConv(
                    int(num_features * width_multiplier), num_features, 3, act=None
                ),
                WeightNormedConv(
                    num_features,
                    num_features,
                    kernel_size,
                    act=None,
                    # res_scale=res_scale,
                ),
                SEBlock(num_features, reduction),
            ]
        )

    def forward(self, x, scale):
        return x + self.body(x)


class DynamicWideConvBlock(nn.Module):
    def __init__(
        self,
        num_features,
        kernel_size,
        width_multiplier=4,
        dynamic_K=4,
        reduction=4,
    ):
        super().__init__()

        self.body = nn.Sequential(
            *[
                WeightNormedConv(
                    num_features,
                    int(num_features * width_multiplier),
                    kernel_size,
                    # res_scale=2.0,
                ),
                WeightNormedConv(
                    int(num_features * width_multiplier),
                    num_features,
                    kernel_size,
                    act=None,
                ),
            ]
        )
        self.d_conv = weight_norm(
            ScaleAwareDynamicConv2d(
                num_features,
                num_features,
                kernel_size,
                padding=kernel_size // 2,
                K=dynamic_K,
            )
        )
        self.se_block = SEBlock(num_features, reduction)

    def forward(self, x, scale):
        r = self.body(x)
        r = self.d_conv(r, scale)
        r = self.se_block(r)
        return x + r


class LocalDenseGroup(nn.Module):
    def __init__(
        self,
        num_features,
        width_multiplier,
        num_layers,
        reduction,
        use_dynamic_conv,
        dynamic_K,
    ):
        super().__init__()
        kSize = 3
        self.num_layers = num_layers

        self.ConvBlockList = nn.ModuleList()
        self.compressList = nn.ModuleList()
        self.use_dynamic_conv = use_dynamic_conv
        for idx in range(num_layers):
            if use_dynamic_conv:
                self.ConvBlockList.append(
                    DynamicWideConvBlock(
                        num_features,
                        kSize,
                        width_multiplier=width_multiplier,
                        # res_scale=1 / math.sqrt(num_layers),
                        dynamic_K=dynamic_K,
                        reduction=reduction,
                    )
                )
            else:
                self.ConvBlockList.append(
                    WideConvBlock(
                        num_features,
                        kSize,
                        width_multiplier=width_multiplier,
                        # res_scale=1 / math.sqrt(num_layers),
                        reduction=reduction,
                    )
                )
        for idx in range(1, num_layers):
            self.compressList.append(
                WeightNormedConv(
                    (idx + 1) * num_features, num_features, 1, act=None
                )
            )

    def forward(self, x, scale):
        concat = x
        for l in range(self.num_layers):
            if l == 0:
                out = self.ConvBlockList[l](concat, scale)
            else:
                concat = torch.cat([concat, out], dim=1)
                out = self.compressList[l - 1](concat)
                out = self.ConvBlockList[l](out, scale)
        return out


class FeedbackBlock(nn.Module):
    def __init__(
        self,
        num_features,
        width_multiplier,
        num_layers,
        num_groups,
        reduction,
        use_dynamic_conv,
        dynamic_K,
    ):
        super().__init__()
        kSize = 3
        self.num_groups = num_groups

        self.LDGList = nn.ModuleList()
        for _ in range(num_groups):
            self.LDGList.append(
                LocalDenseGroup(
                    num_features,
                    width_multiplier,
                    num_layers,
                    reduction,
                    use_dynamic_conv,
                    dynamic_K,
                )
            )

        self.compressList = nn.ModuleList()
        for idx in range(1, num_groups):
            self.compressList.append(
                WeightNormedConv(
                    (idx + 1) * num_features, num_features, 1, act=None
                )
            )

        self.compress_in = WeightNormedConv(
            2 * num_features, num_features, kSize
        )

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x, scale):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size(), device=x.device)
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), 1)

        concat = self.compress_in(x)
        for l in range(self.num_groups):
            if l == 0:
                out = self.LDGList[l](concat, scale)
            else:
                concat = torch.cat([concat, out], dim=1)
                out = self.compressList[l - 1](concat)
                out = self.LDGList[l](out, scale)

        self.last_hidden = out
        return out

    def reset_state(self):
        self.should_reset = True


@register('sadnarc')
class SADN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_layers=4,
        num_groups=4,
        reduction=4,
        width_multiplier=4,
        interpolate_mode='bilinear',
        levels=4,
        use_dynamic_conv=True,
        dynamic_K=3,
        which_uplayer="UPLayer_MS_WN",
        uplayer_ksize=3,
        rgb_range=1,
        # rgb_mean=[0.5, 0.5, 0.5],
        # rgb_std=[0.5, 0.5, 0.5],
        *args,
        **kwargs
    ):
        super().__init__()
        kernel_size = 3
        skip_kernel_size = 5
        num_inputs = in_channels
        n_feats = num_features
        self.interpolate_mode = interpolate_mode
        self.levels = levels

        # self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        # self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(
            *[WeightNormedConv(num_inputs, num_features, kernel_size)]
        )

        self.body = FeedbackBlock(
            num_features,
            width_multiplier,
            num_layers,
            num_groups,
            reduction,
            use_dynamic_conv,
            dynamic_K,
        )

        self.tail = nn.Sequential(
            *[
                WeightNormedConv(
                    num_features, num_features, kernel_size, act=None
                )
            ]
        )

        self.skip = WeightNormedConv(
            num_inputs, num_features, skip_kernel_size, act=None
        )

        UpLayer = getattr(upsampler, which_uplayer)
        self.uplayer = UpLayer(
            n_feats,
            uplayer_ksize,
            out_channels,
            interpolate_mode,
            levels,
        )

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, ScaleAwareDynamicConv2d):
                m.update_temperature()

    def forward(self, x, out_size):
        self.body.reset_state()
        if isinstance(out_size, int):
            out_size = [out_size, out_size]
        scale = torch.tensor([x.shape[2] / out_size[0]], device=x.device)
        # x = self.sub_mean(x)
        skip = self.skip(x)

        x = self.head(x)
        h_list = []

        for _ in range(self.levels):
            h = self.body(x, scale)
            h = self.tail(h)
            h = h + skip
            h_list.append(h)

        x = self.uplayer(h_list, out_size)

        # x = self.add_mean(x)

        return x


class SADN_vis(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_features,
        num_layers,
        num_groups,
        reduction,
        width_multiplier,
        interpolate_mode,
        levels,
        use_dynamic_conv,
        dynamic_K,
        which_uplayer,
        uplayer_ksize,
        rgb_range,
        rgb_mean,
        rgb_std,
    ):
        super().__init__()
        kernel_size = 3
        skip_kernel_size = 5
        num_inputs = in_channels
        n_feats = num_features
        self.interpolate_mode = interpolate_mode
        self.levels = levels

        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(
            *[WeightNormedConv(num_inputs, num_features, kernel_size)]
        )

        self.use_dynamic_conv = use_dynamic_conv
        self.body = FeedbackBlock(
            num_features,
            width_multiplier,
            num_layers,
            num_groups,
            reduction,
            use_dynamic_conv,
            dynamic_K,
        )

        self.tail = nn.Sequential(
            *[
                WeightNormedConv(
                    num_features, num_features, kernel_size, act=None
                )
            ]
        )

        self.skip = WeightNormedConv(
            num_inputs, num_features, skip_kernel_size, act=None
        )

        UpLayer = getattr(upsampler, which_uplayer)
        self.uplayer = UpLayer(
            n_feats,
            uplayer_ksize,
            out_channels,
            interpolate_mode,
            levels,
        )

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, ScaleAwareDynamicConv2d):
                m.update_temperature()

    def forward(self, x, out_size):
        self.body.reset_state()
        if isinstance(out_size, int):
            out_size = [out_size, out_size]
        scale = torch.tensor([x.shape[2] / out_size[0]], device=x.device)
        x = self.sub_mean(x)
        skip = self.skip(x)

        x = self.head(x)
        h_list = []

        for _ in range(self.levels):
            h = self.body(x, scale)
            h = self.tail(h)
            h = h + skip
            h_list.append(h)
        vis = torch.mean(h_list[-1], dim=1)
        vis = (vis - vis.min()) / (vis.max() - vis.min())
        vis = vis[..., 88:217, 32:161]
        # vis = vis + 0.2
        # vis.clamp_max_(1)
        print(torch.min(vis), torch.max(vis))
        # print(vis.shape)

        savepath = "logs/vis"
        filename = "geo_residential_t7.png"

        if self.use_dynamic_conv:
            savepath = os.path.join(savepath, "dy" + filename.replace(".png", ""))
        else:
            savepath = os.path.join(savepath, "wo_dy" + filename.replace(".png", ""))
        if not exists(savepath):
            os.mkdir(savepath)

        savepath = os.path.join(savepath, "x{0}.png".format(int((1 / scale).item())))

        plt.imsave(savepath, vis.cpu().numpy()[0], cmap="hsv")

        x = self.uplayer(h_list, out_size)

        x = self.add_mean(x)

        return x

@register('edsr-sadn')
class EDSR_MS(nn.Module):
    def __init__(
        self,
        n_resblocks=16,
        n_feats=64,
        in_channels=3,
        out_channels=3,
        res_scale=1,
        which_uplayer="UPLayer_MS_WN",
        uplayer_ksize=3,
        interpolate_mode='bilinear',
        levels=4,
            *args,
            **kwargs
    ):
        super().__init__()

        conv = default_conv

        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        UpLayer = getattr(upsampler, which_uplayer)
        self.tail = UpLayer(
            n_feats,
            uplayer_ksize,
            out_channels,
            interpolate_mode,
            levels,
        )

    def forward(self, x, out_size):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res, out_size)

        return x


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(
            *[nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1), nn.ReLU()]
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(
        self,
        scale,
        num_features,
        num_blocks,
        num_layers,
        rgb_range,
        in_channels,
        out_channels,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        r = scale
        G0 = num_features
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = [num_blocks, num_layers, num_features]
        # self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        # self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(
            in_channels, G0, kSize, padding=(kSize - 1) // 2, stride=1
        )
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))

        # Global Feature Fusion
        self.GFF = nn.Sequential(
            *[
                nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            ]
        )

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(
                *[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(
                        G, out_channels, kSize, padding=(kSize - 1) // 2, stride=1
                    ),
                ]
            )
        elif r == 4:
            self.UPNet = nn.Sequential(
                *[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(
                        G, out_channels, kSize, padding=(kSize - 1) // 2, stride=1
                    ),
                ]
            )

    def forward(self, x, return_features=False):
        # x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        feat = x + f__1

        out = self.UPNet(feat)
        # out = self.add_mean(out)

        if return_features:
            return out, feat
        return out


@register('rdn-sadn')
class RDN_MS(RDN):
    """
    The multi scale version of RDN, and you can specify rgb_mean/rgb_std/rgb_range!
    """

    def __init__(self, **args):
        args = EasyDict(args)
        args.num_features = 64
        args.num_blocks = 16
        args.num_layers = 8
        args.rgb_range = 1
        args.in_channels = 3
        args.out_channels = 3
        args.which_uplayer = "UPLayer_MS_V9"
        args.uplayer_ksize = 3
        args.width_multiplier = 4
        args.interpolate_mode = 'bilinear'
        args.levels = 4
        super().__init__(
            scale=0,
            num_features=args.num_features,
            num_blocks=args.num_blocks,
            num_layers=args.num_layers,
            rgb_range=args.rgb_range,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
        )
        # Redefine up-sampling net
        UpLayer = getattr(upsampler, args.which_uplayer)
        self.UPNet = UpLayer(
            args.num_features, 3, args.out_channels, args.interpolate_mode, args.levels
        )

        rgb_mean = args.get("rgb_mean", (0.4488, 0.4371, 0.4040))
        rgb_std = args.get("rgb_std", (1.0, 1.0, 1.0))
        rgb_range = args.get("rgb_range")
        # self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        # self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x, out_size):
        # x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        x = self.UPNet(x, out_size)
        # x = self.add_mean(x)
        return x
