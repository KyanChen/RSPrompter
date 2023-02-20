from models import register
from einops import rearrange
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .swin_neck import SwinTransformer, PatchEmbed, SwinBlockSequence


@register('Swin_backbone')
class SwinTransformerBackbone(nn.Module):
    def __init__(self, in_channels=3, embed_dims=256, depth=4, drop_path_rate=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=5,
            stride=1,
            padding='same',
            norm_cfg=dict(type='LN')
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]

        self.stage = SwinBlockSequence(
            embed_dims=embed_dims,
            num_heads=8,
            feedforward_channels=embed_dims * 2,
            depth=depth,
            window_size=4,
            drop_path_rate=dpr,
            downsample=None
        )

        self.norm_layer = build_norm_layer(dict(type='LN'), embed_dims)[1]
        self.out_dim = embed_dims

    def forward(self, x):

        x, hw_shape = self.patch_embed(x)
        x, hw_shape, out, out_hw_shape = self.stage(x, hw_shape)
        out = self.norm_layer(out)
        x = out.view(-1, *out_hw_shape, self.out_dim).permute(0, 3, 1, 2).contiguous()

        return x

