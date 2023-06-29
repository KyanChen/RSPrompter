import copy
import math
from typing import Type, Tuple

import einops
import mmcv.cnn.bricks.transformer
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import build_transformer_layer
from torch import Tensor

from mmdet.models import SinePositionalEncoding
from mmpl.registry import MODELS
import torch.nn.functional as F


@MODELS.register_module()
class SAMAdaptor(nn.Module):
    def __init__(
            self,
            in_channels=3,
            inner_dim=128,
            embed_dim=768,
            depth=10,
            out_channels=256,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_channels = out_channels

        self.stem_layer = StemLayer(
            in_chans=in_channels,
            inner_dim=inner_dim,
            embed_dim=embed_dim,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.depth = depth

        self.gate_layers = nn.ModuleList()
        for i in range(2*(depth+1)):
            # layer = nn.MultiheadAttention(
            #     embed_dim,
            #     num_heads=8,
            #     dropout=0.1,
            #     bias=True,
            #     batch_first=True
            # )
            layer = nn.Sequential(
                ConvModule(
                    in_channels=embed_dim,
                    out_channels=1,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=None,
                    act_cfg=None
                ),
                nn.Sigmoid()
            )
            self.gate_layers.append(layer)

        self.adap_layers = nn.ModuleList()
        for i in range(depth):
            layer = nn.Sequential(
                ConvModule(
                    in_channels=embed_dim,
                    out_channels=inner_dim,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                ),
                ConvModule(
                    in_channels=inner_dim,
                    out_channels=inner_dim,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                ),
                ConvModule(
                    in_channels=inner_dim,
                    out_channels=embed_dim,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

            self.adap_layers.append(layer)

        self.semantic_neck = nn.Sequential(
            ConvModule(
                in_channels=embed_dim,
                out_channels=out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=None
            )
        )
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs: torch.Tensor, sambackbone: nn.Module):

        x0 = sambackbone.patch_embed(inputs)  # B H W C
        B, H, W, C = x0.shape
        if sambackbone.pos_embed is not None:
            x0 = x0 + sambackbone.pos_embed

        x1 = self.stem_layer(inputs)  # B C H W

        # x0_rearrange = rearrange(x0, 'b h w c -> b (h w) c', h=H, w=W)
        # x1_rearrange = rearrange(x1, 'b c h w -> b (h w) c', h=H, w=W)
        # res_x0 = self.gate_layers[0](
        #     query=x0_rearrange,
        #     key=x0_rearrange,
        #     value=x1_rearrange)[0]
        # res_x0 = rearrange(res_x0, 'b (h w) c -> b h w c', h=H, w=W)
        # x0 = x0 + res_x0
        x0_rearrange = rearrange(x0, 'b h w c -> b c h w')
        res_x0 = self.gate_layers[0](x0_rearrange)*x1
        res_x0 = rearrange(res_x0, 'b c h w -> b h w c')
        x0 = x0 + res_x0

        # res_x1 = self.gate_layers[1](
        #     query=x1_rearrange,
        #     key=x1_rearrange,
        #     value=x0_rearrange)[0]
        # res_x1 = rearrange(res_x1, 'b (h w) c -> b c h w', h=H, w=W)
        # x1 = x1 + res_x1
        res_x1 = self.gate_layers[1](x1)*x0_rearrange
        x1 = x1 + res_x1

        assert len(sambackbone.blocks) == self.depth
        for idx, blk in enumerate(sambackbone.blocks):
            x0 = blk(x0)  # B H W C

            x1 = self.adap_layers[idx](x1) + x1  # B C H W

            # x0_rearrange = rearrange(x0, 'b h w c -> b (h w) c', h=H, w=W)
            # x1_rearrange = rearrange(x1, 'b c h w -> b (h w) c', h=H, w=W)
            # res_x0 = self.gate_layers[2*idx+2](
            #     query=x0_rearrange,
            #     key=x0_rearrange,
            #     value=x1_rearrange)[0]
            # res_x0 = rearrange(res_x0, 'b (h w) c -> b h w c', h=H, w=W)
            # x0 = x0 + res_x0
            #
            # res_x1 = self.gate_layers[2*idx+3](
            #     query=x1_rearrange,
            #     key=x1_rearrange,
            #     value=x0_rearrange)[0]
            # res_x1 = rearrange(res_x1, 'b (h w) c -> b c h w', h=H, w=W)
            # x1 = x1 + res_x1

            x0_rearrange = rearrange(x0, 'b h w c -> b c h w')
            res_x0 = self.gate_layers[2*idx+2](x0_rearrange)*x1
            res_x0 = rearrange(res_x0, 'b c h w -> b h w c')
            x0 = x0 + res_x0

            res_x1 = self.gate_layers[2*idx+3](x1)*x0_rearrange
            x1 = x1 + res_x1

        x0 = sambackbone.neck(x0.permute(0, 3, 1, 2))
        x1 = self.semantic_neck(x1)

        return x0, x1


class StemLayer(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        inner_dim: int = 128,
        embed_dim: int = 768,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            ConvModule(
                in_channels=in_chans,
                out_channels=inner_dim,
                kernel_size=7,
                stride=4,
                padding=3,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=inner_dim,
                out_channels=embed_dim,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x