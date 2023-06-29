import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmpl.registry import MODELS


@MODELS.register_module()
class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super().__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), identity=x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


