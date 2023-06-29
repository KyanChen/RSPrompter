# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch.nn.init import xavier_uniform_

from mmdet.models.backbones.csp_darknet import CSPLayer, Focus
from mmdet.utils import ConfigType, OptMultiConfig

from mmpl.registry import MODELS
from mmyolo.models.layers import CSPLayerWithTwoConv, SPPFBottleneck
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.backbones.base_backbone import BaseBackbone
import torch.nn.functional as F

@MODELS.register_module()
class YOLOv8DyCSPDarknet(BaseBackbone):
    """CSP-Darknet backbone used in YOLOv8.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5}.
            Defaults to P5.
        last_stage_out_channels (int): Final layer output channel.
            Defaults to 1024.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to: 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (Union[dict,list[dict]], optional): Initialization config
            dict. Defaults to None.

    Example:
        >>> from mmyolo.models import YOLOv8CSPDarknet
        >>> import torch
        >>> model = YOLOv8CSPDarknet()
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # the final out_channels will be set according to the param.
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],

        'Dy': [
            # 第一个是stem layer
            [3, 32, 3, True, False],
            [32, 64, 6, True, False], [64, 128, 6, True, False],
            [128, 256, 6, True, False],
            # 最后一个是transformer head
            [256, 512, 3, True, True]
        ],
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (1, 2, 3),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        super().__init__(
            self.arch_settings[arch][1:],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        in_channels, out_channels, num_blocks, add_identity, use_spp = self.arch_setting[0]

        in_channels = in_channels
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = CSPLayerWithTwoConv(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)

        return nn.Sequential(*stage)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        kernel_size = 3 if stage_idx < 3 else 5
        stride = 2 if stage_idx < 3 else 4
        padding = 1 if stage_idx < 3 else 3

        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = CSPLayerWithTwoConv(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)
        if use_spp:
            # spp = SPPFBottleneck(
            #     out_channels,
            #     out_channels,
            #     kernel_sizes=5,
            #     norm_cfg=self.norm_cfg,
            #     act_cfg=self.act_cfg)
            # 更改为transformer
            spp = FusionFormer(
                out_channels,
            )
            stage.append(spp)
        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()


class FusionFormer(BaseModule):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 2,
                 dim_feedforward=None,
                 dropout: float = 0.1,
                 activation=F.gelu,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = True,
                 norm_first: bool = True,
                 init_cfg: OptMultiConfig = None
                 ):
        super().__init__(init_cfg)
        if dim_feedforward is None:
            dim_feedforward = 2 * d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, layer_norm_eps, batch_first,
            norm_first)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.encoder(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


