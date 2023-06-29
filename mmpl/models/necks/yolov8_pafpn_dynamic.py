from typing import List, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.utils import ConfigType, OptMultiConfig

from mmpl.registry import MODELS
from mmyolo.models import CSPLayerWithTwoConv
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.necks.yolov5_pafpn import YOLOv5PAFPN
from abc import ABCMeta, abstractmethod
from mmengine.model import BaseModule

@MODELS.register_module()
class YOLOv8DyPAFPN(BaseModule, metaclass=ABCMeta):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 upsample_feats_cat_first: bool = True,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.num_csp_blocks = num_csp_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.upsample_feats_cat_first = upsample_feats_cat_first
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # build top-down blocks
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 1, -1):
            self.upsample_layers.append(self.build_upsample_layer(idx=idx, n_layers=len(in_channels)))
            self.top_down_layers.append(self.build_top_down_layer(idx))

        # build bottom-up blocks
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(0, len(in_channels) - 2):
            self.downsample_layers.append(self.build_downsample_layer(idx))
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        # build extra fpn layers
        self.extra_fpn_layers = nn.ModuleList()
        self.build_extra_fpnlayers()

    def build_extra_fpnlayers(self):
        idx = 0
        self.extra_fpn_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.extra_fpn_layers.append(
            CSPLayerWithTwoConv(
                 make_divisible((self.all_channels[idx - 1] + self.all_channels[idx]),
                                self.widen_factor),
                 make_divisible(self.all_out_channels[idx - 1], self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg)
        )

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        idx, n_layers = kwargs['idx'], kwargs['n_layers']
        if idx == n_layers - 1:
            return nn.Upsample(scale_factor=4, mode='bilinear')
        else:
            return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs: List[torch.Tensor]):
        """Forward function."""
        assert len(inputs) == 3
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        inner_outs = self.forward_fpn(reduce_outs)  # two inner_outs
        assert len(inner_outs) == 2
        inner_outs_extra_fpn = self.forward_extra_fpn(inner_outs[0], reduce_outs[0])
        outs = self.forward_pan([inner_outs_extra_fpn] + inner_outs)

        # # out_layers
        # results = []
        # for idx in range(len(self.in_channels)):
        #     results.append(self.out_layers[idx](outs[idx]))

        return tuple(inner_outs), tuple(outs)

    def forward_fpn(self, reduce_outs):
        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 1, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)
        return inner_outs

    def forward_extra_fpn(self, feat_high, feat_low):
        # top-down path
        upsample_feat = self.extra_fpn_layers[0](feat_high)
        if self.upsample_feats_cat_first:
            top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
        else:
            top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
        inner_out = self.extra_fpn_layers[1](top_down_layer_inputs)
        return inner_out

    def forward_pan(self, inner_outs):
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(0, len(self.in_channels) - 2):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)
        return outs

