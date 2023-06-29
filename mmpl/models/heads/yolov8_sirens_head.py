import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from mmcv.cnn import ConvModule

from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.utils import multi_apply
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig, InstanceList)
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmpl.registry import MODELS, TASK_UTILS
from mmyolo.models import YOLOv8Head
from ..utils import gt_instances_preprocess, make_divisible
from mmyolo.models.dense_heads.yolov5_head import YOLOv5Head
from torch.nn import functional as F
from mmpl.models.necks.sirens import ModulatedSirens


@MODELS.register_module()
class YOLOv8SIRENSHeadModule(BaseModule):
    """YOLOv8HeadModule head module used in `YOLOv8`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max-1}``
            in QFL setting. Defaults to 16.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence] = [128, 256, 512],
                 widen_factor: float = 1.0,
                 reg_max: int = 16,

                 num_inner_layers: int = 3,
                 modulation_dim: int = 256,
                 func_size: tuple = (64, 64),
                 target_size: tuple = (512, 512),
                 interp_func: str = 'bilinear',
                 
                 encoder_rgb: bool = True,
                 encode_scale_ratio: bool = True,
                 encode_hr_coord: bool = True,
                 decouple_head: bool = True,

                 featmap_strides: Sequence[int] = [8, 16, 32],
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = [int(x*widen_factor) for x in in_channels]

        self.widen_factor = widen_factor
        self.reg_max = reg_max
        self.modulation_dim = modulation_dim
        self.func_size = func_size
        self.target_size = target_size
        self.interp_func = interp_func
        self.encoder_rgb = encoder_rgb
        self.encode_scale_ratio = encode_scale_ratio
        self.encode_hr_coord = encode_hr_coord
        self.num_inner_layers = num_inner_layers
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.in_channels)
        self.decouple_head = decouple_head


        decoder_in_dim = 2
        if self.encode_scale_ratio:
            decoder_in_dim += 2
        if self.encode_hr_coord:
            decoder_in_dim += 2
        self.decoder_in_dim = decoder_in_dim
        
        self.modulation_base_dim = modulation_dim // 2

        self.channel_resize_modules = nn.ModuleList()
        for channel in self.in_channels:
            self.channel_resize_modules.append(
                nn.Sequential(
                    nn.Conv2d(channel, self.modulation_dim, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.modulation_dim, self.modulation_dim, 3, padding=1)
                )
            )
        if self.decouple_head:
            self._init_decopule_layers()
        else:
            self._init_layers()

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        self.cls_pred_head = ModulatedSirens(
            num_inner_layers=self.num_inner_layers,
            in_dim=self.decoder_in_dim,
            modulation_dim=self.modulation_dim,
            out_dim=self.num_classes,
            base_channels=self.modulation_base_dim,
            is_residual=True
        )
        self.reg_pred_head = ModulatedSirens(
            num_inner_layers=self.num_inner_layers,
            in_dim=self.decoder_in_dim,
            modulation_dim=self.modulation_dim,
            out_dim=4 * self.reg_max,
            base_channels=self.modulation_base_dim,
            is_residual=True
        )

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def _init_decopule_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # reg_out_channels = max(
        #     (16, self.in_channels[0] // 4, self.reg_max * 4))
        # cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                ModulatedSirens(
                    num_inner_layers=self.num_inner_layers,
                    in_dim=self.decoder_in_dim,
                    modulation_dim=self.modulation_dim,
                    out_dim=4 * self.reg_max,
                    base_channels=self.modulation_base_dim,
                    is_residual=True
                )
            )
            self.cls_preds.append(
                ModulatedSirens(
                    num_inner_layers=self.num_inner_layers,
                    in_dim=self.decoder_in_dim,
                    modulation_dim=self.modulation_dim,
                    out_dim=self.num_classes,
                    base_channels=self.modulation_base_dim,
                    is_residual=True
                )
            )

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def _meshgrid(self,
                  x: Tensor,
                  y: Tensor,
                  row_major: bool = True) -> Tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def get_coordinate_map(self, lr_size, hr_size, device='cpu'):
        def to_coordinates(size=(56, 56), device='cpu', return_map=True, offset=0.5):
            shift_x = torch.arange(0, size[0], device=device) + offset
            shift_y = torch.arange(0, size[1], device=device) + offset
            shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)

            coordinates = torch.stack([shift_xx, shift_yy], dim=-1)

            # Normalize coordinates to lie in [-.5, .5]
            coordinates[..., 0] = coordinates[..., 0] / (size[0] - 1) - 0.5
            coordinates[..., 1] = coordinates[..., 1] / (size[1] - 1) - 0.5
            # Convert to range [-1, 1]
            coordinates *= 2
            if return_map:
                coordinates = rearrange(coordinates, '(H W) C -> H W C', H=size[1])
            # [x, y]
            return coordinates

        h_lr, w_lr = lr_size
        h_hr, w_hr = hr_size
        lr_coord = to_coordinates(lr_size, device=device).permute(2, 0, 1).unsqueeze(0)
        hr_coord = to_coordinates(hr_size, device=device).permute(2, 0, 1).unsqueeze(0)
        # important! mode='nearest' gives inconsistent results
        rel_grid = hr_coord - F.interpolate(lr_coord, size=hr_size, mode='nearest-exact')
        rel_grid[:, 0, :, :] *= h_lr
        rel_grid[:, 1, :, :] *= w_lr

        return rel_grid.contiguous().detach(), lr_coord.contiguous().detach(), hr_coord.contiguous().detach()

    def forward(self, x):
        assert len(x) == self.num_levels
        if self.decouple_head:
            return multi_apply(self.forward_single, x, self.channel_resize_modules, self.cls_preds, self.reg_preds)
        return multi_apply(self.forward_single, x, self.channel_resize_modules)

    def forward_single(self, x: torch.Tensor, channel_resize_module, cls_pred_head=None, reg_pred_head=None) -> Tuple:
        b, _, h, w = x.shape
        x = channel_resize_module(x)
        func_map = F.interpolate(x, size=self.target_size, mode=self.interp_func, align_corners=True)
        rel_grid, lr_grid, hr_grid = self.get_coordinate_map(
            lr_size=(w, h),
            hr_size=self.target_size,
            device=x.device)

        local_input = rel_grid
        if self.encode_scale_ratio:
            w_ratio = w / self.target_size[0]
            h_ratio = h / self.target_size[1]
            scale_ratio_map = torch.tensor([h_ratio, w_ratio]).view(1, -1, 1, 1).expand(1, -1, *self.target_size).to(
                x.device)
            local_input = torch.cat([local_input, scale_ratio_map], dim=1)
        if self.encode_hr_coord:
            local_input = torch.cat([local_input, hr_grid], dim=1)

        h, w = self.target_size
        local_input = repeat(local_input, 'b c h w -> (b bs) c h w', bs=b)
        if self.decouple_head:
            cls_logit = cls_pred_head(local_input, func_map)
            bbox_dist_preds = reg_pred_head(local_input, func_map)
        else:
            cls_logit = self.cls_pred_head(local_input, func_map)
            bbox_dist_preds = self.reg_pred_head(local_input, func_map)

        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds

@MODELS.register_module()
class YOLOv8SIRENSHead(YOLOv8Head):
    """YOLOv8Head head used in `YOLOv8`.

    Args:
        head_module(:obj:`ConfigDict` or dict): Base module used for YOLOv8Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dfl (:obj:`ConfigDict` or dict): Config of Distribution Focal
            Loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.loss_dfl = MODELS.build(loss_dfl)
        # YOLOv8 doesn't need loss_obj
        self.loss_obj = None

