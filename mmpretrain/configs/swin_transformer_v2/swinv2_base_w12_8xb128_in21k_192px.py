# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base
from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import CutMix, Mixup

with read_base():
    from .._base_.datasets.imagenet21k_bs128 import *
    from .._base_.default_runtime import *
    from .._base_.models.swin_transformer_v2_base import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

# model settings
model.update(
    backbone=dict(
        img_size=192, drop_path_rate=0.5, window_size=[12, 12, 12, 6]),
    head=dict(num_classes=21841),
    init_cfg=[
        dict(type=TruncNormalInit, layer='Linear', std=0.02, bias=0.),
        dict(type=ConstantInit, layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(
        augments=[dict(type=Mixup, alpha=0.8),
                  dict(type=CutMix, alpha=1.0)]))

# dataset settings
data_preprocessor = dict(num_classes=21841)

_base_['train_pipeline'][1]['scale'] = 192  # RandomResizedCrop
_base_['test_pipeline'][1]['scale'] = 219  # ResizeEdge
_base_['test_pipeline'][2]['crop_size'] = 192  # CenterCrop
