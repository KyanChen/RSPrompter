# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

from mmpretrain.engine import EMAHook

with read_base():
    from .._base_.datasets.imagenet21k_bs128 import *
    from .._base_.default_runtime import *
    from .._base_.models.convnext_base import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

# model setting
model.update(head=dict(num_classes=21841))

# dataset setting
data_preprocessor.update(num_classes=21841)
train_dataloader.update(batch_size=64)

# schedule setting
optim_wrapper.update(
    optimizer=dict(lr=4e-3),
    clip_grad=dict(max_norm=5.0),
)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr.update(base_batch_size=4096)
