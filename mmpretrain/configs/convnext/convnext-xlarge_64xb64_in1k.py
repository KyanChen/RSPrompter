# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

from mmpretrain.engine import EMAHook

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_224 import *
    from .._base_.default_runtime import *
    from .._base_.models.convnext_base import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

# dataset setting
train_dataloader.update(batch_size=64)

# schedule setting
optim_wrapper.update(
    optimizer=dict(lr=4e-3),
    clip_grad=None,
)

# runtime setting
custom_hooks = [dict(type=EMAHook, momentum=1e-4, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (64 GPUs) x (64 samples per GPU)
auto_scale_lr.update(base_batch_size=4096)
