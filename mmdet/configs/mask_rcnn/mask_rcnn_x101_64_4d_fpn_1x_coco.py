# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .mask_rcnn_x101_32x4d_fpn_1x_coco import *

model = dict(
    backbone=dict(
        type=ResNeXt,
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type=PretrainedInit, checkpoint='open-mmlab://resnext101_64x4d')))
