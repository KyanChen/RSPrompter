# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.datasets.coco_panoptic import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

from mmcv.ops import nms
from torch.nn import GroupNorm

from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.detectors.panoptic_fpn import PanopticFPN
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.seg_heads.panoptic_fpn_head import PanopticFPNHead
from mmdet.models.seg_heads.panoptic_fusion_heads import HeuristicFusionHead

model.update(
    dict(
        type=PanopticFPN,
        data_preprocessor=dict(
            type=DetDataPreprocessor,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
            pad_mask=True,
            mask_pad_value=0,
            pad_seg=True,
            seg_pad_value=255),
        semantic_head=dict(
            type=PanopticFPNHead,
            num_things_classes=80,
            num_stuff_classes=53,
            in_channels=256,
            inner_channels=128,
            start_level=0,
            end_level=4,
            norm_cfg=dict(type=GroupNorm, num_groups=32, requires_grad=True),
            conv_cfg=None,
            loss_seg=dict(
                type=CrossEntropyLoss, ignore_index=255, loss_weight=0.5)),
        panoptic_fusion_head=dict(
            type=HeuristicFusionHead,
            num_things_classes=80,
            num_stuff_classes=53),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.6,
                nms=dict(type=nms, iou_threshold=0.5, class_agnostic=True),
                max_per_img=100,
                mask_thr_binary=0.5),
            # used in HeuristicFusionHead
            panoptic=dict(mask_overlap=0.5, stuff_area_limit=4096))))

# Forced to remove NumClassCheckHook
custom_hooks = []
