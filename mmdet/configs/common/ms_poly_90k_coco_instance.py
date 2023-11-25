# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *

from mmcv.transforms import RandomChoiceResize
from mmengine.dataset import RepeatDataset
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop
from torch.optim import SGD

from mmdet.datasets import AspectRatioBatchSampler, CocoDataset
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import (FilterAnnotations,
                                               LoadAnnotations,
                                               LoadImageFromFile)
from mmdet.datasets.transforms.transforms import (CachedMixUp, CachedMosaic,
                                                  Pad, RandomCrop, RandomFlip,
                                                  RandomResize, Resize)
from mmdet.evaluation import CocoMetric

# dataset settings
dataset_type = CocoDataset
data_root = 'data/coco/'
# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

# Align with Detectron2
backend = 'pillow'
train_pipeline = [
    dict(
        type=LoadImageFromFile,
        backend_args=backend_args,
        imdecode_backend=backend),
    dict(
        type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type=RandomChoiceResize,
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True,
        backend=backend),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
test_pipeline = [
    dict(
        type=LoadImageFromFile,
        backend_args=backend_args,
        imdecode_backend=backend),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True, backend=backend),
    dict(
        type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader.update(
    dict(
        batch_size=2,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        sampler=dict(type=InfiniteSampler, shuffle=True),
        batch_sampler=dict(type=AspectRatioBatchSampler),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)))
val_dataloader.update(
    dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        pin_memory=True,
        sampler=dict(type=DefaultSampler, shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/instances_val2017.json',
            data_prefix=dict(img='val2017/'),
            test_mode=True,
            pipeline=test_pipeline,
            backend_args=backend_args)))
test_dataloader = val_dataloader

val_evaluator.update(
    dict(
        type=CocoMetric,
        ann_file=data_root + 'annotations/instances_val2017.json',
        metric=['bbox', 'segm'],
        format_only=False,
        backend_args=backend_args))
test_evaluator = val_evaluator

# training schedule for 90k
max_iter = 90000
train_cfg.update(
    dict(type=IterBasedTrainLoop, max_iters=max_iter, val_interval=10000))
val_cfg.update(dict(type=ValLoop))
test_cfg.update(dict(type=TestLoop))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]

# optimizer
optim_wrapper.update(
    dict(
        type=OptimWrapper,
        optimizer=dict(type=SGD, lr=0.02, momentum=0.9, weight_decay=0.0001)))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr.update(dict(enable=False, base_batch_size=16))

default_hooks.update(dict(checkpoint=dict(by_epoch=False, interval=10000)))
log_processor.update(dict(by_epoch=False))
