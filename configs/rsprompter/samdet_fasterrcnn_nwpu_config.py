custom_imports = dict(imports=['mmseg.datasets', 'mmseg.models'], allow_failed_imports=False)

sub_model_train = [
    'whole_model'
]

sub_model_optim = {
    'whole_model': {'lr_mult': 1},
}

max_epochs = 1000

optimizer = dict(
    type='AdamW',
    sub_model=sub_model_optim,
    lr=0.0005,
    weight_decay=1e-3
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=5e-4,
        by_epoch=True,
        begin=0,
        end=1,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        by_epoch=True,
        begin=1,
        end=max_epochs,
    ),
]

param_scheduler_callback = dict(
    type='ParamSchedulerHook'
)

evaluator_ = dict(
        type='CocoPLMetric',
        metric=['bbox', 'segm'],
        proposal_nums=[1, 10, 100]
)

evaluator = dict(
    # train_evaluator=evaluator_,
    val_evaluator=evaluator_,
)


image_size = (1024, 1024)

data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
)

num_things_classes = 10
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

model = dict(
    type='mmdet.FasterRCNN',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='mmdet.RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='mmdet.DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='mmdet.StandardRoIHead',
        bbox_roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='mmdet.Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='mmdet.DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

model_cfg = dict(
    type='SegSAMDetPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    need_train_names=sub_model_train,
    whole_model=model,
    backbone=dict(
        type='vit_h',
        checkpoint='pretrain/sam/sam_vit_h_4b8939.pth',
        # type='vit_b',
        # checkpoint='pretrain/sam/sam_vit_b_01ec64.pth',
    )
)

task_name = 'nwpu_ins'
exp_name = 'E20230531_9'
logger = dict(
    type='WandbLogger',
    project=task_name,
    group='samdet',
    name=exp_name
)
# logger = None

callbacks = [
    param_scheduler_callback,
    dict(
        type='ModelCheckpoint',
        dirpath=f'results/{task_name}/{exp_name}/checkpoints',
        save_last=True,
        mode='max',
        monitor='valsegm_map_0',
        save_top_k=2,
        filename='epoch_{epoch}-map_{valsegm_map_0:.4f}'
    ),
    dict(
        type='LearningRateMonitor',
        logging_interval='step'
    )
]


trainer_cfg = dict(
    compiled_model=False,
    accelerator="auto",
    # strategy="auto",
    # strategy="ddp",
    strategy='ddp_find_unused_parameters_true',
    # precision='32',
    # precision='16-mixed',
    devices=8,
    default_root_dir=f'results/{task_name}/{exp_name}',
    # default_root_dir='results/tmp',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=5,
    check_val_every_n_epoch=5,
    benchmark=True,
    # sync_batchnorm=True,
    # fast_dev_run=True,

    # limit_train_batches=1,
    # limit_val_batches=0,
    # limit_test_batches=None,
    # limit_predict_batches=None,
    # overfit_batches=0.0,

    # val_check_interval=None,
    # num_sanity_val_steps=0,
    # enable_checkpointing=None,
    # enable_progress_bar=None,
    # enable_model_summary=None,
    # accumulate_grad_batches=32,
    # gradient_clip_val=15,
    # gradient_clip_algorithm='norm',
    # deterministic=None,
    # inference_mode: bool=True,
    use_distributed_sampler=True,
    # profiler="simple",
    # detect_anomaly=False,
    # barebones=False,
    # plugins=None,
    # reload_dataloaders_every_n_epochs=0,
)


backend_args = None
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='mmdet.Resize', scale=image_size),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=image_size),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_batch_size_per_gpu = 4
train_num_workers = 4
test_batch_size_per_gpu = 4
test_num_workers = 4
persistent_workers = True

data_parent = '/mnt/search01/dataset/cky_data/NWPU10'
train_data_prefix = ''
val_data_prefix = ''

dataset_type = 'NWPUInsSegDataset'

val_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            ann_file='NWPU_instances_val.json',
            data_prefix=dict(img_path='positive image set'),
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=test_pipeline,
            backend_args=backend_args))

datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            ann_file='NWPU_instances_train.json',
            data_prefix=dict(img_path='positive image set'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)
    ),
    val_loader=val_loader,
    # test_loader=val_loader
    predict_loader=val_loader
)