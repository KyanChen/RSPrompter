custom_imports = dict(
    imports=['mmseg.datasets', 'mmseg.models', 'mmdet.models'],
    allow_failed_imports=False)

sub_model_train = [
    'panoptic_head',
    'data_preprocessor'
]

sub_model_optim = {
    'panoptic_head': {'lr_mult': 1},
}


max_epochs = 1200
optimizer = dict(type='AdamW', lr=0.0005, weight_decay=0.0001)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0005,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=120, by_epoch=True, begin=1, end=120)
]

param_scheduler_callback = dict(type='ParamSchedulerHook')
evaluator_ = dict(type='MeanAveragePrecision', iou_type='segm')
evaluator = dict(
    val_evaluator=dict(type='MeanAveragePrecision', iou_type='segm'))

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
prompt_shape = (60, 4)


model_cfg = dict(
    type='SegSAMAnchorPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    need_train_names=sub_model_train,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='vit_h',
        checkpoint='pretrain/sam/sam_vit_h_4b8939.pth',
        # type='vit_b',
        # checkpoint='pretrain/sam/sam_vit_b_01ec64.pth',
    ),
    panoptic_head=dict(
        type='SAMAnchorInstanceHead',
        neck=dict(
            type='SAMAggregatorNeck',
            in_channels=[1280] * 32,
            # in_channels=[768] * 12,
            inner_channels=32,
            selected_channels=range(4, 32, 2),
            # selected_channels=range(4, 12, 2),
            out_channels=256,
            up_sample_scale=4,
        ),
        rpn_head=dict(
            type='mmdet.RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='mmdet.AnchorGenerator',
                scales=[2, 4, 8, 16, 32, 64],
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32]),
            bbox_coder=dict(
                type='mmdet.DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='mmdet.SmoothL1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='SAMAnchorPromptRoIHead',
            bbox_roi_extractor=dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32]),
            bbox_head=dict(
                type='mmdet.Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='mmdet.DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='mmdet.SmoothL1Loss', loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32]),
            mask_head=dict(
                type='SAMPromptMaskHead',
                per_query_point=prompt_shape[1],
                with_sincos=True,
                class_agnostic=True,
                loss_mask=dict(
                    type='mmdet.CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
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
                    num=512,
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
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=1024,
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
                max_per_img=100,
                mask_thr_binary=0.5)
        )
    )
)


task_name = 'nwpu_ins'
exp_name = 'rsprompter_anchor_E20230601_0'
callbacks = [
    dict(
        type='DetVisualizationHook',
        draw=True,
        interval=1,
        score_thr=0.1,
        show=False,
        wait_time=1.,
        test_out_dir='visualization',
    )
]


vis_backends = [dict(type='mmdet.LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    fig_save_cfg=dict(
        frameon=False,
        figsize=(40, 20),
        # dpi=300,
    ),
    line_width=2,
    alpha=0.8
)


trainer_cfg = dict(
    compiled_model=False,
    accelerator='auto',
    strategy='auto',
    devices=[0],
    default_root_dir=f'results/{task_name}/{exp_name}',
    max_epochs=120,
    logger=False,
    callbacks=callbacks,
    log_every_n_steps=20,
    check_val_every_n_epoch=10,
    benchmark=True,
    use_distributed_sampler=True)

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

train_batch_size_per_gpu = 8
train_num_workers = 4
test_batch_size_per_gpu = 2
test_num_workers = 0
persistent_workers = False

data_parent = '/mnt/search01/dataset/cky_data/NWPU10'
train_data_prefix = ''
val_data_prefix = ''

predict_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=image_size),
    # If you don't have a gt annotation, delete the pipeline
    # dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'NWPUInsSegDataset'
predict_loader = dict(
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
            pipeline=predict_pipeline,
            backend_args=backend_args))

datamodule_cfg = dict(
    type='PLDataModule',
    predict_loader=predict_loader,
)
