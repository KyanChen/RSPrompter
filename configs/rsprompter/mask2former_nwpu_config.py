custom_imports = dict(imports=['mmseg.datasets', 'mmseg.models', 'mmdet.models'], allow_failed_imports=False)
max_epochs = 2000

optimizer = dict(
    type='AdamW',
    lr=0.0002,
    weight_decay=1e-4
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
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
    )
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
    val_evaluator=evaluator_,
    test_evaluator=evaluator_
)


image_size = (1024, 1024)
data_preprocessor = dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        mask_pad_value=0,
        pad_size_divisor=32
)

num_things_classes = 10
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
num_queries = 60

# model settings
model = dict(
    type='mmdet.Mask2Former',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    panoptic_head=dict(
        type='mmdet.Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=num_queries,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                # num_layers=6,
                num_layers=2,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            # num_layers=9,
            num_layers=3,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='mmdet.MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(type='mmdet.ClassificationCost', weight=2.0),
                dict(
                    type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='mmdet.MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=False,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=None)


model_cfg = dict(
    type='MMDetPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    whole_model=model,
)

task_name = 'nwpu_ins'
exp_name = 'E20230604_4'
logger = dict(
    type='WandbLogger',
    project=task_name,
    group='mask2former',
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
    strategy="auto",
    # strategy="ddp",
    # strategy='ddp_find_unused_parameters_true',
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


train_batch_size_per_gpu = 8
train_num_workers = 4
test_batch_size_per_gpu = 8
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
    test_loader=val_loader,
    predict_loader=val_loader
)