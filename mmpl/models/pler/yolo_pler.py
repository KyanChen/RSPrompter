import os
from typing import Any

import mmengine
import torch
import torch.nn as nn
from einops import rearrange

from mmdet.models.utils import samplelist_boxtype2tensor
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from mmpl.registry import MODELS
from ..builder import build_backbone, build_loss, build_neck, build_head
from .base_pler import BasePLer
from mmpl.structures import ClsDataSample
from .base import BaseClassifier
import lightning.pytorch as pl
import torch.nn.functional as F
from mmpl.datasets.data_utils import lafan1_utils_torch


@MODELS.register_module()
class YoloPLer(BasePLer):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = MODELS.build(head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def setup(self, stage: str) -> None:
        pass

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples

    def validation_step(self, batch, batch_idx):
        data = self.data_preprocessor(batch, False)

        img = data['inputs']
        batch_data_samples = data['data_samples']

        x = self.backbone(img)  # [torch.Size([2, 128, 64, 64]), torch.Size([2, 256, 32, 32]), torch.Size([2, 512, 16, 16])]
        if hasattr(self, 'neck'):
            x = self.neck(x)  # [torch.Size([2, 128, 64, 64]), torch.Size([2, 256, 32, 32]), torch.Size([2, 512, 16, 16])]
        results_list = self.head.predict(x, batch_data_samples, rescale=True)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        preds = []
        targets = []
        for data_sample in batch_data_samples:
            result = dict()
            pred = data_sample.pred_instances
            result['boxes'] = pred['bboxes']
            result['scores'] = pred['scores']
            result['labels'] = pred['labels']
            preds.append(result)
            # parse gt
            gt = dict()
            gt['boxes'] = data_sample.gt_instances['bboxes']
            gt['labels'] = data_sample.gt_instances['labels']
            targets.append(gt)
        self.val_evaluator.update(preds, targets)

    def training_step(self, batch, batch_idx):
        data = self.data_preprocessor(batch, True)
        img = data['inputs']
        batch_data_samples = data['data_samples']
        x = self.backbone(
            img)  # [torch.Size([2, 128, 64, 64]), torch.Size([2, 256, 32, 32]), torch.Size([2, 512, 16, 16])]
        if hasattr(self, 'neck'):
            x = self.neck(
                x)  # [torch.Size([2, 128, 64, 64]), torch.Size([2, 256, 32, 32]), torch.Size([2, 512, 16, 16])]
        losses = self.head.loss(x, batch_data_samples)

        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        positions = batch['positions']
        rotations = batch['rotations']
        global_positions = batch['global_positions']
        global_rotations = batch['global_rotations']
        foot_contact = batch['foot_contact']
        parents = batch['parents'][0]
        positions_shift, rotations_shift = lafan1_utils_torch.reduce_frame_root_shift_and_rotation(
            positions, rotations, base_frame_id=0)

        assert positions_shift.shape[1] <= self.max_frames_predict

        while positions_shift.shape[1] < self.max_frames_predict:
            rot_6d_with_position, diff_root_zyx = lafan1_utils_torch.get_shift_model_input(positions_shift.clone(), rotations_shift.clone())
            # rot_6d_with_position BxTxJx9
            # diff_root_zyx BxTxJx3

            rot_6d_with_position_input = rot_6d_with_position[:, -self.block_size:].clone()
            diff_root_zyx_input = diff_root_zyx[:, -self.block_size:].clone()


            x = self.forward(rot_6d_with_position_input, diff_root_zyx_input)
            x = x[:, -1:, :]
            pred_rot_6d, pred_diff_root_zyx = self.head.forward(x)

            pred_rot_6d = rearrange(pred_rot_6d, 'b t (d c) -> b t d c', d=2)
            pred_rot_6d = rearrange(pred_rot_6d, 'b t d (n_j c) -> b t d n_j c', c=6)
            pred_rot_6d = (pred_rot_6d + 1) / 2
            pred_rot_6d = pred_rot_6d * (self.max_rot_6d_with_position[:, :6] - \
                                         self.min_rot_6d_with_position[:, :6]) + \
                          self.min_rot_6d_with_position[:, :6]
            # try:
            #     pred_rot_6d = torch.normal(mean=pred_rot_6d[:, :, 0], std=torch.abs(pred_rot_6d[:, :, 1]))
            # except:
            #     import ipdb;
            #     ipdb.set_trace()
            pred_rot_6d = pred_rot_6d[:, :, 0]

            pred_diff_root_zyx = rearrange(pred_diff_root_zyx, 'b t (d c) -> b t d c', d=2)
            pred_diff_root_zyx = pred_diff_root_zyx.unsqueeze(-2)
            pred_diff_root_zyx = (pred_diff_root_zyx + 1) / 2
            pred_diff_root_zyx = pred_diff_root_zyx * (self.max_diff_root_xz - \
                                                       self.min_diff_root_xz) + \
                                 self.min_diff_root_xz

            # pred_diff_root_zyx = torch.normal(mean=pred_diff_root_zyx[:, :, 0], std=torch.abs(pred_diff_root_zyx[:, :, 1]))
            pred_diff_root_zyx = pred_diff_root_zyx[:, :, 0]

            # project 6D rotation to 9D rotation
            pred_rotations_9d = lafan1_utils_torch.matrix6D_to_9D_torch(pred_rot_6d)
            # accumulate root position shift to the last frame
            position_new = positions_shift[:, -1:].clone()
            position_new[..., 0, :] += pred_diff_root_zyx[..., 0, :]

            rotations_shift = torch.cat([rotations_shift, pred_rotations_9d], dim=1)
            positions_shift = torch.cat([positions_shift, position_new], dim=1)

        return positions_shift, rotations_shift, batch




