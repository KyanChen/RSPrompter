import os
from typing import Any

import mmengine
import torch
import torch.nn as nn
from einops import rearrange

from mmpl.registry import MODELS
from ..builder import build_backbone, build_loss, build_neck, build_head
from .base_pler import BasePLer
from mmpl.structures import ClsDataSample
from .base import BaseClassifier
import lightning.pytorch as pl
import torch.nn.functional as F
from mmpl.datasets.data_utils import lafan1_utils_torch


@MODELS.register_module()
class MotionGPTPLer(BasePLer):
    def __init__(self,
                 proj_nets,
                 spatial_transformer,
                 temporal_transformer,
                 head,
                 mean_std_file,
                 norm_type='meanstd',
                 block_size=512,
                 max_frames_predict=128,
                 n_prompt_tokens=20,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        for k, v in proj_nets.items():
            self.register_module(k, build_neck(v))
        self.spatial_transformer = build_neck(spatial_transformer)
        self.temporal_transformer = build_neck(temporal_transformer)
        self.head = build_head(head)

        self.block_size = block_size
        self.mean_std_file = mean_std_file
        self.max_frames_predict = max_frames_predict
        self.n_prompt_tokens = n_prompt_tokens
        self.norm_type = norm_type

    def setup(self, stage: str) -> None:
        self.mean_std_info = mmengine.load(self.mean_std_file)
        # for k, v in mean_std.items():
        #     for kk, vv in v.items():
        #         self.mean_std_info[k][kk] = vv.to(self.device)
    
    def on_fit_start(self) -> None:
        super().on_fit_start()
        for k, v in self.mean_std_info.items():
            for kk, vv in v.items():
                self.mean_std_info[k][kk] = vv.to(self.device)

    def training_val_step(self, batch, batch_idx, prefix=''):
        positions = batch['positions']
        rotations = batch['rotations']
        # global_positions = batch['global_positions']
        # global_rotations = batch['global_rotations']
        foot_contact = batch['foot_contact']
        parents = batch['parents'][0]

        positions_shift, rotations_shift = lafan1_utils_torch.reduce_frame_root_shift_and_rotation(
            positions, rotations, base_frame_id=0)
        # shift_global_rotations, shift_global_positions = lafan1_utils_torch.fk_torch(
        #     rotations_shift.clone(), positions_shift.clone(), parents)

        arranged_x_dict = lafan1_utils_torch.get_model_input(positions_shift.clone(), rotations_shift.clone())
        x_inputs = {k: v[:, :self.block_size].clone() for k, v in arranged_x_dict.items()}

        x = self.forward(inputs=x_inputs)

        losses = self.head.loss(
            x,
            norm_type=self.norm_type,
            normalization_info=self.mean_std_info,
            block_size=self.block_size,
            parents=parents,
            rotations_shift=rotations_shift,
            positions_shift=positions_shift,
            foot_contact=foot_contact,
            arranged_x_dict=arranged_x_dict,
        )

        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'{prefix}_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses

        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def validation_step(self, batch, batch_idx):
        return self.training_val_step(batch, batch_idx, prefix='val')

    def training_step(self, batch, batch_idx):
        return self.training_val_step(batch, batch_idx, prefix='train')

    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:

        if self.norm_type == 'minmax':
            # min-max normalization
            rot_6d_input = (rot_6d_input - self.min_rot_6d_with_position) / (
                    self.max_rot_6d_with_position - self.min_rot_6d_with_position)
            rot_6d_input = rot_6d_input * 2 - 1

            root_pos_input = (root_pos_input - self.min_diff_root_xz) / (
                    self.max_diff_root_xz - self.min_diff_root_xz)
            root_pos_input = root_pos_input * 2 - 1
        elif self.norm_type == 'meanstd':
            for k, v in inputs.items():
                inputs[k] = (v - self.mean_std_info[k]['mean']) / self.mean_std_info[k]['std']

        # arrange input
        x_inputs = []
        for k, v in inputs.items():
            if hasattr(self, f'{k}_proj'):
                x = getattr(self, f'{k}_proj')(v)
                x_inputs.append(x)
        x = torch.cat(x_inputs, dim=-2)
        bs = x.shape[0]
        x = rearrange(x, 'b t j d -> (b t) j d')
        x, _ = self.spatial_transformer(x)
        x = rearrange(x, '(b t) d -> b t d', b=bs)

        outputs = self.temporal_transformer(inputs_embeds=x)
        x = outputs['last_hidden_state']
        return x

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        positions = batch['positions']
        rotations = batch['rotations']
        global_positions = batch['global_positions']
        global_rotations = batch['global_rotations']
        foot_contact = batch['foot_contact']
        parents = batch['parents'][0]
        positions_shift, rotations_shift = lafan1_utils_torch.reduce_frame_root_shift_and_rotation(
            positions, rotations, base_frame_id=0)

        pred_positions_shift = positions_shift.clone()[:, :self.n_prompt_tokens]
        pred_rotations_shift = rotations_shift.clone()[:, :self.n_prompt_tokens]

        assert pred_positions_shift.shape[1] <= self.max_frames_predict

        while pred_positions_shift.shape[1] < self.max_frames_predict:
            rot_6d_with_position, diff_root_zyx = lafan1_utils_torch.get_shift_model_input(pred_positions_shift.clone(), pred_rotations_shift.clone())
            # rot_6d_with_position BxTxJx9
            # diff_root_zyx BxTxJx3

            rot_6d_input = rot_6d_with_position[:, -self.block_size:].clone()
            root_pos_input = diff_root_zyx[:, -self.block_size:].clone()

            x = self.forward(rot_6d_input, root_pos_input)
            x = x[:, -1:, :]

            pred_rot_6d, pred_diff_root_zyx = self.head.predict(
                x,
                normalization_info=dict(
                    mean_rot_6d=self.mean_rot_6d,
                    std_rot_6d=self.std_rot_6d,
                    mean_root_pos=self.mean_root_pos,
                    std_root_pos=self.std_root_pos,
                    max_rot_6d_with_position=self.max_rot_6d_with_position,
                    min_rot_6d_with_position=self.min_rot_6d_with_position,
                    max_diff_root_xz=self.max_diff_root_xz,
                    min_diff_root_xz=self.min_diff_root_xz,
                ),
                norm_type=self.norm_type,
            )

            # project 6D rotation to 9D rotation
            pred_rotations_9d = lafan1_utils_torch.matrix6D_to_9D_torch(pred_rot_6d)
            # accumulate root position shift to the last frame
            position_new = pred_positions_shift[:, -1:].clone()
            position_new[..., 0, :] += pred_diff_root_zyx[..., 0, :]

            pred_rotations_shift = torch.cat([pred_rotations_shift, pred_rotations_9d], dim=1)
            pred_positions_shift = torch.cat([pred_positions_shift, position_new], dim=1)

        return pred_positions_shift, pred_rotations_shift, positions_shift[:, :self.max_frames_predict], rotations_shift[:, :self.max_frames_predict]




