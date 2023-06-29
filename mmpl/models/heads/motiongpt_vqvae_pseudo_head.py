from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpl.registry import MODELS
from mmengine.model import BaseModel
from einops import rearrange, repeat
from mmpl.datasets.data_utils import lafan1_utils_torch


@MODELS.register_module()
class MotionVQVAEPseudoHead(BaseModel):
    def __init__(
            self,
            nb_joints=21,  # 22
            commit_loss_weight=0.01,
            losses=dict(),
            init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        self.nb_joints = nb_joints
        self.commit_loss_weight = commit_loss_weight

        self.loss_names = [k for k in losses.keys()]
        for k, v in losses.items():
            self.register_module(f'{k}', MODELS.build(v))
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.motion_dim_end = (nb_joints - 1) * 12 + 4 + 3 + 4

    def forward(self, x, *args, **kwargs):
        pass

    def loss(
            self,
            pred_motion,
            loss_commit,
            perplexity,
            gt_motion,
            *args,
            **kwargs
    ):
        """Compute loss.
        """
        loss_motion = self.motion_loss(pred_motion[..., :self.motion_dim_end], gt_motion[..., :self.motion_dim_end])

        vec_dim_start = 4
        vec_dim_end = (self.nb_joints - 1) * 3 + 4
        loss_motion_vec = self.motion_vec_loss(
            pred_motion[..., vec_dim_start:vec_dim_end], gt_motion[..., vec_dim_start:vec_dim_end])

        loss_commit = loss_commit * self.commit_loss_weight

        losses = dict(
            loss_motion=loss_motion,
            loss_motion_vec=loss_motion_vec,
            loss_commit=loss_commit,
            perplexity=perplexity,
        )
        return losses

    def predict(
            self,
            x,
            normalization_info=dict(
                mean_rot_6d=0,
                std_rot_6d=0,
                mean_root_pos=0,
                std_root_pos=0,
                max_rot_6d_with_position=0,
                min_rot_6d_with_position=0,
                max_diff_root_xz=0,
                min_diff_root_xz=0,
            ),
            norm_type='minmax',
            *args,
            **kwargs):
        """Compute loss.
        """

        pred_rot_6d, pred_root_pos = self.forward(x)
        if self.certainty:
            pred_rot_6d = rearrange(pred_rot_6d, 'b t (n_j c) -> b t n_j c', c=6)
        else:
            pred_rot_6d = rearrange(pred_rot_6d, 'b t (d n_j c) -> b t d n_j c', d=2, c=6)

        if norm_type == 'minmax':
            pred_rot_6d = (pred_rot_6d + 1) / 2
            pred_rot_6d = pred_rot_6d * (normalization_info['max_rot_6d_with_position'][:, :6] - normalization_info['min_rot_6d_with_position'][:, :6]) + \
                          normalization_info['min_rot_6d_with_position'][:, :6]
        elif norm_type == 'meanstd':
            pred_rot_6d = pred_rot_6d * normalization_info['std_rot_6d'][:, :6] + \
                          normalization_info['mean_rot_6d'][:, :6]

        if self.certainty:
            pred_root_pos = rearrange(pred_root_pos, 'b t (n_j c) -> b t n_j c', n_j=1)
        else:
            pred_root_pos = rearrange(pred_root_pos, 'b t (d n_j c) -> b t d n_j c', d=2, n_j=1)

        if norm_type == 'minmax':
            pred_root_pos = (pred_root_pos + 1) / 2
            pred_root_pos = pred_root_pos * (normalization_info['max_diff_root_xz'] - normalization_info['min_diff_root_xz']) + \
                                 normalization_info['min_diff_root_xz']
        elif norm_type == 'meanstd':
            pred_root_pos = pred_root_pos * normalization_info['std_root_pos'] + \
                                    normalization_info['mean_root_pos']

        if self.return_certainty:
            if not self.certainty:
                pred_rot_6d = pred_rot_6d[:, :, 0]
                pred_root_pos = pred_root_pos[:, :, 0]
        else:
            pred_rot_6d = torch.normal(mean=pred_rot_6d[:, :, 0], std=torch.abs(pred_rot_6d[:, :, 1]/self.beta))
            pred_root_pos = torch.normal(mean=pred_root_pos[:, :, 0], std=torch.abs(pred_root_pos[:, :, 1]/self.beta))

        return pred_rot_6d, pred_root_pos
