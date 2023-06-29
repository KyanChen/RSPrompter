from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpl.registry import MODELS
from mmengine.model import BaseModel
from einops import rearrange, repeat
from mmpl.datasets.data_utils import lafan1_utils_torch


@MODELS.register_module()
class MotionGPTPseudoHead(BaseModel):
    def __init__(
            self,
            losses=dict(),
            init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.loss_names = [k for k in losses.keys()]
        for k, v in losses.items():
            self.register_module(f'{k}', MODELS.build(v))

    def forward(self, x, *args, **kwargs):
        pass

    def loss(
            self,
            logits,
            labels,
            *args,
            **kwargs
    ):
        """Compute loss.
        """
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = self.cls_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        losses = dict(
            loss_ce=loss_ce,
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
