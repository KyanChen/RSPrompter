from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpl.registry import MODELS
from mmengine.model import BaseModel
from einops import rearrange, repeat
from mmpl.datasets.data_utils import lafan1_utils_torch

@MODELS.register_module()
class MotionGPTHead(BaseModel):
    def __init__(
            self,
            in_channels=768,
            out_channels=dict(
                rot_6d=22 * 6 * 2,
                root_pos=3 * 2,
                foot_contact=2 * 2,
            ),
            loss='certainty_loss',
            return_certainty=False,
            uncertainty_beta=10,
            num_layers: int = 2,
            losses=dict(),
            init_cfg: Optional[dict] = None):
        super(MotionGPTHead, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        for k, v in out_channels.items():
            layers = []
            for i in range(self.num_layers - 1):
                layers.append(
                    nn.Sequential(
                        nn.Linear(self.in_channels, self.in_channels),
                        nn.LeakyReLU()
                    )
                )
            layers.append(nn.Linear(self.in_channels, v))
            self.register_module(f'{k}', nn.Sequential(*layers))

        for k, v in losses.items():
            self.register_module(f'{k}', MODELS.build(v))

        self.get_loss = getattr(self, f'get_{loss}')
        self.certainty = False
        if loss == 'certainty_loss':
            self.certainty = True
        self.return_certainty = return_certainty
        self.beta = uncertainty_beta

    def forward(self, x):
        res_dict = {}
        for k, v in self.out_channels.items():
            res_dict[k] = getattr(self, k)(x)
        return res_dict

    def loss(self, *args, **kwargs):
        return self.get_loss(*args, **kwargs)

    def get_uncertainty_loss(
            self,
            x,
            norm_type,
            normalization_info,
            block_size,
            parents,
            rotations_shift,
            positions_shift,
            foot_contact,
            arranged_x_dict,
            *args,
            **kwargs) -> dict:

        pred_dict = self.forward(x)
        pred_rot_6d = rearrange(pred_dict['rot_6d'], 'b t (d c) -> b t d c', d=2)
        pred_rot_6d = rearrange(pred_rot_6d, 'b t d (n_j c) -> b t d n_j c', c=6)

        pred_root_pos = rearrange(pred_dict['root_pos'], 'b t (d c) -> b t d c', d=2)

        if norm_type == 'minmax':
            pred_rot_6d = (pred_rot_6d + 1) / 2
            pred_rot_6d = pred_rot_6d * (normalization_info['max_rot_6d_with_position'][:, :6] - normalization_info['min_rot_6d_with_position'][:, :6]) + \
                          normalization_info['min_rot_6d_with_position'][:, :6]
        elif norm_type == 'meanstd':
            pred_rot_6d = pred_rot_6d * normalization_info['rot_6d']['std'][:, :6] + \
                            normalization_info['rot_6d']['mean'][:, :6]

        pred_root_pos = pred_root_pos.unsqueeze(-2)
        if norm_type == 'minmax':
            pred_root_pos = (pred_root_pos + 1) / 2
            pred_root_pos = pred_root_pos * (normalization_info['max_diff_root_xz'] - normalization_info['min_diff_root_xz']) + \
                                 normalization_info['min_diff_root_xz']
        elif norm_type == 'meanstd':
            pred_root_pos = pred_root_pos * normalization_info['root_pos']['std'] + \
                                    normalization_info['root_pos']['mean']

        # local rotation loss
        gt_rotation_6d = arranged_x_dict['rot_6d'][:, -block_size:, :, :6].detach()  # B, T, N, 6
        rotation_loss = self.rot_6d_loss(pred_rot_6d, gt_rotation_6d)

        # root position loss
        root_position_loss = self.root_pos_loss(pred_root_pos, arranged_x_dict['root_pos'][:, -block_size:, 0:1, :].detach())

        # global position loss
        # 从预测值恢复全局坐标，注意预测值是基于前一帧的相对坐标
        # 先恢复9D旋转量，基于均值
        pred_rotations_9d = lafan1_utils_torch.matrix6D_to_9D_torch(pred_rot_6d[..., 0, :, :])
        # 然后恢复root节点的zyx坐标，预测的是与上一帧的偏移
        position_new = positions_shift[:, -block_size:].clone()
        position_new[..., 0, :] = pred_root_pos[:, :, 0, 0, :]

        grot_new, gpos_new = lafan1_utils_torch.fk_torch(pred_rotations_9d, position_new, parents)

        gt_global_rotations, gt_global_positions = lafan1_utils_torch.fk_torch(rotations_shift, positions_shift, parents)
        global_position_loss = self.global_pos_loss(gpos_new, gt_global_positions[:, -block_size:].detach())

        # smoothness loss
        smoothness_loss = self.smoothness_loss(gpos_new, gt_global_positions[:, :block_size].detach())

        # foot contact loss
        pred_foot_contact = torch.sigmoid(pred_dict['foot_contact'])
        foot_contact_loss = self.foot_contact_loss(pred_foot_contact, foot_contact[:, -block_size:].detach())

        # foot velocity loss
        foot_vel = lafan1_utils_torch.extract_foot_vel(gpos_new, foot_joint_idx=(3, 4, 7, 8))
        # sever gradient back propagation on c_out,
        # otherwise c_out will tend to be zero
        pred_foot_contact = repeat(pred_foot_contact, 'b t c -> b t c o', o=3)
        foot_velocity_loss = self.foot_velocity_loss(foot_vel, pred_foot_contact.detach())

        losses = dict(
            rotation_loss=rotation_loss,
            global_position_loss=global_position_loss,
            root_position_loss=root_position_loss,
            smoothness_loss=smoothness_loss,
            foot_contact_loss=foot_contact_loss,
            foot_velocity_loss=foot_velocity_loss,
        )
        return losses

    def get_certainty_loss(
            self,
            x,
            # normalization_info=dict(
            # max_rot_6d_with_position=0,
            # min_rot_6d_with_position=0,
            # max_diff_root_xz=0,
            # min_diff_root_xz=0,
            # ),
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
            block_size=64,
            parents=[],
            rot_6d_with_position=None,
            diff_root_zyx=None,
            positions_shift=None,
            rotations_shift=None,
            *args,
            **kwargs) -> dict:
        """Compute loss.
        """

        pred_rot_6d, pred_root_pos = self.forward(x)
        pred_rot_6d = rearrange(pred_rot_6d, 'b t (n_j c) -> b t n_j c', c=6)

        if norm_type == 'minmax':
            pred_rot_6d = (pred_rot_6d + 1) / 2
            pred_rot_6d = pred_rot_6d * (normalization_info['max_rot_6d_with_position'][:, :6] - normalization_info['min_rot_6d_with_position'][:, :6]) + \
                          normalization_info['min_rot_6d_with_position'][:, :6]
        elif norm_type == 'meanstd':
            pred_rot_6d = pred_rot_6d * normalization_info['std_rot_6d'][:, :6] + \
                            normalization_info['mean_rot_6d'][:, :6]

        pred_root_pos = rearrange(pred_root_pos, 'b t (n_j c) -> b t n_j c', n_j=1)
        if norm_type == 'minmax':
            pred_root_pos = (pred_root_pos + 1) / 2
            pred_root_pos = pred_root_pos * (normalization_info['max_diff_root_xz'] - normalization_info['min_diff_root_xz']) + \
                                 normalization_info['min_diff_root_xz']
        elif norm_type == 'meanstd':
            pred_root_pos = pred_root_pos * normalization_info['std_root_pos'] + \
                                    normalization_info['mean_root_pos']

        # local rotation loss
        gt_rotation_6d = rot_6d_with_position[:, -block_size:, :, :6].detach()  # B, T, N_j, 6
        rotation_loss = self.rotation_loss(pred_rot_6d, gt_rotation_6d)

        # root position loss
        root_position_loss = self.root_position_loss(pred_root_pos, diff_root_zyx[:, -block_size:, 0:1, :].detach())

        # global position loss
        # 从预测值恢复全局坐标，注意预测值是基于前一帧的相对坐标
        # 先恢复9D旋转量，基于均值
        pred_rotations_9d = lafan1_utils_torch.matrix6D_to_9D_torch(pred_rot_6d[..., :, :])
        # 然后恢复root节点的zyx坐标，预测的是与上一帧的偏移
        position_new = positions_shift[:, :block_size].clone()
        position_new[..., 0, :] += pred_root_pos[:, :, 0, :]

        grot_new, gpos_new = lafan1_utils_torch.fk_torch(pred_rotations_9d, position_new, parents)

        gt_global_rotations, gt_global_positions = lafan1_utils_torch.fk_torch(rotations_shift, positions_shift, parents)
        global_position_loss = self.global_position_loss(gpos_new, gt_global_positions[:, -block_size:].detach())

        losses = dict(
            rotation_loss=rotation_loss,
            global_position_loss=global_position_loss,
            root_position_loss=root_position_loss
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
