import os
from typing import Any

import einops
import mmengine
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from lightning.pytorch.utilities import grad_norm
from mmengine.structures import InstanceData

from mmpl.registry import MODELS
from mmseg.utils import SampleList
from ..builder import build_backbone, build_loss, build_neck, build_head
from .base_pler import BasePLer
from mmpl.structures import ClsDataSample
from .base import BaseClassifier
import lightning.pytorch as pl
import torch.nn.functional as F


@MODELS.register_module()
class SegPLer(BasePLer):
    def __init__(self,
                 sam=None,
                 sam_checkpoint='',
                 points_per_side=None,
                 sam_prompt_generator=None,
                 only_img_encoder=False,
                 only_decoder=False,
                 global_prompt=None,
                 need_train_names=None,
                 head=None,
                 with_clip=False,
                 train_head=False,
                 threshold=0.5,
                 ignore_index=255,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.need_train_names = need_train_names
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.only_img_encoder = only_img_encoder
        self.only_decoder = only_decoder
        self.global_prompt = global_prompt
        self.train_head = train_head

        if sam is not None:
            if self.only_img_encoder:
                self.sam = sam_model_registry[sam](sam_checkpoint).image_encoder
            elif self.only_decoder:
                self.prompt_encoder = sam_model_registry[sam](sam_checkpoint).prompt_encoder
                self.mask_decoder = sam_model_registry[sam](sam_checkpoint).mask_decoder
            else:
                sam = sam_model_registry[sam](sam_checkpoint, train_head=train_head)
                self.img_encoder = sam.image_encoder
                self.prompt_encoder = sam.prompt_encoder
                self.mask_decoder = sam.mask_decoder
                self.prompt_encoder_no_mask_embed = sam.prompt_encoder.no_mask_embed

        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side, 0, 1)
        if sam_prompt_generator is not None:
            self.sam_prompt_generator = MODELS.build(sam_prompt_generator)
        if head is not None:
            self.head = MODELS.build(head)
        self.with_clip = with_clip
        if global_prompt is not None:
            if with_clip:
                self.logits_prompt = nn.Sequential(
                    nn.Linear(1, 8),
                    nn.ReLU(),
                    nn.Linear(8, 16)
                )
                self.global_prompt = nn.Sequential(
                    nn.Conv2d(768+16, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 1, kernel_size=3, padding=1),
                )
            else:
                self.global_prompt = nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 1, kernel_size=3, padding=1),
                )

    def setup(self, stage: str) -> None:
        if self.need_train_names is not None:
            self._set_grad(self.need_train_names, noneed_train_names=[])

    def configure_sharded_model(self) -> None:
        if self.trainer.strategy.__class__.__name__ == 'FSDPStrategy':
            from torch.distributed.fsdp.wrap import wrap
            self.sam_prompt_generator = wrap(self.sam_prompt_generator)
            self.img_encoder = wrap(self.img_encoder)
            self.prompt_encoder_no_mask_embed = wrap(self.prompt_encoder_no_mask_embed)
            self.mask_decoder = wrap(self.mask_decoder)
            self.prompt_encoder = wrap(self.prompt_encoder)
            from torch.distributed.fsdp import CPUOffload
            # from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            # import functools
            # strategy = dict(
            #     type='FSDPStrategy',
            #     cpu_offload=CPUOffload(offload_params=True),
            #     auto_wrap_policy=functools.partial(
            #         size_based_auto_wrap_policy, min_num_params=int(1e8)
            #     )
            #
            # )
        else:
            super().configure_sharded_model()

    def configure_optimizers(self):
        if self.trainer.strategy.__class__.__name__ == 'DeepSpeedStrategy':
            import deepspeed
            # optimizer = deepspeed.runtime.
            optimizer = deepspeed.ops.adam.FusedAdam(self.sam_prompt_generator.parameters(), lr=1e-4)
            # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.sam_prompt_generator.parameters(), lr=1e-4)
            # optimizer = torch.optim.Adam(self.sam_prompt_generator.parameters(), lr=1e-4)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            return [optimizer], [lr_scheduler]
        else:
            return super().configure_optimizers()

    def init_weights(self):
        import ipdb; ipdb.set_trace()
        pass

    # def on_fit_start(self) -> None:
    #     if hasattr(self, 'train_evaluator'):
    #         self.train_evaluator = self.train_evaluator.to(self.device)
    #     if hasattr(self, 'val_evaluator'):
    #         self.val_evaluator = self.val_evaluator.to(self.device)

    def train(self, mode=True):
        if self.need_train_names is not None:
            return self._set_train_module(mode, self.need_train_names)
        else:
            super().train(mode)
            return self

    def validation_step(self, batch, batch_idx):
        seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)
        if self.only_img_encoder:
            masks_pred = self.forward_only_img_encoder(batch)
            masks_pred = F.interpolate(masks_pred, size=seg_label.shape[-2:], mode='bilinear',
                                       align_corners=True)
            seg_logits = masks_pred > 0
        elif self.only_decoder:
            cls_logits, masks, n_iou_preds = self.forward_sam_prompt_generator(batch)  # 1x100x2, 1x100x1x256x256, 1x100x1
            masks = masks.squeeze(2)
            masks = F.interpolate(masks, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)
            # cls_logits[..., 1:2] = cls_logits[..., 1:2] * n_iou_preds
            seg_logits = self.post_process(cls_logits.detach(), masks.detach())
            seg_logits = seg_logits > self.threshold
        else:
            cls_logits, pred_masks, n_iou_preds = self.forward_sam_prompt_generator_all(
                batch)  # 1x100x2, 1x100x1x256x256, 1x100x1
            pred_masks = pred_masks.squeeze(2)
            pred_masks = F.interpolate(pred_masks, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)
            # cls_logits[..., 1:2] = cls_logits[..., 1:2] * n_iou_preds
            seg_logits = self.post_process(cls_logits.detach(), pred_masks.detach())
            seg_logits = seg_logits > self.threshold
        # import ipdb; ipdb.set_trace()
        self.val_evaluator.update(seg_logits, seg_label)

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        cls_logits, n_img_masks = self.forward(batch)

        seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)
        seg_label = seg_label.squeeze(1)
        masks = F.interpolate(n_img_masks, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)
        masks = masks.squeeze(1) > 0
        self.evaluator.update(masks, seg_label)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_masks = data_sample.instances_data.long()
            gt_labels = data_sample.instances_label.long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def training_step(self, batch, batch_idx):
        if self.only_img_encoder:
            masks_pred = self.forward_only_img_encoder(batch)
            seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)
            masks_pred = F.interpolate(masks_pred, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)
            losses = self.head.loss(masks_pred, seg_label)
            masks_pred_result = masks_pred > 0
            self.train_evaluator.update(masks_pred_result.detach(), seg_label.detach())

        elif self.only_decoder:
            cls_logits, masks, n_iou_preds = self.forward_sam_prompt_generator(batch)  # 1x100x2, 1x100x1x256x256, 1x100x1
            masks = masks.squeeze(2)
            seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)
            masks = F.interpolate(masks, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)
            # cls_logits[..., 1:2] = cls_logits[..., 1:2] * n_iou_preds
            seg_logits = self.post_process(cls_logits.clone().detach(), masks.clone().detach())
            seg_logits = seg_logits > self.threshold
            self.train_evaluator.update(seg_logits, seg_label)

            batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
                batch['data_samples'])

            losses = self.head.loss(cls_logits, masks, batch_gt_instances, batch_img_metas)
        else:
            cls_logits, pred_masks, n_iou_preds = self.forward_sam_prompt_generator_all(
                batch)  # 1x100x2, 1x100x1x256x256, 1x100x1
            pred_masks = pred_masks.squeeze(2)
            if torch.isinf(pred_masks).any() or torch.isnan(pred_masks).any():
                # import ipdb;
                # ipdb.set_trace()
                # raise ValueError('cost is nan in CrossEntropyLossCost')
                print('!!!!!!!!!!!!!!!!!!!!loss is nan or inf!!!!!!!!!!!!!!!!!!')
                return torch.tensor(0.0, requires_grad=True, device=self.device)
            seg_label = torch.stack([x.gt_sem_seg.data for x in batch['data_samples']], dim=0)
            pred_masks = F.interpolate(pred_masks, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)
            # cls_logits[..., 1:2] = cls_logits[..., 1:2] * n_iou_preds
            seg_logits = self.post_process(cls_logits.clone().detach(), pred_masks.clone().detach())
            seg_logits = seg_logits > self.threshold
            self.train_evaluator.update(seg_logits, seg_label)

            batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
                batch['data_samples'])

            losses = self.head.loss(cls_logits, pred_masks, batch_gt_instances, batch_img_metas)

        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def on_before_optimizer_step(self, optimizer) -> None:
        self.log_grad(module=self.sam_prompt_generator)

    def post_process(self, mask_cls_results, mask_pred_results):
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., 1:2]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits

    def forward_only_img_encoder(self, batch, *args: Any, **kwargs: Any) -> Any:
        if self.with_clip:
            clip_dense_embs = torch.stack([x.clip_dense_embs for x in batch['data_samples']], dim=0)
            logits_per_images = torch.stack([x.logits_per_image for x in batch['data_samples']], dim=0)
            logits_per_images = self.logits_prompt(logits_per_images)  # Bx576x16
            clip_dense_embs = torch.cat([clip_dense_embs, logits_per_images], dim=-1)
            clip_dense_embs = rearrange(clip_dense_embs, 'b (h w) c -> b c h w', h=int(clip_dense_embs.shape[1]**0.5))
            masks_pred = self.global_prompt(clip_dense_embs)
        else:
            image_embeddings = torch.stack([x.image_embeddings for x in batch['data_samples']], dim=0)
            masks_pred = self.global_prompt(image_embeddings)
        return masks_pred

    def forward_sam_prompt_generator(self, batch, *args: Any, **kwargs: Any) -> Any:
        inner_states = [x.inner_states for x in batch['data_samples']]
        image_embeddings = torch.stack([x.image_embeddings for x in batch['data_samples']], dim=0)

        inner_states_tmp = []
        for idx in range(len(inner_states[0])):
            inner_states_tmp.append(torch.stack([x[idx] for x in inner_states], dim=0).to(image_embeddings.device))

        point_embs, cls_logits = self.sam_prompt_generator(inner_states_tmp)

        # if has points prompt, then get points embeddings
        if hasattr(self, 'point_grids'):
            points_scale = np.array(img.shape[-2:], dtype=np.float32).reshape(1, -1)  # 2,
            points_for_image = self.point_grids[0] * points_scale
            in_points = torch.as_tensor(points_for_image, device=img.device)
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            in_points = rearrange(in_points, 'n c -> n () c')
            in_labels = rearrange(in_labels, 'n -> n ()')
            points = (in_points, in_labels)

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )  # 1024x2x256; 1024x256x64x64
        else:
            # ponits_embeddings B T N C
            sparse_embeddings = point_embs
            dense_embeddings = self.prompt_encoder.no_mask_embed.weight.view(1, 1, -1, 1, 1).expand(
                sparse_embeddings.shape[0], sparse_embeddings.shape[1], -1,
                self.prompt_encoder.image_embedding_size[0], self.prompt_encoder.image_embedding_size[1]
                )


        n_img_masks = []
        n_iou_preds = []
        n_class_aware_probs = []
        for curr_img_embedding, cur_s_emb, cur_d_emb in zip(image_embeddings, sparse_embeddings, dense_embeddings):
            lr_masks, iou_pred, class_aware_prob = self.mask_decoder(
                image_embeddings=curr_img_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=cur_s_emb,
                dense_prompt_embeddings=cur_d_emb
            )
            mask_slice = slice(0, 1)
            masks = lr_masks[:, mask_slice, :, :]
            iou_pred = iou_pred[:, mask_slice]
            class_aware_prob = class_aware_prob[:, mask_slice]

            n_img_masks.append(masks)
            n_iou_preds.append(iou_pred)
        n_img_masks = torch.stack(n_img_masks, dim=0)
        n_iou_preds = torch.stack(n_iou_preds, dim=0)

        return cls_logits, n_img_masks, n_iou_preds

    def forward_sam_prompt_generator_all(self, batch, *args: Any, **kwargs: Any) -> Any:
        x = torch.stack(batch['inputs'], dim=0)
        # if self.local_rank == 0:
        #     import pdb; pdb.set_trace()
        # self.trainer.strategy.barrier()
        x = x[:, [2, 1, 0], :, :]  # BGR -> RGB
        x = (x - self.img_encoder.pixel_mean) / self.img_encoder.pixel_std
        with torch.no_grad():
            image_embeddings, inner_states = self.img_encoder(x)

        point_embs, cls_logits = self.sam_prompt_generator(inner_states)

        # if has points prompt, then get points embeddings
        if hasattr(self, 'point_grids'):
            points_scale = np.array(img.shape[-2:], dtype=np.float32).reshape(1, -1)  # 2,
            points_for_image = self.point_grids[0] * points_scale
            in_points = torch.as_tensor(points_for_image, device=img.device)
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            in_points = rearrange(in_points, 'n c -> n () c')
            in_labels = rearrange(in_labels, 'n -> n ()')
            points = (in_points, in_labels)

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )  # 1024x2x256; 1024x256x64x64
        else:
            # ponits_embeddings B T N C
            sparse_embeddings = point_embs
            dense_embeddings = self.prompt_encoder_no_mask_embed(torch.tensor([0], device=self.device)).view(1, 1, -1, 1, 1).expand(
                sparse_embeddings.shape[0], sparse_embeddings.shape[1], -1,
                image_embeddings.shape[-2], image_embeddings.shape[-1]
                )


        n_img_masks = []
        n_iou_preds = []
        n_class_aware_probs = []
        for curr_img_embedding, cur_s_emb, cur_d_emb in zip(image_embeddings, sparse_embeddings, dense_embeddings):
            lr_masks, iou_pred, class_aware_prob = self.mask_decoder(
                image_embeddings=curr_img_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=cur_s_emb,
                dense_prompt_embeddings=cur_d_emb
            )
            if self.train_head:
                masks = lr_masks
                iou_pred = iou_pred
            else:
                mask_slice = slice(0, 1)
                masks = lr_masks[:, mask_slice, :, :]
                iou_pred = iou_pred[:, mask_slice]

            n_img_masks.append(masks)
            n_iou_preds.append(iou_pred)
        n_img_masks = torch.stack(n_img_masks, dim=0)
        n_iou_preds = torch.stack(n_iou_preds, dim=0)

        return cls_logits, n_img_masks, n_iou_preds

    def vis_inter_states(self, batch, masks, *args: Any, **kwargs: Any):
        folder = 'results/tmp'
        import cv2
        cv2.imwrite(os.path.join(folder, f'img.png'), batch['inputs'][0].permute((1, 2, 0)).detach().cpu().numpy())
        cv2.imwrite(os.path.join(folder, f'label_mask.png'), seg_label[0][0].detach().cpu().numpy() * 255)
        masks = masks > 0
        for idx, mask_pred in enumerate(masks[0]):
            cv2.imwrite(os.path.join(folder, f'pred_mask_{idx}.png'), mask_pred[0].detach().cpu().numpy() * 255)
        import ipdb; ipdb.set_trace()






