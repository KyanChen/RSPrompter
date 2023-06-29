import os
from typing import Any

import mmengine
import torch
import torch.nn as nn
from einops import rearrange

from mmpl.registry import MODELS
from ..builder import build_backbone, build_loss, build_neck, build_head
from .base_pler import BasePLer


@MODELS.register_module()
class MotionLMGPTPLer(BasePLer):
    def __init__(self,
                 backbone,
                 head=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.backbone = build_backbone(backbone)
        if head is not None:
            self.head = build_head(head)

    def training_val_step(self, batch, batch_idx, prefix=''):
        if hasattr(self, 'data_preprocessor'):
            data = self.data_preprocessor(batch)
            x = data['inputs']
        outputs = self.backbone(**x)

        if hasattr(self, 'head'):
            losses = self.head.loss(
                logits=outputs,
                labels=x['labels'],
                # input_token_len=data['inputs']['input_token_len'],
                # pad_token=self.data_preprocessor.pad_token,
            )
        else:
            losses = dict(
                ce_loss=outputs.loss,
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

    def on_predict_start(self) -> None:
        self.vqvae = self.test_cfg['backbone']
        self.vqvae_preprocessor = self.test_cfg['data_preprocessor']
        load_ckpt_backbone = self.test_cfg['load_ckpt_backbone']
        self.vqvae_preprocessor = MODELS.build(self.vqvae_preprocessor)
        self.vqvae = build_backbone(self.vqvae)

        state_dict = torch.load(load_ckpt_backbone, map_location='cpu')['state_dict']
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = self.vqvae.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            print(f'missing keys: {missing_keys}')
        if len(unexpected_keys) > 0:
            print(f'unexpected keys: {unexpected_keys}')

        self.vqvae = self.vqvae.to(self.device)
        self.vqvae.eval()
        self.vqvae_preprocessor = self.vqvae_preprocessor.to(self.device)
        self.vqvae_preprocessor.eval()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        '''
        data = dict(
            inputs=gt_motion,
            data_samples=batch
        )
        '''

        data = self.vqvae_preprocessor(batch)
        gt_motion = data['inputs']
        assert len(gt_motion) == 1, 'only support batch size 1'

        pred_tokens = self.vqvae.encode(gt_motion)
        pred_vqvae_pose = self.vqvae.forward_decoder(pred_tokens)

        num_prompt = self.test_cfg['num_prompt']
        max_new_tokens = self.test_cfg['max_new_tokens']

        num_return_sequences = self.test_cfg.get('num_return_sequences', 1)
        num_beams = self.test_cfg.get('num_beams', 0)
        do_sample = self.test_cfg.get('do_sample', False)

        pred_tokens_clip = pred_tokens[:, :num_prompt]

        if hasattr(self, 'head'):
            index_motions = self.backbone.sample(pred_tokens_clip, sample_length=max_new_tokens, if_categorial=True)
        else:
            inputs = dict(
                input_ids=pred_tokens_clip,
                attention_mask=torch.ones_like(pred_tokens_clip, dtype=torch.long, device=pred_tokens_clip.device),
            )
            if num_beams > 1:
                outputs = self.backbone.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            else:
                outputs = self.backbone.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            index_motions = outputs.sequences

        pred_gpt_pose = self.vqvae.forward_decoder(index_motions)

        pred_vqvae_pose = self.vqvae_preprocessor.denormalize(pred_vqvae_pose)
        pred_gpt_pose = self.vqvae_preprocessor.denormalize(pred_gpt_pose)

        res = dict()
        res['pred_vqvae_pose'] = pred_vqvae_pose
        for idx, pred_gpt_pose_i in enumerate(pred_gpt_pose):
            res[f'pred_gpt_pose_{idx+1}'] = pred_gpt_pose_i[None, ...]
        return res

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        for k, v in self.mean_std_info.items():
            for kk, vv in v.items():
                self.mean_std_info[k][kk] = vv.to(self.device, dtype=torch.float32)
        gt_motion = batch['motion']
        gt_motion = (gt_motion - self.mean_std_info['motion']['mean']) / self.mean_std_info['motion']['std']
        pred_tokens = self.backbone.encode(gt_motion)
        return pred_tokens





