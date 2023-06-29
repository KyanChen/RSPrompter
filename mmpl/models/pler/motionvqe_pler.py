from typing import Any
from mmpl.registry import MODELS
from ..builder import build_backbone, build_head
from .base_pler import BasePLer


@MODELS.register_module()
class MotionVQVQEPLer(BasePLer):
    def __init__(self,
                 backbone,
                 head,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

    def training_val_step(self, batch, batch_idx, prefix=''):
        if hasattr(self, 'data_preprocessor'):
            data = self.data_preprocessor(batch)
        gt_motion = data['inputs']
        pred_motion, loss_commit, perplexity = self.backbone(gt_motion)

        losses = self.head.loss(
            pred_motion=pred_motion,
            loss_commit=loss_commit,
            perplexity=perplexity,
            gt_motion=gt_motion,
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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if hasattr(self, 'data_preprocessor'):
            data = self.data_preprocessor(batch)
        gt_motion = data['inputs']

        pred_motion, loss_commit, perplexity = self.backbone(gt_motion)
        pred_denorm = self.data_preprocessor.denormalize(pred_motion)
        return pred_denorm

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if hasattr(self, 'data_preprocessor'):
            data = self.data_preprocessor(batch)
        gt_motion = data['inputs']

        pred_tokens = self.backbone.encode(gt_motion)
        return pred_tokens





