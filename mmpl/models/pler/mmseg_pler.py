import os
from typing import Any

import mmengine
import numpy as np
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


@MODELS.register_module()
class MMSegPLer(BasePLer):
    def __init__(self,
                 whole_model=None,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.whole_model = MODELS.build(whole_model)

    def setup(self, stage: str) -> None:
        pass

    def init_weights(self):
        import ipdb; ipdb.set_trace()
        pass

    def training_step(self, batch, batch_idx):
        data = self.whole_model.data_preprocessor(batch, True)
        losses = self.whole_model._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars
        # return torch.tensor(0.0, requires_grad=True, device=self.device)

    def validation_step(self, batch, batch_idx):
        data = self.whole_model.data_preprocessor(batch, False)
        data_samples = self.whole_model._run_forward(data, mode='predict')
        pred = [data_sample.pred_sem_seg.data for data_sample in data_samples]
        label = [data_sample.gt_sem_seg.data for data_sample in data_samples]
        pred = torch.cat(pred, dim=0)
        label = torch.cat(label, dim=0)
        self.val_evaluator.update(pred, label)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        data = self.whole_model.data_preprocessor(batch, False)
        data_samples = self.whole_model._run_forward(data, mode='predict')
        return data_samples









