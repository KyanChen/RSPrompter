import torch
from mmengine.structures import InstanceData
from typing import List, Any

from mmpl.registry import MODELS
from mmseg.utils import SampleList
from .base_pler import BasePLer
import torch.nn.functional as F
from modules.sam import sam_model_registry


@MODELS.register_module()
class SegSAMDetPLer(BasePLer):
    def __init__(self,
                 whole_model,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 need_train_names=None,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.need_train_names = need_train_names

        self.whole_model = MODELS.build(whole_model)
        backbone_type = backbone.pop('type')
        self.backbone = sam_model_registry[backbone_type](**backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.need_train_names is not None:
            self._set_grad(self.need_train_names, noneed_train_names=[])

    def init_weights(self):
        import ipdb; ipdb.set_trace()
        pass

    def train(self, mode=True):
        if self.need_train_names is not None:
            return self._set_train_module(mode, self.need_train_names)
        else:
            super().train(mode)
            return self

    def validation_step(self, batch, batch_idx):
        data = self.whole_model.data_preprocessor(batch, False)
        batch_data_samples = self.whole_model._run_forward(data, mode='predict')  # type: ignore

        batch_inputs = data['inputs']
        feat, inter_features = self.backbone.image_encoder(batch_inputs)
        # import ipdb; ipdb.set_trace()
        for idx, data_sample in enumerate(batch_data_samples):
            bboxes = data_sample.pred_instances['bboxes']
            ori_img_shape = data_sample.ori_shape
            if len(bboxes) == 0:
                im_mask = torch.zeros(
                    0,
                    ori_img_shape[0],
                    ori_img_shape[1],
                    device=self.device,
                    dtype=torch.bool)
            else:
                scale_factor = data_sample.scale_factor
                repeat_num = int(bboxes.size(-1) / 2)
                scale_factor = bboxes.new_tensor(scale_factor).repeat((1, repeat_num))
                bboxes = bboxes * scale_factor

                # Embed prompts
                sparse_embeddings, dense_embeddings = self.backbone.prompt_encoder(
                    points=None,
                    boxes=bboxes,
                    masks=None,
                )

                # Predict masks
                low_res_masks, iou_predictions = self.backbone.mask_decoder(
                    image_embeddings=feat[idx:idx + 1],
                    image_pe=self.backbone.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                # Upscale the masks to the original image resolution
                im_mask = F.interpolate(low_res_masks, ori_img_shape, mode="bilinear", align_corners=False)
                im_mask = im_mask > 0
                im_mask = im_mask.squeeze(1)
            data_sample.pred_instances.masks = im_mask

        self.val_evaluator.update(batch, batch_data_samples)

    def training_step(self, batch, batch_idx):
        data = self.whole_model.data_preprocessor(batch, True)
        losses = self.whole_model._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def on_before_optimizer_step(self, optimizer) -> None:
        self.log_grad(module=self.whole_model)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        data = self.whole_model.data_preprocessor(batch, False)
        batch_data_samples = self.whole_model._run_forward(data, mode='predict')  # type: ignore

        batch_inputs = data['inputs']
        feat, inter_features = self.backbone.image_encoder(batch_inputs)
        # import ipdb; ipdb.set_trace()
        for idx, data_sample in enumerate(batch_data_samples):
            bboxes = data_sample.pred_instances['bboxes']
            ori_img_shape = data_sample.ori_shape
            if len(bboxes) == 0:
                im_mask = torch.zeros(
                    0,
                    ori_img_shape[0],
                    ori_img_shape[1],
                    device=self.device,
                    dtype=torch.bool)
            else:
                scale_factor = data_sample.scale_factor
                repeat_num = int(bboxes.size(-1) / 2)
                scale_factor = bboxes.new_tensor(scale_factor).repeat((1, repeat_num))
                bboxes = bboxes * scale_factor

                # Embed prompts
                sparse_embeddings, dense_embeddings = self.backbone.prompt_encoder(
                    points=None,
                    boxes=bboxes,
                    masks=None,
                )

                # Predict masks
                low_res_masks, iou_predictions = self.backbone.mask_decoder(
                    image_embeddings=feat[idx:idx + 1],
                    image_pe=self.backbone.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                # Upscale the masks to the original image resolution
                im_mask = F.interpolate(low_res_masks, ori_img_shape, mode="bilinear", align_corners=False)
                im_mask = im_mask > 0
                im_mask = im_mask.squeeze(1)
            data_sample.pred_instances.masks = im_mask

        return batch_data_samples





