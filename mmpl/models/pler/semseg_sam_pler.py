import torch
from mmengine.structures import InstanceData, PixelData
from typing import List

from torch import Tensor

from mmpl.registry import MODELS
from mmseg.models.utils import resize
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList, OptSampleList
from .base_pler import BasePLer
import torch.nn.functional as F
from modules.sam import sam_model_registry


@MODELS.register_module()
class SemSegSAMPLer(BasePLer):
    def __init__(self,
                 backbone,
                 adaphead=None,
                 decode_head=None,
                 need_train_names=None,
                 align_corners=False,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.need_train_names = need_train_names
        self.align_corners = align_corners

        backbone_type = backbone.pop('type')
        delete_submodel = backbone.pop('delete_submodel', [])
        self.backbone = sam_model_registry[backbone_type](**backbone)
        for submodel in delete_submodel:
            delattr(self.backbone, submodel)

        if adaphead is not None:
            self.adaphead = MODELS.build(adaphead)

        decode_head_ = decode_head.deepcopy()
        decode_head_.update(train_cfg=train_cfg)
        decode_head_.update(test_cfg=test_cfg)
        self.decode_head = MODELS.build(decode_head_)

        self.num_classes = self.decode_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def setup(self, stage: str) -> None:
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

    def extract_feat(self, batch_inputs):
        x0, x1 = self.adaphead(batch_inputs, self.backbone.image_encoder)
        return x0, x1

    def validation_step(self, batch, batch_idx):
        data = self.data_preprocessor(batch, False)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']

        if batch_data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in batch_data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=batch_inputs.shape[2:],
                    img_shape=batch_inputs.shape[2:],
                    pad_shape=batch_inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * batch_inputs.shape[0]

        x = self.extract_feat(batch_inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)

        results = self.postprocess_result(seg_logits, batch_data_samples)

        preds = []
        targets = []
        for data_sample in results:
            pred_label = data_sample.pred_sem_seg.data.squeeze()
            label = data_sample.gt_sem_seg.data.squeeze().to(pred_label)

            preds.append(pred_label)
            targets.append(label)
        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)
        self.val_evaluator.update(preds, targets)

    def training_step(self, batch, batch_idx):
        # import ipdb; ipdb.set_trace()
        data = self.data_preprocessor(batch, True)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']
        x = self.extract_feat(batch_inputs)
        losses = self.decode_head.loss(x, batch_data_samples)
        # import ipdb; ipdb.set_trace()
        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def on_before_optimizer_step(self, optimizer) -> None:
        self.log_grad(module=self.adaphead)

    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples




