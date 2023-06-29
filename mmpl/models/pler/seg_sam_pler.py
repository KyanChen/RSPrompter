import torch
from mmengine.structures import InstanceData
from typing import List, Any

from mmpl.registry import MODELS
from mmseg.utils import SampleList
from .base_pler import BasePLer
import torch.nn.functional as F
from modules.sam import sam_model_registry


@MODELS.register_module()
class SegSAMPLer(BasePLer):
    def __init__(self,
                 backbone,
                 sam_neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 need_train_names=None,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.need_train_names = need_train_names

        backbone_type = backbone.pop('type')
        self.backbone = sam_model_registry[backbone_type](**backbone)

        if sam_neck is not None:
            self.sam_neck = MODELS.build(sam_neck)

        panoptic_head_ = panoptic_head.deepcopy()
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = MODELS.build(panoptic_head_)

        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

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

    @torch.no_grad()
    def extract_feat(self, batch_inputs):
        feat, inter_features = self.backbone.image_encoder(batch_inputs)
        return feat, inter_features

    def validation_step(self, batch, batch_idx):
        data = self.data_preprocessor(batch, False)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']

        feats = self.extract_feat(batch_inputs)
        if hasattr(self, 'sam_neck'):
            feats = self.sam_neck(feats)
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                feats, batch_data_samples)
        else:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                feats, batch_data_samples, self.backbone)

        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            batch_data_samples,
            rescale=True)
        results = self.add_pred_to_datasample(batch_data_samples, results_list)

        # preds = []
        # targets = []
        # for data_sample in results:
        #     result = dict()
        #     pred = data_sample.pred_instances
        #     result['boxes'] = pred['bboxes']
        #     result['scores'] = pred['scores']
        #     result['labels'] = pred['labels']
        #     if 'masks' in pred:
        #         result['masks'] = pred['masks']
        #     preds.append(result)
        #     # parse gt
        #     gt = dict()
        #     gt_data = data_sample.get('gt_instances', None)
        #     gt['boxes'] = gt_data['bboxes']
        #     gt['labels'] = gt_data['labels']
        #     if 'masks' in pred:
        #         gt['masks'] = gt_data['masks'].to_tensor(dtype=torch.bool, device=result['masks'].device)
        #     targets.append(gt)
        #
        # self.val_evaluator.update(preds, targets)
        self.val_evaluator.update(batch, results)

    def training_step(self, batch, batch_idx):
        data = self.data_preprocessor(batch, True)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']
        x = self.extract_feat(batch_inputs)
        if hasattr(self, 'sam_neck'):
            x = self.sam_neck(x)
            losses = self.panoptic_head.loss(x, batch_data_samples)
        else:
            losses = self.panoptic_head.loss(x, batch_data_samples, self.backbone)
        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def on_before_optimizer_step(self, optimizer) -> None:
        self.log_grad(module=self.panoptic_head)


    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[dict]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (List[dict]): Instance segmentation, segmantic
                segmentation and panoptic segmentation results.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                data_sample.pred_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                data_sample.pred_instances = pred_results['ins_results']

            assert 'sem_results' not in pred_results, 'segmantic ' \
                'segmentation results are not supported yet.'

        return data_samples


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        data = self.data_preprocessor(batch, False)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']
        # import ipdb; ipdb.set_trace()
        feats = self.extract_feat(batch_inputs)
        if hasattr(self, 'sam_neck'):
            feats = self.sam_neck(feats)
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                feats, batch_data_samples)
        else:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                feats, batch_data_samples, self.backbone)

        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            batch_data_samples,
            rescale=True)
        results = self.add_pred_to_datasample(batch_data_samples, results_list)
        return results





