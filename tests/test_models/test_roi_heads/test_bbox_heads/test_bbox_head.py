# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.roi_heads.bbox_heads import (BBoxHead, Shared2FCBBoxHead,
                                               Shared4Conv1FCBBoxHead)
from mmdet.models.task_modules.samplers import SamplingResult


class TestBboxHead(TestCase):

    def test_init(self):
        # Shared2FCBBoxHead
        bbox_head = Shared2FCBBoxHead(
            in_channels=1, fc_out_channels=1, num_classes=4)
        self.assertTrue(bbox_head.fc_cls)
        self.assertTrue(bbox_head.fc_reg)
        self.assertEqual(len(bbox_head.shared_fcs), 2)

        # Shared4Conv1FCBBoxHead
        bbox_head = Shared4Conv1FCBBoxHead(
            in_channels=1, fc_out_channels=1, num_classes=4)
        self.assertTrue(bbox_head.fc_cls)
        self.assertTrue(bbox_head.fc_reg)
        self.assertEqual(len(bbox_head.shared_convs), 4)
        self.assertEqual(len(bbox_head.shared_fcs), 1)

    def test_bbox_head_get_results(self):
        num_classes = 6
        bbox_head = BBoxHead(reg_class_agnostic=True, num_classes=num_classes)
        s = 128
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
        }]

        num_samples = 2
        rois = [torch.rand((num_samples, 5))]
        cls_scores = [torch.rand((num_samples, num_classes + 1))]
        bbox_preds = [torch.rand((num_samples, 4))]

        # with nms
        rcnn_test_cfg = ConfigDict(
            score_thr=0.,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        result_list = bbox_head.predict_by_feat(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas,
            rcnn_test_cfg=rcnn_test_cfg)

        self.assertLessEqual(len(result_list[0]), num_samples * num_classes)
        self.assertIsInstance(result_list[0], InstanceData)
        self.assertEqual(result_list[0].bboxes.shape[1], 4)
        self.assertEqual(len(result_list[0].scores.shape), 1)
        self.assertEqual(len(result_list[0].labels.shape), 1)

        # without nms
        result_list = bbox_head.predict_by_feat(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas)

        self.assertIsInstance(result_list[0], InstanceData)
        self.assertEqual(len(result_list[0]), num_samples)
        self.assertEqual(result_list[0].bboxes.shape, bbox_preds[0].shape)
        self.assertEqual(result_list[0].scores.shape, cls_scores[0].shape)
        self.assertIsNone(result_list[0].get('label', None))

        # num_samples is 0
        num_samples = 0
        rois = [torch.rand((num_samples, 5))]
        cls_scores = [torch.rand((num_samples, num_classes + 1))]
        bbox_preds = [torch.rand((num_samples, 4))]

        # with nms
        rcnn_test_cfg = ConfigDict(
            score_thr=0.,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        result_list = bbox_head.predict_by_feat(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas,
            rcnn_test_cfg=rcnn_test_cfg)

        self.assertIsInstance(result_list[0], InstanceData)
        self.assertEqual(len(result_list[0]), 0)
        self.assertEqual(result_list[0].bboxes.shape[1], 4)

        # without nms
        result_list = bbox_head.predict_by_feat(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas)

        self.assertIsInstance(result_list[0], InstanceData)
        self.assertEqual(len(result_list[0]), 0)
        self.assertEqual(result_list[0].bboxes.shape, bbox_preds[0].shape)
        self.assertIsNone(result_list[0].get('label', None))

    def test_bbox_head_refine_bboxes(self):
        num_classes = 6
        bbox_head = BBoxHead(reg_class_agnostic=True, num_classes=num_classes)
        s = 128
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        sampling_results = [SamplingResult.random()]
        num_samples = 20
        rois = torch.rand((num_samples, 4))
        roi_img_ids = torch.zeros(num_samples, 1)
        rois = torch.cat((roi_img_ids, rois), dim=1)
        cls_scores = torch.rand((num_samples, num_classes + 1))
        bbox_preds = torch.rand((num_samples, 4))
        labels = torch.randint(0, num_classes + 1, (num_samples, )).long()
        bbox_targets = (labels, None, None, None)
        bbox_results = dict(
            rois=rois,
            bbox_pred=bbox_preds,
            cls_score=cls_scores,
            bbox_targets=bbox_targets)

        bbox_list = bbox_head.refine_bboxes(
            sampling_results=sampling_results,
            bbox_results=bbox_results,
            batch_img_metas=img_metas)

        self.assertGreaterEqual(num_samples, len(bbox_list[0]))
        self.assertIsInstance(bbox_list[0], InstanceData)
        self.assertEqual(bbox_list[0].bboxes.shape[1], 4)
