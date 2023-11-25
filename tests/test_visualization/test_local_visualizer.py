import os
from unittest import TestCase

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData, PixelData

from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer, TrackLocalVisualizer


def _rand_bboxes(num_boxes, h, w):
    cx, cy, bw, bh = torch.rand(num_boxes, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clamp(0, w)
    tl_y = ((cy * h) - (h * bh / 2)).clamp(0, h)
    br_x = ((cx * w) + (w * bw / 2)).clamp(0, w)
    br_y = ((cy * h) + (h * bh / 2)).clamp(0, h)

    bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=0).T
    return bboxes


def _create_panoptic_data(num_boxes, h, w):
    sem_seg = np.zeros((h, w), dtype=np.int64) + 2
    bboxes = _rand_bboxes(num_boxes, h, w).int()
    labels = torch.randint(2, (num_boxes, ))
    for i in range(num_boxes):
        x, y, w, h = bboxes[i]
        sem_seg[y:y + h, x:x + w] = (i + 1) * INSTANCE_OFFSET + labels[i]

    return sem_seg[None]


class TestDetLocalVisualizer(TestCase):

    def test_add_datasample(self):
        h = 12
        w = 10
        num_class = 3
        num_bboxes = 5
        out_file = 'out_file.jpg'

        image = np.random.randint(0, 256, size=(h, w, 3)).astype('uint8')

        # test gt_instances
        gt_instances = InstanceData()
        gt_instances.bboxes = _rand_bboxes(num_bboxes, h, w)
        gt_instances.labels = torch.randint(0, num_class, (num_bboxes, ))
        det_data_sample = DetDataSample()
        det_data_sample.gt_instances = gt_instances

        det_local_visualizer = DetLocalVisualizer()
        det_local_visualizer.add_datasample(
            'image', image, det_data_sample, draw_pred=False)

        # test out_file
        det_local_visualizer.add_datasample(
            'image',
            image,
            det_data_sample,
            draw_pred=False,
            out_file=out_file)
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == (h, w, 3)
        os.remove(out_file)

        # test gt_instances and pred_instances
        pred_instances = InstanceData()
        pred_instances.bboxes = _rand_bboxes(num_bboxes, h, w)
        pred_instances.labels = torch.randint(0, num_class, (num_bboxes, ))
        pred_instances.scores = torch.rand((num_bboxes, ))
        det_data_sample.pred_instances = pred_instances

        det_local_visualizer.add_datasample(
            'image', image, det_data_sample, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2, 3))

        det_local_visualizer.add_datasample(
            'image', image, det_data_sample, draw_gt=False, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        det_local_visualizer.add_datasample(
            'image',
            image,
            det_data_sample,
            draw_pred=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        # test gt_panoptic_seg and pred_panoptic_seg
        det_local_visualizer.dataset_meta = dict(classes=('1', '2'))
        gt_sem_seg = _create_panoptic_data(num_bboxes, h, w)
        panoptic_seg = PixelData(sem_seg=gt_sem_seg)

        det_data_sample = DetDataSample()
        det_data_sample.gt_panoptic_seg = panoptic_seg

        pred_sem_seg = _create_panoptic_data(num_bboxes, h, w)
        panoptic_seg = PixelData(sem_seg=pred_sem_seg)
        det_data_sample.pred_panoptic_seg = panoptic_seg

        det_local_visualizer.add_datasample(
            'image', image, det_data_sample, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2, 3))

        # class information must be provided
        det_local_visualizer.dataset_meta = {}
        with self.assertRaises(AssertionError):
            det_local_visualizer.add_datasample(
                'image', image, det_data_sample, out_file=out_file)

    def _assert_image_and_shape(self, out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape
        os.remove(out_file)


class TestTrackLocalVisualizer(TestCase):

    @staticmethod
    def _get_gt_instances():
        bboxes = np.array([[912, 484, 1009, 593], [1338, 418, 1505, 797]])
        masks = np.zeros((2, 1080, 1920), dtype=np.bool_)
        for i, bbox in enumerate(bboxes):
            masks[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = True
        instances_data = dict(
            bboxes=torch.tensor(bboxes),
            masks=masks,
            instances_id=torch.tensor([1, 2]),
            labels=torch.tensor([0, 1]))
        instances = InstanceData(**instances_data)
        return instances

    @staticmethod
    def _get_pred_instances():
        instances_data = dict(
            bboxes=torch.tensor([[900, 500, 1000, 600], [1300, 400, 1500,
                                                         800]]),
            instances_id=torch.tensor([1, 2]),
            labels=torch.tensor([0, 1]),
            scores=torch.tensor([0.955, 0.876]))
        instances = InstanceData(**instances_data)
        return instances

    @staticmethod
    def _assert_image_and_shape(out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape
        os.remove(out_file)

    def test_add_datasample(self):
        out_file = 'out_file.jpg'
        h, w = 1080, 1920
        image = np.random.randint(0, 256, size=(h, w, 3)).astype('uint8')
        gt_instances = self._get_gt_instances()
        pred_instances = self._get_pred_instances()
        image_data_sample = DetDataSample()
        image_data_sample.gt_instances = gt_instances
        image_data_sample.pred_track_instances = pred_instances

        track_local_visualizer = TrackLocalVisualizer(alpha=0.2)
        track_local_visualizer.dataset_meta = dict(
            classes=['pedestrian', 'vehicle'])

        # test gt_instances
        track_local_visualizer.add_datasample('image', image,
                                              image_data_sample, None)

        # test out_file
        track_local_visualizer.add_datasample(
            'image', image, image_data_sample, None, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        # test gt_instances and pred_instances
        track_local_visualizer.add_datasample(
            'image', image, image_data_sample, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, 2 * w, 3))

        track_local_visualizer.add_datasample(
            'image',
            image,
            image_data_sample,
            draw_gt=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        track_local_visualizer.add_datasample(
            'image',
            image,
            image_data_sample,
            draw_pred=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))
