import os
import cv2
import sys
sys.path.append(sys.path[0] + '/../../../..')
import torch
from mmengine import Config, ProgressBar
from torchvision.transforms import InterpolationMode
from mmpl.registry import DATASETS
from tools.ins_seg.sam.sam_cls.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
from mmdet.evaluation.functional import bbox_overlaps


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BICUBIC = InterpolationMode.BICUBIC

data_set_name = 'whu'

config_file = f'configs/ins_seg/samcls_{data_set_name}_config.py'
# sam_checkpoint = "pretrain/sam/sam_vit_b_01ec64.pth"
# model_type = "vit_b"
sam_checkpoint = "pretrain/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
phase = 'val'

cache_data_root = f'/data/kyanchen/cache_data/ins_seg/sam_cls/{data_set_name}'
cache_data_root = os.path.join(cache_data_root, phase)
if not os.path.exists(cache_data_root):
    os.makedirs(cache_data_root)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    pred_iou_thresh=0.5,
    box_nms_thresh=0.5,
    stability_score_thresh=0.6,
    min_mask_region_area=16,
    crop_nms_thresh=0.6,
)

cfg = Config.fromfile(config_file)
dataset_cfg = cfg.get('datamodule_cfg')
dataset_cfg = dataset_cfg.get(f'{phase}_loader').dataset
dataset = DATASETS.build(dataset_cfg)
class_names = dataset.METAINFO['classes']

progress_bar = ProgressBar(len(dataset))
expand_ratio = 2
iou_thresh = 0.2
# 1741 2700 3700
for index in list(range(len(dataset)))[:500]:
    print(index)
    x = dataset[index]
    img_file = x['data_samples'].img_path
    gt_bbox = x['data_samples'].gt_instances.bboxes
    labels = x['data_samples'].gt_instances.labels
    gt_bbox = gt_bbox.tensor
    # image = cv2.imread(img_file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = x['inputs'].permute(1, 2, 0).numpy()[..., ::-1]
    masks = mask_generator.generate(image)
    pred_boxes = [mask['bbox'] for mask in masks]
    pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32, device=device)
    pred_boxes[:, 2:] += pred_boxes[:, :2]
    # # debug to show the image
    # img = image.copy().astype('uint8')
    # for gt_box in gt_bbox:
    #     gt_box = gt_box.cpu().numpy().astype(int)
    #     cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
    # for gt_box in pred_boxes:
    #     gt_box = gt_box.cpu().numpy().astype(int)
    #     cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 255), 2)
    # cv2.imshow("image", img.astype('uint8'))
    # cv2.waitKey(0)

    ious = bbox_overlaps(gt_bbox.cpu().numpy(), pred_boxes.cpu().numpy())
    idxs = ious.argmax(axis=0)
    ious = ious[idxs, range(ious.shape[1])]
    ious_mask = ious > iou_thresh

    for idx, mask in enumerate(masks):
        # expand box
        x, y, w, h = mask['bbox']
        x = x + w // 2
        y = y + h // 2
        w = int(w * expand_ratio)
        h = int(h * expand_ratio)
        l = int(x - w // 2)
        t = int(y - h // 2)
        r = int(x + w // 2)
        b = int(y + h // 2)
        l = max(0, l)
        t = max(0, t)
        r = min(image.shape[1], r)
        b = min(image.shape[0], b)
        if r - l < 16 or b - t < 16:
            continue

        # blur image
        blur_image = image.copy()
        blur_image = cv2.blur(blur_image, (7, 7))
        blur_image[mask['segmentation']] = image[mask['segmentation']]
        crop_image = blur_image[t:b, l:r]

        # # # debug to show the image
        # cv2.imshow("crop_image", crop_image)
        # seg_mask = image.copy()
        # seg_mask[mask['segmentation']] = (0, 0, 255)
        # seg_mask[mask['segmentation']] = 0.5 * seg_mask[mask['segmentation']] + 0.5 * image[mask['segmentation']]
        # cv2.imshow("image", seg_mask.astype('uint8'))
        # cv2.waitKey(0)

        label = 255
        if ious_mask[idx]:
            label = labels[idxs[idx]].item()
        cv2.imwrite(os.path.join(cache_data_root, f"{index}_{idx}_crop_{label}.jpg"), crop_image)

    progress_bar.update()

