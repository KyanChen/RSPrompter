import sys

import cv2
sys.path.append(sys.path[0] + '/../../../..')
from mmengine import Config, ProgressBar
from mmengine.dataset import Compose
from mmengine.structures import InstanceData
from torchvision.transforms import InterpolationMode

from mmpl.registry import DATASETS, MODELS, METRICS
from tools.ins_seg.sam.sam_cls.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import mmpl.evaluation
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

dataset_name = 'whu'
seg_model_cfg_file = f'configs/ins_seg/samcls_{dataset_name}_config.py'
# sam_checkpoint = "pretrain/sam/sam_vit_b_01ec64.pth"
# model_type = "vit_b"
sam_checkpoint = "pretrain/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()
mask_generator = SamAutomaticMaskGenerator(
    sam,
    pred_iou_thresh=0.5,
    box_nms_thresh=0.5,
    stability_score_thresh=0.6,
    min_mask_region_area=16,
    crop_nms_thresh=0.6,
)

# cls model
cls_model_cfg_file = f'configs/ins_seg/samcls_res18_{dataset_name}_config.py'
cls_ckpt = 'results/whu_ins/E20230607_0/checkpoints/epoch_epoch=59-map_valmulticlassaccuracy_0=0.9516.ckpt'
cls_cfg = Config.fromfile(cls_model_cfg_file)
cls_model_cfg = cls_cfg.get('model_cfg').whole_model
cls_model = MODELS.build(cls_model_cfg)
cls_state_dict = torch.load(cls_ckpt, map_location='cpu')['state_dict']
cls_state_dict = {k.replace('whole_model.', ''): v for k, v in cls_state_dict.items()}
cls_model.load_state_dict(cls_state_dict, strict=True)
cls_model.to(device=device)
cls_model.eval()

cls_transform_cfg = cls_cfg.get('datamodule_cfg').val_loader.dataset.pipeline[1:]
cls_transforms = Compose(cls_transform_cfg)

seg_cfg = Config.fromfile(seg_model_cfg_file)
seg_dataset_cfg = seg_cfg.get('datamodule_cfg').val_loader.dataset
seg_dataset = DATASETS.build(seg_dataset_cfg)
val_evaluator_cfg = seg_cfg['evaluator'].val_evaluator
val_evaluator = METRICS.build(val_evaluator_cfg)

val_evaluator.dataset_meta = seg_dataset.metainfo


progress_bar = ProgressBar(len(seg_dataset))
expand_ratio = 2
for index in range(len(seg_dataset)):
    seg_data = seg_dataset[index]
    img_file = seg_data['data_samples'].img_path
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    refined_masks = []
    for mask in masks:
        bbbox = mask['bbox']
        x, y, w, h = bbbox
        if w < 4 or h < 4:
            continue
        refined_masks.append(mask)
    masks = refined_masks
    mask_cls = []
    for mask in masks:
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

        # blur image
        blur_image = image.copy()
        blur_image = cv2.blur(blur_image, (7, 7))
        blur_image[mask['segmentation']] = image[mask['segmentation']]
        crop_image = blur_image[t:b, l:r]

        # # debug to show the image
        # cv2.imshow("crop_image", crop_image)
        # seg_mask = image.copy()
        # seg_mask[mask['segmentation']] = (0, 0, 255)
        # seg_mask[mask['segmentation']] = 0.5 * seg_mask[mask['segmentation']] + 0.5 * image[mask['segmentation']]
        # cv2.imshow("image", seg_mask.astype('uint8'))
        # cv2.waitKey(0)
        results = {
            'img': crop_image,
        }
        transform_crop_data = cls_transforms(results)
        transform_crop_data['inputs'] = transform_crop_data['inputs'].unsqueeze(0).to(device)
        transform_crop_data['data_samples'] = [transform_crop_data['data_samples']]
        data = cls_model.data_preprocessor(transform_crop_data, False)
        results = cls_model._run_forward(data, mode='predict')

        mask_cls.append(results[0].pred_score)

    mask_pred = torch.stack([torch.from_numpy(mask['segmentation']) for mask in masks], dim=0).to(device=device)
    bbox_pred = [mask['bbox'] for mask in masks]
    bbox_pred = torch.tensor(bbox_pred, dtype=torch.float32, device=device)
    bbox_pred[:, 2:] += bbox_pred[:, :2]

    mask_cls = torch.stack(mask_cls, dim=0)
    max_per_image = 100
    num_queries = mask_cls.shape[0]
    num_classes = mask_cls.shape[-1] - 1
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    labels = torch.arange(num_classes, device=mask_cls.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
    try:
        scores_per_image, top_indices = scores.flatten(0, 1).topk(max_per_image, sorted=False)
    except Exception as e:
        print(e)
        continue

    labels_per_image = labels[top_indices]
    query_indices = top_indices // num_classes
    mask_pred = mask_pred[query_indices]
    bbox_pred = bbox_pred[query_indices]

    # # extract things
    # is_thing = labels_per_image < self.num_things_classes
    # scores_per_image = scores_per_image[is_thing]
    # labels_per_image = labels_per_image[is_thing]
    # mask_pred = mask_pred[is_thing]

    mask_pred_binary = (mask_pred > 0).float()
    # mask_scores_per_image = (mask_pred.sigmoid() *
    #                          mask_pred_binary).flatten(1).sum(1) / (
    #                                 mask_pred_binary.flatten(1).sum(1) + 1e-6)
    # det_scores = scores_per_image * mask_scores_per_image
    det_scores = scores_per_image
    mask_pred_binary = mask_pred_binary.bool()

    results = InstanceData()
    results.bboxes = bbox_pred
    results.labels = labels_per_image
    results.scores = det_scores
    results.masks = mask_pred_binary

    data_samples = seg_data['data_samples']
    data_samples.pred_instances = results

    val_evaluator.update(None, [data_samples])
    progress_bar.update()

metrics = val_evaluator.compute()
print(metrics)

