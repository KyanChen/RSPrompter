import argparse
import glob
import os
import os.path as osp

# from cityscapesscripts.helpers.labels import Label
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmengine.fileio import dump
from mmengine.utils import (Timer, mkdir_or_exist, track_parallel_progress,
                            track_progress)
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    'm_color',
    ] )


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color          multiplied color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0      ),
    Label(  'ship'                 ,  1 ,        0 , 'transport'       , 1       , True         , False        , (  0,  0, 63) , 4128768),
    Label(  'storage_tank'         ,  2 ,        1 , 'transport'       , 1       , True         , False        , (  0, 63, 63) , 4144896),
    Label(  'baseball_diamond'     ,  3 ,        2 , 'land'            , 2       , True         , False        , (  0, 63, 63) , 16128  ),
    Label(  'tennis_court'         ,  4 ,        3 , 'land'            , 2       , True         , False        , (  0, 63,127) , 8339200),
    Label(  'basketball_court'     ,  5 ,        4 , 'land'            , 2       , True         , False        , (  0, 63,191) , 12533504),
    Label(  'Ground_Track_Field'   ,  6 ,        5 , 'land'            , 2       , True         , False        , (  0, 63,255) , 16727808),
    Label(  'Bridge'               ,  7 ,        6 , 'land'            , 2       , True         , False        , (  0,127, 63) , 4161280),
    Label(  'Large_Vehicle'        ,  8 ,        7 , 'transport'       , 1       , True         , False        , (  0,127,127) , 8355584),
    Label(  'Small_Vehicle'        ,  9 ,        8 , 'transport'       , 1       , True         , False        , (  0,  0,127) , 8323072),
    Label(  'Helicopter'           , 10 ,        9 , 'transport'       , 1       , True         , False        , (  0,  0,191) , 12517376),
    Label(  'Swimming_pool'        , 11 ,       10 , 'land'            , 2       , True         , False        , (  0,  0,255) , 16711680),
    Label(  'Roundabout'           , 12 ,       11 , 'land'            , 2       , True         , False        , (  0,191,127) , 8371968),
    Label(  'Soccer_ball_field'    , 13 ,       12 , 'land'            , 2       , True         , False        , (  0,127,191) , 12549888),
    Label(  'plane'                , 14 ,       13 , 'transport'       , 1       , True         , False        , (  0,127,255) , 16744192),
    Label(  'Harbor'               , 15 ,       14 , 'transport'       , 1       , True         , False        , (  0,100,155) , 10183680),


]
m2label        = { label.m_color : label for label in labels           }
id2label        = { label.id      : label for label in labels           }


def collect_files(img_dir, gt_dir):
    suffix = 'leftImg8bit.png'
    files = []
    img_files = glob.glob(osp.join(img_dir, 'images/*[0-9].png'))
    for img_file in img_files:
        # assert img_file.endswith(suffix), img_file
        inst_file = gt_dir + '/images/' + os.path.basename(img_file).replace('.png', '') + '_instance_id_RGB.png'
        # Note that labelIds are not converted to trainId for seg map
        segm_file = gt_dir + '/images/' + os.path.basename(img_file).replace('.png', '') + '_instance_color_RGB.png'
        files.append((img_file, inst_file, segm_file))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = track_parallel_progress(load_img_info, files, nproc=nproc)
    else:
        images = track_progress(load_img_info, files)

    return images


def load_img_info(files):
    img_file, inst_file, segm_file = files
    inst_img = mmcv.imread(inst_file, channel_order='rgb', backend='cv2')
    segm_img = mmcv.imread(segm_file, channel_order='rgb', backend='cv2')
    # ids < 24 are stuff labels (filtering them first is about 5% faster)
    # imgNp = imgNp[:, :, 0] + 256 * imgNp[:, :, 1] + 256 * 256 * imgNp[:, :, 2]
    # imgNp_seg = imgNp_seg[:, :, 0] + 256 * imgNp_seg[:, :, 1] + 256 * 256 * imgNp_seg[:, :, 2]
    inst_img = inst_img[:, :, 0] + 256 * inst_img[:, :, 1] + 256 * 256 * inst_img[:, :, 2]
    segm_img = segm_img[:, :, 0] + 256 * segm_img[:, :, 1] + 256 * 256 * segm_img[:, :, 2]
    # unique_inst_ids = np.unique(inst_img[inst_img >= 24])
    unique_inst_ids = np.unique(inst_img[inst_img > 1000])
    anno_info = []
    for inst_id in unique_inst_ids:
        # For non-crowd annotations, inst_id // 1000 is the label_id
        # Crowd annotations have <1000 instance ids
        cls_id = np.unique(segm_img[inst_img == inst_id])
        dcls_id = cls_id[0]
        # label_id = m2label[dcls_id]

        # label_id = inst_id // 1000 if inst_id >= 1000 else inst_id
        # label = CSLabels.id2label[label_id]
        label = m2label[dcls_id]

        if not label.hasInstances or label.ignoreInEval:
            continue

        category_id = label.id
        iscrowd = 0
        mask = np.asarray(inst_img == inst_id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]

        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)

        # for json encoding
        mask_rle['counts'] = mask_rle['counts'].decode()

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox.tolist(),
            area=area.tolist(),
            segmentation=mask_rle)
        anno_info.append(anno)
    video_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.join(video_name, osp.basename(img_file)),
        height=inst_img.shape[0],
        width=inst_img.shape[1],
        anno_info=anno_info,
        segm_file=osp.join(video_name, osp.basename(segm_file)))

    return img_info


def cvt_annotations(image_infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            cat = dict(id=label.id, name=label.name)
            out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to COCO format')
    parser.add_argument('--cityscapes_path', default='/Users/kyanchen/datasets/seg/iSAID_patches', help='cityscapes data path')
    parser.add_argument('--img-dir', default='', type=str)
    parser.add_argument('--gt-dir', default='', type=str)
    parser.add_argument('-o', '--out-dir', default='/Users/kyanchen/datasets/seg/iSAID_patches/annotations', help='output path')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mkdir_or_exist(out_dir)

    img_dir = osp.join(cityscapes_path, args.img_dir)
    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    set_name = dict(
        train='instancesonly_filtered_gtFine_train.json',
        val='instancesonly_filtered_gtFine_val.json',
        # test='instancesonly_filtered_gtFine_test.json'
    )

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with Timer(print_tmpl='It took {}s to convert Cityscapes annotation'):
            files = collect_files(
                osp.join(img_dir, split), osp.join(gt_dir, split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
