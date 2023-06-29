"""This script helps to convert txt-style dataset to the coco format.

Usage:
    $ python txt2coco.py /path/to/dataset # image_dir

Note:
    1. Before running this script, please make sure the root directory
    of your dataset is formatted in the following struction:
    .
    └── $ROOT_PATH
        ├── classes.txt
        ├── labels
        │    ├── a.txt
        │    ├── b.txt
        │    └── ...
        ├── images
        │    ├── a.jpg
        │    ├── b.png
        │    └── ...
        └── ...
    2. The script will automatically check whether the corresponding
    `train.txt`, ` val.txt`, and `test.txt` exist under your `image_dir`
    or not. If these files are detected, the script will organize the
    dataset. The image paths in these files must be ABSOLUTE paths.
    3. Once the script finishes, the result files will be saved in the
    directory named 'annotations' in the root directory of your dataset.
    The default output file is result.json. The root directory folder may
    look like this in the root directory after the converting:
    .
    └── $ROOT_PATH
        ├── annotations
        │    ├── result.json
        │    └── ...
        ├── classes.txt
        ├── labels
        │    ├── a.txt
        │    ├── b.txt
        │    └── ...
        ├── images
        │    ├── a.jpg
        │    ├── b.png
        │    └── ...
        └── ...
    4. After converting to coco, you can use the
    `tools/analysis_tools/browse_coco_json.py` script to visualize
    whether it is correct.
"""
import argparse
import glob
import os
import os.path as osp

import mmcv
import mmengine

IMG_EXTENSIONS = ('.jpg', '.png', '.jpeg', '.tiff')


def check_existence(file_path: str):
    """Check if target file is existed."""
    if not osp.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist!')


def get_image_info(idx, img_file_name):
    """Retrieve image information."""
    check_existence(img_file_name)

    img = mmcv.imread(img_file_name)
    height, width = img.shape[:2]
    # file_name = osp.basename(osp.dirname(img_file_name)) + '/' + os.path.basename(img_file_name)
    file_name = os.path.basename(img_file_name)
    img_info_dict = {
        'file_name': file_name,
        'id': idx,
        'width': width,
        'height': height
    }
    return img_info_dict, height, width


def convert_bbox_info(
        label, idx, obj_count,
        image_height=None, image_width=None,
        is_ann_absolute=True, is_ann_xyxy=True):
    """Convert txt-style bbox info to the coco format."""
    label = label.strip().split()
    x = float(label[1])
    y = float(label[2])
    w = float(label[3])
    h = float(label[4])
    if is_ann_xyxy:
        x1 = x
        y1 = y
        x2 = w
        y2 = h
    else:
        # convert x,y,w,h to x1,y1,x2,y2
        x1 = (x - w / 2)
        y1 = (y - h / 2)
        x2 = (x + w / 2)
        y2 = (y + h / 2)

    if not is_ann_absolute:
        x1 = (x - w / 2) * image_width
        y1 = (y - h / 2) * image_height
        x2 = (x + w / 2) * image_width
        y2 = (y + h / 2) * image_height

    cls_id = int(label[0])
    width = max(0., x2 - x1)
    height = max(0., y2 - y1)
    coco_format_info = {
        'image_id': idx,
        'id': obj_count,
        'category_id': cls_id,
        'bbox': [x1, y1, width, height],
        'area': width * height,
        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
        'iscrowd': 0
    }
    obj_count += 1
    return coco_format_info, obj_count


def convert_txt_to_coco(
        image_dir: str,
        idx2classname: dict,
        img_format: str='.tiff',
        json_name: str='result.json',
        output_folder: str='',
):
    """Convert annotations from txt style to coco style.

    Args:
        image_dir (str): the root directory of your datasets which contains
            labels, images, classes.txt, etc
    """
    print(f'Start to load existing images and annotations from {image_dir}')
    check_existence(image_dir)

    img_paths = glob.glob(image_dir+f'/*{img_format}')

    # prepare the output folders
    if not output_folder:
        output_folder = osp.join(image_dir, 'annotations')
    if not osp.exists(output_folder):
        os.makedirs(output_folder)
        check_existence(output_folder)

    dataset = {'images': [], 'annotations': [], 'categories': []}

    # category id starts from 0
    for idx, name in idx2classname.items():
        dataset['categories'].append({'id': idx, 'name': name})

    obj_count = 0
    skipped = 0
    converted = 0
    for idx, image_file in enumerate(mmengine.track_iter_progress(img_paths)):
        img_info_dict, image_height, image_width = get_image_info(idx, image_file)

        dataset['images'].append(img_info_dict)
        label_path = image_file.replace(f'{img_format}', '.txt')
        if not osp.exists(label_path):
            # if current image is not annotated or the annotation file failed
            print(
                f'WARNING: {label_path} does not exist. Please check the file.'
            )
            skipped += 1
            continue

        with open(label_path) as f:
            labels = f.readlines()
            for label in labels:
                coco_info, obj_count = convert_bbox_info(
                    label, idx, obj_count, image_height, image_width)
                dataset['annotations'].append(coco_info)
        converted += 1

    # saving results to result json

    out_file = osp.join(output_folder, json_name)
    print(f'Saving converted results to {out_file} ...')
    mmengine.dump(dataset, out_file, indent=4)

    # simple statistics
    print(f'Process finished! Please check at {output_folder} .')
    print(f'Number of images found: {len(img_paths)}, converted: {converted},',
          f'and skipped: {skipped}. Total annotation count: {obj_count}.')
    print('You can use tools/analysis_tools/browse_coco_json.py to visualize!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        default='/Users/kyanchen/codes/lightning_framework/samples/det/levir-ship2/sample_data',
        type=str,
        help='dataset directory with images and labels, etc.')
    arg = parser.parse_args()

    idx2classname = {0: 'ship'}
    img_format = '.tiff'
    json_name = 'sample_test.json'
    output_folder = '/Users/kyanchen/codes/lightning_framework/samples/det/levir-ship2/annotations'

    convert_txt_to_coco(arg.image_dir, idx2classname, img_format, json_name, output_folder)
