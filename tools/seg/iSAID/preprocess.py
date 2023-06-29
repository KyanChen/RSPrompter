# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
import tqdm
from natsort import natsorted
import re
import argparse
import h5py
import json
import os
import scipy.misc
import sys
import cv2
import sys
import random
random.seed(0)
# import cityscapesscripts
# from instances2dict_with_polygons import instances2dict_with_polygons
from cityscapesscripts.evaluation.instances2dict_with_polygons import instances2dict_with_polygons
import detectron_utils.segms as segms_util
import detectron_utils.boxes as bboxs_util
from cityscapesscripts.helpers.labels import *

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--outdir', help="path of output directory for json files", default='/Users/kyanchen/datasets/seg/iSAID_patches', type=str)
    parser.add_argument('--datadir', help="root path of dataset (patches)",default='/Users/kyanchen/datasets/seg/iSAID_patches', type=str)
    parser.add_argument('--set', default="train,val", type=str, help='evaluation mode')
    return parser.parse_args()


# for Cityscapes
def getLabelID(self, instID):
    if (instID < 1000):
        return instID
    else:
        return int(instID / 1000)


def convert_cityscapes_instance_only(data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = args.set.split(',')
    for i in sets:
        if i == 'train' or i == 'val':
            ann_dirs = ['train/images','val/images']
        elif i == 'test': # NEED TEST MASK ANNOTATIONS
            ann_dirs = ['test/images']
        else:
            print('Invalid input')

    json_name = 'instancesonly_filtered_%s.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        'unlabeled',
        'ship',
        'storage_tank',
        'baseball_diamond',
        'tennis_court',
        'basketball_court',
        'Ground_Track_Field',
        'Bridge',
        'Large_Vehicle',
        'Small_Vehicle',
        'Helicopter',
        'Swimming_pool',
        'Roundabout',
        'Soccer_ball_field',
        'plane',
        'Harbor'
    ]
    for data_set, ann_dir in tqdm.tqdm(zip(sets, ann_dirs)):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)
        print(ann_dir)
        img_id = 0  # for every image_id with different indexing
        c_images = 0
        for root, _, files in os.walk(ann_dir):
            for filename in natsorted(files):
                if filename.endswith('_color_RGB.png'): #if re.match(r'\w*\d+.png', filename) or filename.split('.')[0].count('_')==4:
                    #import pdb;pdb.set_trace()
                    c_images+=1
                    filename = ''.join(filename)
                    filename = filename.split('_')[:-3]
                    if len(filename) > 1:
                        filename = '_'.join(filename)
                    else:
                        filename = ''.join(filename)
                    filename = filename + '.png'
                    print("Processed %s images" % (c_images))
                    image_dim = cv2.imread(os.path.join(root,filename))
                    imgHeight,imgWidth,_  = image_dim.shape
                    image = {}
                    image['id'] = img_id
                    img_id += 1
                    image['width'] = imgWidth
                    image['height'] = imgHeight
                    print("Processing Image",filename)
                    image['file_name'] = filename.split('.')[0] + '.png'
                    print("Processing Image",image['file_name'])
                    image['ins_file_name'] = filename.split('.')[0] + '_instance_id_RGB.png'
                    image['seg_file_name'] = filename.split('.')[0] + '_instance_color_RGB.png'
                    images.append(image)

                    #import pdb;pdb.set_trace()
                    seg_fullname = os.path.join(root, image['seg_file_name'])
                    inst_fullname = os.path.join(root, image['ins_file_name'])

                    if not os.path.exists(seg_fullname):
                        print("YOU DONT HAVE TEST MASKS")
                        sys.exit(0)
                    objects = instances2dict_with_polygons([seg_fullname],[inst_fullname], verbose=True)
                    for k,v in objects.items():
                        for object_cls in list(v.keys()):
                            if object_cls not in category_instancesonly: #to get the labels only mentioned in category_instancesonly
                                print('Warning: Ignoring {} instances...'.format(object_cls))
                                continue
                            for obj in v[object_cls]:
                                if obj['contours'] == []:
                                    print('Warning: empty contours.')
                                    continue  
                                len_p = [len(p) for p in obj['contours']]
                                if min(len_p) <= 4:
                                    print('Warning: invalid contours.')
                                    continue 

                                ann = {}
                                ann['id'] = ann_id
                                ann_id += 1
                                ann['image_id'] = image['id']
                                ann['segmentation'] = obj['contours']
                                if object_cls not in category_dict:
                                    category_dict[object_cls] = label2id[object_cls]

                                ann['category_id'] = category_dict[object_cls]
                                ann['category_name'] = object_cls
                                ann['iscrowd'] = 0
                                ann['area'] = obj['pixelCount']
                                ann['bbox'] = bboxs_util.xyxy_to_xywh(segms_util.polys_to_boxes([ann['segmentation']])).tolist()[0]

                                #annotations.append(ann)
                                if ann['area'] > 10:
                                    annotations.append(ann)

        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(os.path.join(out_dir,data_set), json_name % data_set), "w") as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    convert_cityscapes_instance_only(args.datadir, args.outdir)
