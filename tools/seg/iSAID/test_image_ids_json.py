#!/usr/bin/env python

# modified by @akshitac8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
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

import cityscapesScripts.cityscapesscripts.evaluation.instances2dict_with_polygons as cs
import detectron.utils.segms as segms_util
import detectron.utils.boxes as bboxs_util
from cityscapesScripts.cityscapesscripts.helpers.labels import *

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--outdir', help="path of output directory for json files", default='./dataset/iSAID_patches', type=str)
    parser.add_argument('--datadir', help="root path of dataset (patches)",default='./dataset/iSAID_patches', type=str)
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
    #NOTE : The following mentioned category order must be mentained during validation and testing time
    category_dict = [{'id': 1, 'name': 'ship'},{'id': 2, 'name': 'storage_tank'},{'id': 3, 'name': 'baseball_diamond'},{'id': 4, 'name': 'tennis_court'},{'id': 5, 'name': 'basketball_court'},{'id': 6, 'name': 'Ground_Track_Field'},{'id': 7, 'name': 'Bridge'},{'id': 8, 'name': 'Large_Vehicle'},{'id': 9, 'name': 'Small_Vehicle'},{'id': 10, 'name': 'Helicopter'},{'id': 11, 'name': 'Swimming_pool'},{'id': 12, 'name': 'Roundabout'},{'id': 13, 'name': 'Soccer_ball_field'},{'id': 14, 'name': 'plane'},{'id': 15, 'name': 'Harbor'}]
    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)
        print(ann_dir)
        img_id = 0
        c_images = 0
        for root, _, files in os.walk(ann_dir):
            for filename in natsorted(files):
                if re.match(r'\w*\d+.png', filename) or filename.split('.')[0].count('_')==4:
                    c_images+=1
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
                    images.append(image)

        ann_dict['images'] = images
        import pdb;pdb.set_trace()
        #categories = [{"id": category_dict[name], "name": name} for name in category_dict]
        categories = category_dict
        ann_dict['categories'] = categories
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        with open(os.path.join(os.path.join(out_dir,data_set), json_name % data_set), "w") as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    convert_cityscapes_instance_only(args.datadir, args.outdir)
