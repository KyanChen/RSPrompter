# -*- coding: utf-8 -*-

import json
import random

datasets_path = './'

f = open('{}annotations.json'.format(datasets_path) ,encoding='utf-8')
gt = json.load(f)

print(gt['info'])
print(gt['licenses'])

print(len(gt['categories']))
print(len(gt['images']))
print(len(gt['annotations']))

train = dict()

train['info'] = gt['info']
train['licenses'] = gt['licenses']
train['categories'] = gt['categories']
train['images'] = []
train['annotations'] = []


val = dict()

val['info'] = gt['info']
val['licenses'] = gt['licenses']
val['categories'] = gt['categories']
val['images'] = []
val['annotations'] = []

train_image_size = int(len(gt['images']) * 0.8)

print('train_img_num:{}'.format(train_image_size))
print('val_img_num:{}'.format(len(gt['images']) - train_image_size))


random.shuffle(gt['images'])

for img_info in gt['images']:

    if len(train['images']) < train_image_size:

        train['images'].append(img_info)


        for anno in gt['annotations']:
            if anno['image_id'] == img_info['id']:
                train['annotations'].append(anno)

    else:
        val['images'].append(img_info)

        for anno in gt['annotations']:
            if anno['image_id'] == img_info['id']:
                val['annotations'].append(anno)


with open("{}/instances_train2017.json".format(datasets_path), 'w', encoding='utf-8') as json_file:
    json.dump(train, json_file, ensure_ascii=False)

with open("{}/instances_val2017.json".format(datasets_path), 'w', encoding='utf-8') as json_file:
    json.dump(val, json_file, ensure_ascii=False)








