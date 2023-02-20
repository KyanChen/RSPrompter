import glob
import json
import os

import numpy as np
import pickle
import sys
import tqdm
import shutil
from skimage import io

pre_path = '/Users/kyanchen/Documents/AID/AID'
sub_folder_list = glob.glob(pre_path +'/*')
# train_val_frac = [0.6, 0.2]
train_val_frac = [0.8, 0.2]

train_list = []
val_list = []
test_list = []
for sub_folder in sub_folder_list:
    img_list = glob.glob(sub_folder+'/*')
    # img_list = [x for x in img_list if 0 < io.imread(x).shape[0] < 60]
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)
    # img_list = img_list

    # for UC datasets
    # num_train_samps = int(len(img_list) * train_val_frac[0])
    # num_val_samps = int(len(img_list) * train_val_frac[1])
    # train_list += img_list[:num_train_samps]
    # val_list += img_list[num_train_samps:num_train_samps+num_val_samps]
    # test_list += img_list[num_train_samps+num_val_samps:]

    # for AID datasets
    num_train_samps = int(len(img_list) * train_val_frac[0]) - 10
    num_val_samps = 10

    train_list += img_list[:num_train_samps]
    val_list += img_list[num_train_samps:num_train_samps + num_val_samps]
    test_list += img_list[num_train_samps + num_val_samps:]

data = {}
folder = pre_path + f'/..'
os.makedirs(folder, exist_ok=True)
for phase in ['train_list', 'val_list', 'test_list']:
    data[phase.split('_')[0]] = [os.path.basename(os.path.dirname(file)) + '/' + os.path.basename(file) for file in eval(phase)]

json.dump(data, open(folder+'/AID_split.json', 'w'))