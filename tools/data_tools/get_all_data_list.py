import glob
import os

import numpy as np
import pickle
import sys
import tqdm
import shutil
from skimage import io

pre_path = r'H:\DataSet\SceneCls\UCMerced_LandUse\UCMerced_LandUse\Images'
sub_folder_list = glob.glob(pre_path +'/*')
all_data_list = []
for sub_folder in sub_folder_list:
    img_list = glob.glob(sub_folder+'/*')
    all_data_list += img_list

with open(pre_path+f'/../all_img_list.txt', 'w') as f:
    for file in tqdm.tqdm(all_data_list):
        img = io.imread(file, as_gray=True)
        if 0 < img.shape[0]:
            file_name = os.path.basename(os.path.dirname(file)) + '/' + os.path.basename(file)
            gt_label = os.path.basename(os.path.dirname(file))
            f.write(file_name+' '+gt_label+'\n')

