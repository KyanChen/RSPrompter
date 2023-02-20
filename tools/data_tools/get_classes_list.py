import glob
import os

import numpy as np
import pickle
import sys
import tqdm
import shutil

pre_path = r'H:\DataSet\SceneCls\UCMerced_LandUse\UCMerced_LandUse\Images'
sub_folder_list = glob.glob(pre_path +'/*')

with open(pre_path+f'/../class_names.txt', 'w') as f:
    for sub_folder in sub_folder_list:
        sub_folder_name = os.path.basename(sub_folder)
        f.write(sub_folder_name+'\n')

