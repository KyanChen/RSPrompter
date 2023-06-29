import glob
import os
import random
import shutil

folder = '/Users/kyanchen/datasets/seg/semantic_drone_dataset/training_set/images'
save_folder = '/Users/kyanchen/datasets/seg/semantic_drone_dataset/training_set/'
img_files = glob.glob(folder+'/*.jpg')
random.shuffle(img_files)
random.shuffle(img_files)
train_len = int(len(img_files) * 0.8)
train_list = img_files[:train_len]
val_list = img_files[train_len:]
for phase in ['train', 'val']:
    to_folder = save_folder+'/'+phase
    os.makedirs(to_folder)
    for item in eval(phase+'_list'):
        shutil.copy(item, to_folder+'/'+os.path.basename(item))
        label_file = '/Users/kyanchen/datasets/seg/semantic_drone_dataset/training_set/gt/semantic/label_images/' + os.path.basename(item).replace('jpg', 'png')
        shutil.copy(label_file, to_folder+'/'+os.path.basename(label_file))
