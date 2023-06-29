import glob
import os.path

import cv2

phase = 'val'
folder = '/Users/kyanchen/datasets/seg/semantic_drone_dataset/training_set/'+phase
to_folder = '/Users/kyanchen/datasets/seg/semantic_drone_dataset/training_set/'+phase+'_slice'
os.makedirs(to_folder)
img_files = glob.glob(folder+'/*.jpg')
slice_size = [1024, 1024]
for img_file in img_files:
    img = cv2.imread(img_file)
    label = cv2.imread(img_file.replace('.jpg', '.png'))
    h, w, c = img.shape
    h_list = list(range(0, h, slice_size[1]))
    w_list = list(range(0, w, slice_size[0]))
    h_list[-1] = h - slice_size[1]
    w_list[-1] = w - slice_size[0]
    for i_h in h_list:
        for i_w in w_list:
            img_save = img[i_h:i_h+slice_size[1], i_w:i_w+slice_size[0], :]
            label_save = label[i_h:i_h+slice_size[1], i_w:i_w+slice_size[0], :]
            cv2.imwrite(to_folder+'/'+os.path.basename(img_file).split('.')[0]+f'_{i_w}_{i_h}.jpg', img_save)
            cv2.imwrite(to_folder + '/' + os.path.basename(img_file).split('.')[0] + f'_{i_w}_{i_h}.png', label_save)
