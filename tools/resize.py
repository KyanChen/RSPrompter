import os
from skimage import io
from tqdm import tqdm
import cv2
import glob
import mmcv

inp = r'H:\DataSet\SceneCls\UCMerced_LandUse\UCMerced_LandUse\Images'

for size in [256, 32]:
    print(size)
    filenames = glob.glob(inp+'/*/*')
    for filename in tqdm(filenames):
        img = io.imread(filename)
        h, w = img.shape[:2]
        if size != h:
            img = cv2.resize(img, (size, size), cv2.INTER_CUBIC)
        save_path = os.path.dirname(inp) + f'/{size}/' + os.path.basename(os.path.dirname(filename)) + '/' + os.path.basename(filename)
        mmcv.mkdir_or_exist(os.path.dirname(save_path))
        io.imsave(save_path, img)
