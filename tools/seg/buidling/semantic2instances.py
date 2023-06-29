import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.mask import encode, decode
from skimage.measure import label, regionprops

seg_map = np.zeros((100, 100), dtype=np.uint8)
seg_map[10:20, 20:50] = 1
seg_map[50:90, 65:90] = 1
seg_map[55:70, 75:80] = 0
# plt.imshow(seg_map)
# plt.show()



all_instances = []
num_labels, instances, stats, centroids = cv2.connectedComponentsWithStats(seg_map, connectivity=8)
for idx_label in range(1, num_labels):
    all_instances.append(instances == idx_label)
if len(all_instances) > 0:
    all_instances = np.vstack(all_instances)

# instances = label(seg_map)
# num_labels, instances, stats, centroids = cv2.connectedComponentsWithStats(seg_map, connectivity=8)
# contours, h = cv2.findContours(seg_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 在二值化图像上搜索轮廓

# instances = []
# masks = decode(contours)
# for i in range(len(contours)):
#     draw_img = np.zeros(seg_map.shape, dtype=np.uint8)
#     cv2.drawContours(draw_img, contours, i, 1, 1)
#     instances.append(draw_img)

# instances = np.vstack([seg_map]+instances)
plt.imshow(all_instances)
plt.show()