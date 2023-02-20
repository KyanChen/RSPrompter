import glob
import os

import lmdb
import numpy as np
import pickle
import sys
import tqdm
import shutil

pre_path = r'H:\DataSet\SceneCls\UCMerced_LandUse\UCMerced_LandUse\Images'
file_list = glob.glob(pre_path+'/*/*')
dataset_name = 'UCMerced'
cache_keys = ['filename', 'gt_label']


lmdb_path = os.path.abspath(pre_path + f'/../{dataset_name}_lmdb')
# if os.path.exists(pre_path + f'/lmdb'):
#     shutil.rmtree(pre_path + f'/lmdb')
os.makedirs(lmdb_path, exist_ok=True)

data_size_per_item = sys.getsizeof(open(file_list[0], 'rb').read())
print(f'data size:{data_size_per_item}')


env = lmdb.open(lmdb_path+f'\\{os.path.basename(lmdb_path)}.lmdb', map_size=data_size_per_item * 1e5)
txn = env.begin(write=True)

commit_interval = 5
keys_list = []
for idx, file in enumerate(file_list):
    key = f'{dataset_name}_{os.path.basename(file).split(".")[0]}'
    keys_list.append(key)

    for cache_key in cache_keys:
        if cache_key == 'filename':
            value = os.path.basename(os.path.dirname(file)) + '/' + os.path.basename(file)
        elif cache_key == 'img':
            with open(file, 'rb') as f:
                # 读取图像文件的二进制格式数据
                value = f.read()
        elif  cache_key == 'gt_label':
            value = os.path.basename(os.path.dirname(file))
        cache_key = key + f'_{cache_key}'
        cache_key = cache_key.encode()

        if isinstance(value, bytes):
            txn.put(cache_key, value)
        else:
            # 标签类型为str, 转为bytes
            txn.put(cache_key, value.encode())  # 编码
    if idx % commit_interval == 1:
        txn.commit()
        txn = env.begin(write=True)
txn.commit()
env.close()
keys_list = np.array(keys_list)
np.savetxt(open(pre_path+'/../keys_list.txt', 'w'), keys_list, fmt='%s')
print(f'Finish writing!')

