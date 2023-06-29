import glob
import random
import mmcv
import mmengine.dist as dist
import torch
import mmengine
from mmpl.registry import DATASETS
from mmengine.dataset import BaseDataset as _BaseDataset
import numpy as np
import os.path as osp
from lightning.fabric.utilities import rank_zero_info
from torch.distributed import get_rank


@DATASETS.register_module()
class VQMotionDataset(_BaseDataset):
    def __init__(
            self,
            ann_file,
            data_root,
            # dataset_name='kit',
            block_size=64,
            n_offset=1,
            # unit_length=4,
            # max_motion_length=196,
            # joints_num=21,
            test_mode=False,
            cache_mode=False,
            predict_seq_len=None,
            pipeline=[],
            **kwargs
    ):
        self.block_size = block_size
        # self.unit_length = unit_length
        # self.dataset_name = dataset_name
        self.data_root = data_root
        # self.max_motion_length = max_motion_length
        # self.joints_num = joints_num
        self.test_mode = test_mode
        self.n_offset = n_offset
        self.predict_seq_len = predict_seq_len
        self.cache_mode = cache_mode

        self.motion_dir = osp.join(self.data_root, 'new_joint_vecs')
        self.text_dir = osp.join(self.data_root, 'texts')

        # if dataset_name == 't2m':
        #     assert self.joints_num == 22
        # elif dataset_name == 'kit':
        #     assert self.joints_num == 21
        # else:
        #     raise NotImplementedError

        # mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        # std = np.load(pjoin(self.meta_dir, 'std.npy'))
        # self.mean = mean
        # self.std = std

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            test_mode=test_mode,
            pipeline=pipeline,
            **kwargs)

    def load_data_list(self):
        data_list = []
        id_list = np.loadtxt(self.ann_file, dtype=str)
        tracker = mmengine.ProgressBar(len(id_list))
        for name in id_list:
            try:
                motion = np.load(osp.join(self.motion_dir, name + '.npy')).astype(np.float32)
                if motion.shape[0] < self.block_size:
                    # rank_zero_info("Motion {} is too short".format(name))
                    # if get_rank() == 0:
                    #     print("Motion {} is too short".format(name))
                    continue
                # self.lengths.append(motion.shape[0] - self.block_size)
                data_list.append(
                    dict(motion=motion, name=name)
                )
            except Exception as e:
                # if get_rank() == 0:
                print(e)
            tracker.update()

        # rank_zero_info("Total number of motion files {}".format(len(glob.glob(osp.join(self.motion_dir, '*.npy')))))
        # rank_zero_info("Total number of motions txt ids {}".format(len(id_list)))
        # rank_zero_info("Total number of motions final {}".format(len(data_list)))
        # if get_rank() == 0:
        print("Total number of motion files {}".format(len(glob.glob(osp.join(self.motion_dir, '*.npy')))))
        print("Total number of motions txt ids {}".format(len(id_list)))
        print("Total number of motions final {}".format(len(data_list)))

        self.idx2seq = {}
        count = 0
        for idx, data in enumerate(data_list):
            num_train_frames = len(data['motion']) - self.block_size
            assert num_train_frames >= 0, "num_train_frames should be positive"
            for i in range(num_train_frames):
                if i % self.n_offset == 0:
                    self.idx2seq[count] = (idx, i)
                    count += 1
        self.num_samples = count
        return data_list

    def __len__(self):
        if self.cache_mode:
            return super().__len__()
        if self.test_mode and self.predict_seq_len is not None:
            return self.predict_seq_len
        return self.num_samples

    def get_cache_data(self, idx):
        data_info = self.get_data_info(idx)
        motion = torch.from_numpy(data_info['motion'])
        m_length = motion.shape[0]
        unit_length = 2 * 2
        m_length = (m_length // unit_length) * unit_length

        start_idx = random.randint(0, len(motion) - m_length)
        motion = motion[start_idx:start_idx + m_length]

        res = dict(
            motion=motion,
            name=data_info['name'],
        )
        return res

    def __getitem__(self, idx):
        if self.cache_mode:
            return self.get_cache_data(idx)

        seq_idx, frame_idx = self.idx2seq[idx]
        data_info = self.get_data_info(seq_idx)
        motion = torch.from_numpy(data_info['motion'][frame_idx:frame_idx + self.block_size])
        res = dict(
            motion=motion,
            name=data_info['name'],
            frame_idx=frame_idx,
            seq_idx=seq_idx,
        )
        return res

