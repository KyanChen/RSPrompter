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
class MotionGPTDataset(_BaseDataset):
    def __init__(
            self,
            ann_file,
            data_root,
            token_dir,
            # dataset_name='kit',
            block_size=64,
            n_offset=1,
            # unit_length=4,
            # max_motion_length=196,
            # joints_num=21,
            test_mode=False,
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

        self.motion_dir = osp.join(self.data_root, 'new_joint_vecs')
        self.text_dir = osp.join(self.data_root, 'texts')
        self.token_dir = token_dir

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
                motion_token_list = np.load(osp.join(self.token_dir, name + '.npy'))
                data_list.append(
                    dict(motion_token=motion_token_list, name=name)
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
            num_train_frames = data['motion_token'].shape[-1] - self.block_size
            if num_train_frames > 0:
                for i in range(num_train_frames):
                    if i % self.n_offset == 0:
                        self.idx2seq[count] = (idx, i)
                        count += 1
            else:
                self.idx2seq[count] = (idx, 0)
                count += 1
        self.num_samples = count
        return data_list

    def __len__(self):
        if self.test_mode and self.predict_seq_len is not None:
            return self.predict_seq_len
        return self.num_samples

    def get_predict_item(self, idx):
        id_list = np.loadtxt(self.ann_file, dtype=str)
        name = random.choice(id_list)
        while True:
            try:
                motion = np.load(osp.join(self.motion_dir, name + '.npy')).astype(np.float32)
                if motion.shape[0] < 20*4:
                    name = random.choice(id_list)
                    continue
                break
            except Exception as e:
                # if get_rank() == 0:
                print(e)
                name = random.choice(id_list)

        motion = torch.from_numpy(motion)
        m_length = motion.shape[0]
        unit_length = 2 * 2
        m_length = (m_length // unit_length) * unit_length

        start_idx = random.randint(0, len(motion) - m_length)
        motion = motion[start_idx:start_idx + m_length]

        res = dict(
            motion=motion,
            name=name,
            frame_idx=0,
            seq_idx=0
        )
        return res

    def __getitem__(self, idx):
        if self.test_mode and self.predict_seq_len is not None:
            return self.get_predict_item(idx)

        seq_idx, frame_idx = self.idx2seq[idx]
        data_info = self.get_data_info(seq_idx)
        motion_token_list = data_info['motion_token']
        m_tokens = random.choice(motion_token_list)

        token_len = len(m_tokens)
        if token_len > self.block_size:
            m_tokens = m_tokens[frame_idx:frame_idx + self.block_size]

        if random.random() < 0.5:
            if random.random() < 0.5:
                m_tokens = m_tokens[:-1]
            if random.random() < 0.5:
                m_tokens = m_tokens[1:]

        seq_name = data_info['name']
        res = dict(
            motion_token=torch.from_numpy(m_tokens),
            name=seq_name,
            frame_idx=frame_idx,
            seq_idx=seq_idx,
        )
        return res

