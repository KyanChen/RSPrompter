import glob

import mmcv
import mmengine.dist as dist
import torch
import mmengine
from mmpl.registry import DATASETS
from mmengine.dataset import BaseDataset as _BaseDataset
import os
import requests
import numpy as np
from .data_utils import load_bvh_file, lafan1_utils_np


@DATASETS.register_module()
class BvhDataset(_BaseDataset):
    def __init__(self,
                 block_size: int = 1024,
                 data_root: str = '../data/lafan1/',
                 test_mode: bool = False,
                 n_offset=20,
                 phase='train',
                 pipeline=[],
                 **kwargs):
        self.test_mode = test_mode
        self.block_size = block_size
        self.n_offset = n_offset
        self.phase = phase
        if self.test_mode or phase in ['predict', 'val']:
            self.actors = ['subject5']
        else:
            self.actors = ['subject1', 'subject2', 'subject3', 'subject4']
        super().__init__(
            ann_file='',
            data_root=data_root,
            test_mode=test_mode,
            pipeline=pipeline,
            **kwargs)

    def load_data_list(self):
        bvh_files = []
        # load bvh files that match given actors
        for f in glob.glob(os.path.join(self.data_root, "*.bvh")):
            file_name = os.path.basename(f).rsplit(".", 1)[0]
            seq_name, actor = file_name.split("_")[:2]
            if actor in self.actors:
                bvh_files.append(f)

        if not bvh_files:
            raise FileNotFoundError(
                "No bvh files found in {}. (Actors: {})".format(
                    self.data_root, ", ".join(self.actors))
            )

        data_list = []
        bvh_files.sort()
        for bvh_path in bvh_files:
            data_info = {}
            print("Processing file {}".format(bvh_path))
            anim = load_bvh_file(bvh_path)

            # global joint rotation, position
            gr, gp = lafan1_utils_np.fk(anim.rotations, anim.positions, anim.parents)
            # left, right foot contact
            cl, cr = lafan1_utils_np.extract_feet_contacts(gp, [3, 4], [7, 8], vel_threshold=0.2)
            data_info['positions'] = anim.positions
            data_info['rotations'] = anim.rotations
            data_info['global_positions'] = gp
            data_info['global_rotations'] = gr
            data_info['foot_contact'] = np.concatenate([cl, cr], axis=-1)
            data_info['frames'] = anim.positions.shape[0]
            data_info['parents'] = anim.parents
            data_info['bvh_file'] = os.path.basename(bvh_path)
            data_list.append(data_info)

        self.idx2seq = {}
        count = 0
        for idx, data in enumerate(data_list):
            num_frame = data['frames']
            num_train_frames = num_frame - self.block_size
            assert num_train_frames > 0, "num_train_frames should be positive"
            for i in range(num_train_frames):
                if i % self.n_offset == 0:
                    self.idx2seq[count] = (idx, i)
                    count += 1
        self.num_samples = count
        return data_list

    def __len__(self):
        # return int(self.num_samples // 100)
        return self.num_samples
        # return 4

    def get_train_item(self, idx):
        # idx = idx * 100
        seq_idx, frame_idx = self.idx2seq[idx]
        data_info = self.get_data_info(seq_idx)
        positions = torch.from_numpy(data_info['positions'].astype(np.float32))
        rotations = torch.from_numpy(data_info['rotations'].astype(np.float32))
        global_positions = torch.from_numpy(data_info['global_positions'].astype(np.float32))
        global_rotations = torch.from_numpy(data_info['global_rotations'].astype(np.float32))
        foot_contact = torch.from_numpy(data_info['foot_contact'].astype(np.float32))
        parents = data_info['parents']
        bvh_file = data_info['bvh_file']
        x = dict(
            positions=positions[frame_idx:frame_idx + self.block_size + 1],
            rotations=rotations[frame_idx:frame_idx + self.block_size + 1],
            global_positions=global_positions[frame_idx:frame_idx + self.block_size + 1],
            global_rotations=global_rotations[frame_idx:frame_idx + self.block_size + 1],
            foot_contact=foot_contact[frame_idx:frame_idx + self.block_size + 1],
            parents=parents,
            bvh_file=bvh_file,
            frame_idx=frame_idx,
            seq_idx=seq_idx
        )
        return x

    def get_test_item(self, idx):
        seq_idx, frame_idx = self.idx2seq[idx]
        data_info = self.get_data_info(seq_idx)
        positions = torch.from_numpy(data_info['positions'].astype(np.float32))
        rotations = torch.from_numpy(data_info['rotations'].astype(np.float32))
        global_positions = torch.from_numpy(data_info['global_positions'].astype(np.float32))
        global_rotations = torch.from_numpy(data_info['global_rotations'].astype(np.float32))
        foot_contact = torch.from_numpy(data_info['foot_contact'].astype(np.float32))
        parents = data_info['parents']
        bvh_file = data_info['bvh_file']
        x = dict(
            positions=positions[frame_idx:frame_idx + self.block_size],
            rotations=rotations[frame_idx:frame_idx + self.block_size],
            global_positions=global_positions[frame_idx:frame_idx + self.block_size],
            global_rotations=global_rotations[frame_idx:frame_idx + self.block_size],
            foot_contact=foot_contact[frame_idx:frame_idx + self.block_size],
            parents=parents,
            bvh_file=bvh_file,
            frame_idx=frame_idx,
            seq_idx=seq_idx
        )
        return x

    def __getitem__(self, idx):
        if self.phase == 'predict':
            return self.get_test_item(idx)
        return self.get_train_item(idx)


