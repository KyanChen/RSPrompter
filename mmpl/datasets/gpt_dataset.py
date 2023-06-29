import mmcv
import mmengine.dist as dist
import torch
import mmengine
from mmpl.registry import DATASETS
from mmengine.dataset import BaseDataset as _BaseDataset
import os
import requests
import numpy as np


@DATASETS.register_module()
class GPTDataset(_BaseDataset):
    def __init__(self,
                 tokenizer: str = 'gpt2',
                 block_size: int = 1024,
                 sample_times: int = 1280,
                 test_mode: bool = False,
                 data_root: str = '../data/tinyshakespeare/',
                 download: bool = True,
                 **kwargs):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.download = download
        self.file_path = data_root + '/input.txt'
        self.sample_times = sample_times
        self.data_len = 0
        mmengine.mkdir_or_exist(data_root)
        super().__init__(
            ann_file='',
            data_root=data_root,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        if dist.is_main_process() and not self._check_exists():
            if self.download:
                data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
                with open(self.file_path, 'w') as f:
                    f.write(requests.get(data_url).text)
            else:
                raise RuntimeError
        dist.barrier()

        with open(self.file_path, 'r') as f:
            data = f.read()
        n = len(data)
        train_data = data[:int(n * 0.9)]
        val_data = data[int(n * 0.9):]
        if not self.test_mode:
            data = train_data
        else:
            data = val_data
        # tokenize
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer)
        data = tokenizer.tokenize(data)
        data = tokenizer.convert_tokens_to_ids(data)
        self.data_len = len(data)
        return [{'data': data}]

    def _check_exists(self):
        if os.path.exists(self.file_path):
            return True
        return False

    def __len__(self):
        return self.sample_times

    def __getitem__(self, idx):
        data_info = self.get_data_info(0)
        all_tokens = np.array(data_info['data'])
        index = torch.randint(self.data_len - self.block_size, [])
        x = torch.from_numpy((all_tokens[index:index + self.block_size]).astype(np.int64))
        gt_label = torch.from_numpy((all_tokens[index + 1:index + 1 + self.block_size]).astype(np.int64))
        results = dict(
            x=x,
            gt_label=gt_label
        )
        return results



