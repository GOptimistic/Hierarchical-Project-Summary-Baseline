"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import h5py
import os.path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import json
from transformers import AutoTokenizer


class MyDataset(Dataset):

    def __init__(self, data_dir_path, data_path_prefix, part_size, total_length, max_token_length):
        super(MyDataset, self).__init__()

        self.data_dir_path = data_dir_path
        self.data_path_prefix = data_path_prefix
        self.part_size = part_size
        self.total_length = total_length
        self.max_token_length = max_token_length

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        # 根据index计算所在的part
        part_num = index // self.part_size
        part_index = index % self.part_size
        hdf5_file_path = self.data_dir_path + os.sep + self.data_path_prefix + str(part_num) + '.hdf5'
        with h5py.File(hdf5_file_path, 'r') as f:
            repo_info = f.get("repo_info")[part_index:part_index + 1]
            repo_valid_len = f.get("repo_valid_len")[part_index:part_index + 1]
            summary = f.get("summary")[part_index:part_index + 1]
            summary_valid_len = f.get("summary_valid_len")[part_index:part_index + 1]
        if repo_info.shape[4] > self.max_token_length:
            repo_info = repo_info[..., 0:self.max_token_length]
            repo_valid_len = np.where(repo_valid_len < self.max_token_length, repo_valid_len, self.max_token_length)
        return np.squeeze(repo_info, 0), np.squeeze(repo_valid_len, 0), np.squeeze(summary, 0), np.squeeze(summary_valid_len, 0)


if __name__ == '__main__':
    test = MyDataset(data_dir_path="./hdf5_no_compress_data", data_path_prefix="train_java_",
                     part_size=3200, total_length=24357)
    # print(test.__getitem__(index=1)[0].shape)

    training_params = {"batch_size": 16,
                       "shuffle": True,
                       "drop_last": True}
    training_generator = DataLoader(test, **training_params)
    num = 0
    start_time = time.time()
    for repo_info, repo_valid_len, summary, summary_valid_len in tqdm(training_generator):
        print(repo_info.shape)
        print(repo_valid_len.shape)
        print(summary.shape)
        print(summary_valid_len.shape)
        num = num + 1
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time is {total_time}s, total {num} batches, {total_time / num}s per batch')
