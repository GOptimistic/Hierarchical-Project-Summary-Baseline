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
import csv
from tqdm import tqdm
import json
from transformers import AutoTokenizer

csv.field_size_limit(sys.maxsize)


class MyDataset(Dataset):

    def __init__(self, data_path, max_token_length=512, max_summary_length=48, pretrained_tokenizer=None, pad_id=1):
        super(MyDataset, self).__init__()

        self.max_token_length = max_token_length
        self.max_summary_length = max_summary_length
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.data_list = json.loads(text)
        if pretrained_tokenizer is None:
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained("codebert-base")
        else:
            self.pretrained_tokenizer = pretrained_tokenizer
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        method = self.data_list[index]
        code_str = method["code"]
        summary_tokens = method["summary_tokens"]
        code_tokens = self.pretrained_tokenizer.tokenize(code_str, max_length=512, truncation=True)
        summary_tokens = ' '.join(summary_tokens)
        summary_tokens = self.pretrained_tokenizer.tokenize(summary_tokens, max_length=512, truncation=True)
        code_tokens_id = self.pretrained_tokenizer.convert_tokens_to_ids(code_tokens)
        summary_tokens_id = self.pretrained_tokenizer.convert_tokens_to_ids(summary_tokens)
        method_valid_len = len(code_tokens_id)
        summary_valid_len = len(summary_tokens_id)

        if method_valid_len > self.max_token_length:
            method_valid_len = self.max_token_length
            code_tokens_id = code_tokens_id[:self.max_token_length]
        else:
            padding_tokens = [self.pad_id for _ in
                              range(self.max_token_length - method_valid_len)]
            code_tokens_id.extend(padding_tokens)

        if summary_valid_len > self.max_summary_length:
            summary_valid_len = self.max_summary_length
            summary_tokens_id = summary_tokens_id[:self.max_summary_length]
        else:
            padding_tokens = [self.pad_id for _ in
                              range(self.max_summary_length - summary_valid_len)]
            summary_tokens_id.extend(padding_tokens)

        return np.array(code_tokens_id), np.array(method_valid_len), np.array(summary_tokens_id), np.array(summary_valid_len)


if __name__ == '__main__':
    pretrained_tokenizer = AutoTokenizer.from_pretrained(
        "/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/src/pretrained/codebert-base")
    pad_id = pretrained_tokenizer.convert_tokens_to_ids(pretrained_tokenizer.pad_token)
    test = MyDataset(data_path="/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/function_summary_data/spring-boot_train.json",
                     pretrained_tokenizer=pretrained_tokenizer,
                     pad_id=pad_id)
    print(test.__getitem__(index=1)[0].shape)

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
    # with open("/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/function_summary_data/spring-boot_valid.json", 'r', encoding='utf-8') as f:
    #     text = f.read()
    # data = json.loads(text)
    # print(len(data))
    # pretrained_tokenizer = AutoTokenizer.from_pretrained("/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/src/pretrained/codebert-base")
    # pad_id = pretrained_tokenizer.convert_tokens_to_ids(pretrained_tokenizer.pad_token)
    # data = data
    # code_token_maxlen = 0
    # summary_token_maxlen = 0
    # for method in data:
    #     code_str = method["code"]
    #     summary_tokens = method["summary_tokens"]
    #     code_tokens = pretrained_tokenizer.tokenize(code_str, max_length=512, truncation=True)
    #     summary_tokens = ' '.join(summary_tokens)
    #     summary_tokens = pretrained_tokenizer.tokenize(summary_tokens, max_length=512, truncation=True)
    #     code_tokens_id = pretrained_tokenizer.convert_tokens_to_ids(code_tokens)
    #     summary_tokens_id = pretrained_tokenizer.convert_tokens_to_ids(summary_tokens)
    #     code_token_maxlen = max(code_token_maxlen, len(code_tokens_id))
    #     summary_token_maxlen = max(summary_token_maxlen, len(summary_tokens_id))
    # print(code_token_maxlen)
    # print(summary_token_maxlen)

    # 分割数据
    # split_index = int(len(data) * 0.8)
    # train_data = data[:split_index]
    # valid_data = data[split_index:]
    # print(len(train_data))
    # print(len(valid_data))
    # train_path = "/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/function_summary_data/spring-boot_train.json"
    # valid_path = "/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/function_summary_data/spring-boot_valid.json"
    # with open(train_path, 'w', encoding='utf-8') as train_file, open(valid_path, 'w', encoding='utf-8') as valid_file:
    #     json.dump(train_data, train_file)
    #     json.dump(valid_data, valid_file)

