"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import re
import sys
import time
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import json
from transformers import T5Tokenizer

from tokenizer import MyTokenizer

csv.field_size_limit(sys.maxsize)


class MyDatasetFileSummaryT5(Dataset):

    def __init__(self, csv_data_path, max_token_length, max_summary_length, tokenizer):
        super(MyDatasetFileSummaryT5, self).__init__()

        self.max_token_length = max_token_length
        self.max_summary_length = max_summary_length
        self.tokenizer = tokenizer

        self.data_list = []
        with open(csv_data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                repo_name = row[0].split('/')[1]
                file_summary = row[1]
                file_summary = 'Give a brief summary about the code project {} based the json {}'.format(repo_name, file_summary)
                # # 去掉非数字字母空格符号，比如标点
                # file_summary = re.sub(r'[^\w\s]', ' ', file_summary)
                # # 处理下划线
                # file_summary = file_summary.replace('_', ' ')
                # # 处理驼峰命名的词，切分成多个单词
                # file_summary = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', file_summary)
                repo_summary = row[2]
                self.data_list.append((file_summary, repo_summary))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_summary, repo_summary = self.data_list[index]
        source = self.tokenizer.batch_encode_plus([file_summary], max_length=self.max_token_length, truncation=True, pad_to_max_length=True,
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([repo_summary], max_length=self.max_summary_length, truncation=True, pad_to_max_length=True,
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


if __name__ == '__main__':
    t5_path = '/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/src/pretrained/google-flan-t5-large'
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    test = MyDatasetFileSummaryT5('/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/src/file_summary_data/mini_test_one_level.csv', 500, 100, tokenizer)
    print(test.__getitem__(index=1))

    training_params = {"batch_size": 128,
                       "shuffle": True,
                       "drop_last": True}
    training_generator = DataLoader(test, **training_params)
    num = 0
    start_time = time.time()
    for object in tqdm(training_generator):
        print(object['source_ids'].shape)
        print(object['source_mask'].shape)
        print(object['target_ids'].shape)
        print(object['target_ids_y'].shape)
        num = num + 1
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time is {total_time}s, total {num} batches, {total_time / num}s per batch')
