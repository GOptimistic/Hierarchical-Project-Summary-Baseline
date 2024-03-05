"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""

import sys
import time
import csv
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import json
from transformers import AutoTokenizer

csv.field_size_limit(sys.maxsize)


class MyDataset(Dataset):

    def __init__(self, csv_data_path, max_package_length, max_file_length, max_token_length, max_summary_length,
                 tokenizer):
        super(MyDataset, self).__init__()

        self.max_package_length = max_package_length
        self.max_file_length = max_file_length
        self.max_token_length = max_token_length
        self.max_summary_length = max_summary_length
        self.tokenizer = tokenizer
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        self.pad_id = self.tokenizer.pad_id

        self.data_list = []
        with open(csv_data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                repo_name = row[0].split('/')[1]
                file_summaries_object = json.loads(row[1])
                repo_summary = row[2]
                self.data_list.append((repo_name, file_summaries_object, repo_summary))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        repo_name, file_summaries_object, repo_summary = self.data_list[index]
        file_summaries_list = []
        for package in file_summaries_object.keys():
            package_info = file_summaries_object[package]
            package_file_list = []
            for file in package_info.keys():
                file_summary = package_info[file]
                # 加上package和file的信息
                file_summary = 'Repo {}. Package {}. File {}. '.format(repo_name, package, file) + file_summary
                # pad应补充在eos的后面
                encode_ids = self.tokenizer.encode(file_summary, True, True)
                if len(encode_ids) > self.max_token_length:
                    encode_ids = encode_ids[:self.max_token_length - 1]
                    encode_ids = encode_ids + [self.eos_id]
                else:
                    extended_words = [self.pad_id for _ in range(self.max_token_length - len(encode_ids))]
                    encode_ids.extend(extended_words)
                package_file_list.append(encode_ids)
            if len(package_file_list) > self.max_file_length:
                package_file_list = package_file_list[:self.max_file_length]
            else:
                extended_files = [[self.pad_id for _ in range(self.max_token_length)] for _ in
                                  range(self.max_file_length - len(package_file_list))]
                package_file_list.extend(extended_files)
            file_summaries_list.append(package_file_list)
        if len(file_summaries_list) > self.max_package_length:
            file_summaries_list = file_summaries_list[:self.max_package_length]
        else:
            extended_packages = [[[self.pad_id for _ in range(self.max_token_length)] for _ in
                                  range(self.max_file_length)] for _ in
                                 range(self.max_package_length - len(file_summaries_list))]
            file_summaries_list.extend(extended_packages)

        # 处理summary
        repo_summary = self.tokenizer.encode(repo_summary, True, True)
        if len(repo_summary) > self.max_summary_length:
            repo_summary = repo_summary[:self.max_summary_length - 1]
            repo_summary = repo_summary + [self.eos_id]
        else:
            extended_words = [self.pad_id for _ in range(self.max_summary_length - len(repo_summary))]
            repo_summary.extend(extended_words)

        return np.array(file_summaries_list), np.array(repo_summary)


if __name__ == '__main__':
    model_path = '/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/src/pretrained/chatglm3-6b-128k'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    test = MyDataset('/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/src/file_summary_data/combined_csv.csv', 5, 5, 100, 30, tokenizer.tokenizer)
    print(test.__getitem__(index=1))

    training_params = {"batch_size": 16,
                       "shuffle": True,
                       "drop_last": True}
    training_generator = DataLoader(test, **training_params)
    num = 0
    start_time = time.time()
    for file_summaries_list, repo_summary in tqdm(training_generator):
        print(file_summaries_list.shape)
        print(repo_summary.shape)
        num = num + 1
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time is {total_time}s, total {num} batches, {total_time / num}s per batch')
