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

from tokenizer import MyTokenizer

csv.field_size_limit(sys.maxsize)


class MyDatasetFlat(Dataset):

    def __init__(self, csv_data_path, max_token_length, max_summary_length, tokenizer):
        super(MyDatasetFlat, self).__init__()

        self.max_token_length = max_token_length
        self.max_summary_length = max_summary_length
        self.tokenizer = tokenizer
        self.sos_id = self.tokenizer.sos_index
        self.eos_id = self.tokenizer.eos_index
        self.pad_id = self.tokenizer.pad_index

        self.data_list = []
        with open(csv_data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                flat_input = row[2]
                repo_summary = row[1]
                self.data_list.append((flat_input, repo_summary))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        flat_input, repo_summary = self.data_list[index]

        flat_input = self.tokenizer.encode(flat_input, self.max_token_length)
        repo_summary = self.tokenizer.encode(repo_summary, self.max_summary_length)

        return np.array(flat_input), np.array(repo_summary)


if __name__ == '__main__':
    tokenizer = MyTokenizer('/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/w2v_vocab_flat.json')
    test = MyDatasetFlat('/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/src/flat_input_data/mini_test_flat.csv', 500, 100, tokenizer)
    print(test.__getitem__(index=1))

    training_params = {"batch_size": 64,
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
