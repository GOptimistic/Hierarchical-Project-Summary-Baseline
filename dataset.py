"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import json
import os.path
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from random import sample
from transformers import AutoTokenizer, AutoModel

csv.field_size_limit(sys.maxsize)


class MyDataset(Dataset):

    def __init__(self, data_path, repo_base_path, pretrained_model, max_length_package=30, max_length_file=10, max_length_method=10,
                 max_length_token=50, max_length_summary=50):
        super(MyDataset, self).__init__()

        repos, summarys = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                # 由于split_data到train.csv导致多了一列出来，本来是第1列和第9列
                repo_full_name = row[2]
                summary = row[10]
                summary = summary.replace("\n", "").replace("\r", "")
                repos.append(repo_full_name)
                summarys.append(summary)

        self.repos = repos
        self.summarys = summarys
        self.repo_base_path = repo_base_path
        self.max_length_package = max_length_package
        self.max_length_file = max_length_file
        self.max_length_method = max_length_method
        self.max_length_token = max_length_token
        self.max_length_summary = max_length_summary
        # 得到embedding放在模型里，数据集里只截取token
        # self.pretrained_model = AutoModel.from_pretrained(pretrained_model)
        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.pad_id = self.pretrained_tokenizer.convert_tokens_to_ids(self.pretrained_tokenizer.pad_token)

    def __len__(self):
        return len(self.summarys)

    def __getitem__(self, index):
        repo_full_name = self.repos[index]
        summary = self.summarys[index]
        # 对summary进行分词并得到id
        summary = self.pretrained_tokenizer.tokenize(summary, max_length=512, truncation=True)
        summary = self.pretrained_tokenizer.convert_tokens_to_ids(summary)
        if len(summary) < self.max_length_summary:
            extend_tokens = [self.pad_id for _ in
                             range(self.max_length_summary - len(summary))]
            summary.extend(extend_tokens)
        summary = summary[:self.max_length_summary]
        # 从repo的json文件中读出package file method token_ids组成一个四维列表
        repo_info = self.get_repo_info(repo_full_name)

        repo_info = np.stack(arrays=repo_info, axis=0)
        summary = np.array(summary)

        return repo_info.astype(np.int64), summary.astype(np.int64)

    # 读取json文件，解析出来package file method token_ids,得到一个四维列表
    def get_repo_info(self, full_name):
        json_path = self.repo_base_path + os.sep + full_name.replace('/', '_') + os.sep + 'repo_tree_info.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            text = f.read()
        project = json.loads(text)
        packages = project["packages"]
        repo_info = []
        for package in packages.keys():
            package_info = []
            files = packages[package]['files']
            for file in files.keys():
                file_info = []
                methods = files[file]['methods']
                # 对methods数量进行限制
                if len(methods) > self.max_length_method:
                    methods = sample(methods, self.max_length_method)
                for method in methods:
                    method = method.replace('\n', '').replace('\r', '').replace('\t', '')
                    # 对method进行分词
                    code_tokens = self.pretrained_tokenizer.tokenize(method, max_length=512, truncation=True)
                    # 如果不够token，用pad补足
                    if len(code_tokens) < self.max_length_token:
                        extend_tokens = [self.pretrained_tokenizer.pad_token for _ in
                                         range(self.max_length_token - len(code_tokens))]
                        code_tokens.extend(extend_tokens)
                    code_tokens = code_tokens[:self.max_length_token]
                    tokens_ids = self.pretrained_tokenizer.convert_tokens_to_ids(code_tokens)
                    # method_embeddings = self.pretrained_model(torch.tensor(tokens_ids)[None, :])[0][0]

                    file_info.append(tokens_ids)
                # method数量不够，补足
                if len(file_info) < self.max_length_method:
                    extend_methods = [[self.pad_id for _ in range(self.max_length_token)] for _ in
                                      range(self.max_length_method - len(file_info))]
                    file_info.extend(extend_methods)
                file_info = file_info[:self.max_length_method]
                package_info.append(file_info)
            if len(package_info) < self.max_length_file:
                extend_files = [[[self.pad_id for _ in range(self.max_length_token)] for _ in
                                 range(self.max_length_method)] for _ in
                                range(self.max_length_file - len(package_info))]
                package_info.extend(extend_files)
            package_info = package_info[:self.max_length_file]
            repo_info.append(package_info)
        if len(repo_info) < self.max_length_package:
            extend_packages = [[[[self.pad_id for _ in range(self.max_length_token)] for _ in
                                 range(self.max_length_method)] for _ in
                                range(self.max_length_file)] for _ in
                               range(self.max_length_package - len(repo_info))]
            repo_info.extend(extend_packages)
        repo_info = repo_info[:self.max_length_package]
        return repo_info


if __name__ == '__main__':
    # test = MyDataset(data_path="./data/data_python_output_100rows.csv", repo_base_path="./data/python",
    #                  pretrained_model="./pretrained/codebert-base")
    # # print(test.__getitem__(index=1)[1].shape)
    #
    # training_params = {"batch_size": 3,
    #                    "shuffle": True,
    #                    "drop_last": True}
    # training_generator = DataLoader(test, **training_params)
    # for iter, (feature, label) in enumerate(training_generator):
    #     print(feature.shape)
    #     print(label.shape)
    model = AutoModel.from_pretrained("./pretrained/codebert-base")
    tokenizer = AutoTokenizer.from_pretrained("./pretrained/codebert-base")
    configuration = model.config
    print(configuration.vocab_size)
    print(configuration.hidden_size)

    CODE = "def max(a,b): if a>b: return a else return b"
    code_tokens = tokenizer.tokenize(CODE)
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.pad_token for i in range(5)] + [tokenizer.eos_token]
    print(tokens)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(tokens_ids)
    print(tokenizer.decode(tokens_ids))

    convert_code = tokenizer.convert_ids_to_tokens(tokens_ids)
    print(convert_code)

    # print(len(tokens_ids))
    # batch_token_ids = [tokens_ids for _ in range(10)]
    # batch_token_ids = np.stack(arrays=batch_token_ids, axis=0)
    # batch_token_ids = torch.from_numpy(batch_token_ids)
    # print(batch_token_ids.shape)
    # batch_token_ids = batch_token_ids.permute(1, 0)
    # print(batch_token_ids.shape)
    # context_embeddings = model(batch_token_ids)[0]
    #
    # print(context_embeddings.shape)
