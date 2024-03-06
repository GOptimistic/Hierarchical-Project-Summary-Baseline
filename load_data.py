"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""

import sys
import time
import csv
from nltk import word_tokenize
import torch
from torchtext.legacy import data
from torchtext.vocab import GloVe

device = "cuda" if torch.cuda.is_available() else 'cpu'


def tokenizer(text):
    tokens = [tok for tok in word_tokenize(text)]
    return tokens


TEXT = data.Field(tokenize=tokenizer,
                  init_token='<sos>',
                  eos_token='<eos>',
                  lower=True,
                  batch_first=True)

train, val = data.TabularDataset.splits(
    path='./src/file_summary_data/',
    train='mini_train_one_level.csv',
    validation='mini_valid_one_level.csv',
    format='csv',
    skip_header=True,
    fields=[('file_summaries', TEXT), ('repo_summary', TEXT)])

TEXT.build_vocab(train, min_freq=2, vectors=GloVe(name='6B', dim=300))
id2vocab = TEXT.vocab.itos
vocab2id = TEXT.vocab.stoi
print(len(TEXT.vocab.stoi))
print(TEXT.vocab.stoi[:100])
PAD_IDX = vocab2id[TEXT.pad_token]
UNK_IDX = vocab2id[TEXT.unk_token]
SOS_IDX = vocab2id[TEXT.init_token]
EOS_IDX = vocab2id[TEXT.eos_token]

# train_iter 自动shuffle, val_iter 按照sort_key排序
train_iter, val_iter = data.BucketIterator.splits(
    (train, val),
    batch_sizes=(32, 32),
    sort_key=lambda x: len(x.src),
    device=device)
