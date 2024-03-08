"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import argparse
import csv
import os
import shutil
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MyDataset
from src.model.Summary_Two_Level import SummaryTwoLevel
from src.utils import get_evaluation, computebleu
from tokenizer import MyTokenizer


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_data_path", type=str, default="test.csv")
    parser.add_argument("--pretrained_model", type=str, default="/home/LAB/guanz/gz_graduation/code_embedding_pretrained_model/chatglm3-6b-128k")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--token_hidden_size", type=int, default=256)
    parser.add_argument("--method_hidden_size", type=int, default=256)
    parser.add_argument("--file_hidden_size", type=int, default=256)
    parser.add_argument("--package_hidden_size", type=int, default=256)
    parser.add_argument("--decoder_hidden_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="predictions")
    parser.add_argument("--max_package_length", type=int, default=5)
    parser.add_argument("--max_file_length", type=int, default=5)
    parser.add_argument("--max_method_length", type=int, default=5)
    parser.add_argument("--max_token_length", type=int, default=100)
    parser.add_argument("--max_summary_length", type=int, default=30)
    parser.add_argument("--lang", type=str, default="java")
    parser.add_argument("--checkpoint", type=int, default="-1")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    tokenizer = MyTokenizer(opt.vocab_file)
    print(len(tokenizer.vocab))
    opt.vocab_size = len(tokenizer.vocab)
    pad_token_id = tokenizer.pad_index

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    test_set = MyDataset(opt.test_data_path, opt.max_package_length, opt.max_file_length, opt.max_token_length,
                          opt.max_summary_length, tokenizer)
    test_generator = DataLoader(test_set, num_workers=opt.num_workers, **test_params)
    if os.path.isdir(opt.output_path):
        shutil.rmtree(opt.output_path)
    os.makedirs(opt.output_path)
    load_model_path = "./trained_models/java/checkpoint_{}.pkl".format(opt.checkpoint)  # 读取模型位置

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = SummaryTwoLevel(opt, pad_token_id, device)
    model = model.to(device)
    model.to(device)
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id).to(device)
    # 测试模型
    loss_test, bleu_test, acc_test = 0.0, 0.0, 0
    n = 0
    result = []
    for te_file_summaries, te_repo_summary in tqdm(test_generator):
        te_file_summaries = te_file_summaries.to(device)
        te_repo_summary = te_repo_summary.to(device)
        batch_size = te_repo_summary.size(0)
        # print(batch_size)
        outputs_test, preds_test = model.evaluation(te_file_summaries, te_repo_summary)
        # print(preds_test)
        # targets 的第一个 token 是 '<BOS>' 所以忽略
        outputs_test = outputs_test[:, 1:].reshape(-1, outputs_test.size(2))
        te_repo_summary = te_repo_summary[:, 1:].reshape(-1)
        loss = criterion(outputs_test, te_repo_summary)
        loss_test += loss.item()
        acc_test += torch.eq(outputs_test.argmax(1), te_repo_summary).float().mean().item() * batch_size

        # 将预测结果转为文字
        te_repo_summary = te_repo_summary.view(te_file_summaries.size(0), -1)
        preds_val_result = []
        for pred in preds_test:
            preds_val_result.append(tokenizer.decode(pred.int().tolist()))
        targets_result = []
        for tgt in te_repo_summary:
            targets_result.append(tokenizer.decode(tgt.int().tolist()))

        # 记录验证集结果
        for pred, target in zip(preds_val_result, targets_result):
            bleu_test += computebleu(pred, target)
            result.append((pred, target))
        n += batch_size
    loss_test = loss_test / len(test_generator)
    acc_test = acc_test / n
    bleu_test = bleu_test / n
    print('test loss: {}, bleu_score: {}, acc: {}'.format(loss_test, bleu_test, acc_test))
    # 储存结果
    with open(opt.output_path + '/test_pred.txt', 'w') as p, open(opt.output_path + '/test_tgt.txt', 'w') as t:
        for line in result:
            print(line[0], file=p)
            print(line[1], file=t)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
