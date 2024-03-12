"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import argparse
import csv
import os
import shutil

from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MyDataset
from dataset_flat import MyDatasetFlat
from src.model.Summary_Two_Level import SummaryTwoLevel
from src.model.model_seq2seq import Seq2Seq
from src.utils import get_evaluation, computebleu
from tokenizer import MyTokenizer


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_data_path", type=str, default="./data/mini_test_flat.csv")
    parser.add_argument("--pretrained_model", type=str,
                        default="/home/LAB/guanz/gz_graduation/code_embedding_pretrained_model/chatglm3-6b-128k")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="predictions_flat")
    parser.add_argument("--max_input_length", type=int, default=300)
    parser.add_argument("--max_output_length", type=int, default=90)
    parser.add_argument("--lang", type=str, default="java")
    parser.add_argument("--checkpoint", type=int, default="9")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--vocab_file", type=str, default="./w2v_vocab_flat.json")
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
    sos_index = tokenizer.sos_index

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    test_set = MyDatasetFlat(opt.test_data_path,
                             opt.max_input_length,
                             opt.max_output_length,
                             tokenizer)
    test_generator = DataLoader(test_set, **test_params)
    load_model_path = "./flat_result/model_{}.pt".format(opt.checkpoint)  # 读取模型位置

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = Seq2Seq(opt.vocab_size,
                    opt.embedding_size,
                    opt.hidden_size,
                    opt.hidden_size,
                    opt.dropout,
                    device)
    model.to(device)
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint)

    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id).to(device)
        # 测试模型
        loss_test = []
        bleu_test, acc_test = 0.0, 0
        bleu_one, bleu_two, bleu_three, bleu_four = 0.0, 0.0, 0.0, 0.0
        n = 0
        n_token = 0
        result = []
        for src, trg in tqdm(test_generator):
            trg, src = trg.to(device), src.to(device)
            model.zero_grad()
            batch_size = src.size(0)
            n += batch_size
            output = model.inference(src, opt.max_output_length, sos_index)
            preds = output.argmax(2)
            # trg = [batch size, trg len]
            # output = [batch size, trg len, output dim]
            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            loss_test.append(loss.item())
            accuracy = torch.eq(output.argmax(1), trg).float().mean().item()
            acc_test += accuracy * trg.size(0)
            n_token += trg.size(0)
            # 将预测结果转为文字
            trg = trg.view(src.size(0), -1)
            preds_val_result = []
            for pred in preds:
                preds_val_result.append(tokenizer.decode(pred.int().tolist()))
            targets_result = []
            for t in trg:
                targets_result.append(tokenizer.decode(t.int().tolist()))

            for pred, target in zip(preds_val_result, targets_result):
                result.append((pred, target))
                # 计算 Bleu Score
                bleu_one += sentence_bleu([target.split()], pred.split(), weights=(1, 0, 0, 0))
                bleu_two += sentence_bleu([target.split()], pred.split(), weights=(0, 1, 0, 0))
                bleu_three += sentence_bleu([target.split()], pred.split(), weights=(0, 0, 1, 0))
                bleu_four += sentence_bleu([target.split()], pred.split(), weights=(0, 0, 0, 1))
                bleu_test += computebleu(pred, target)
        loss_test = loss_test / len(test_generator)
        acc_test = acc_test / n_token
        bleu_test = bleu_test / n
        bleu_one = bleu_one / n
        bleu_two = bleu_two / n
        bleu_three = bleu_three / n
        bleu_four = bleu_four / n
        print('test loss: {}, Bleu1 {} Bleu2 {} Bleu3 {} Bleu4 {} Bleu4_score: {}, acc: {}'.format(
            loss_test,
            bleu_one,
            bleu_two,
            bleu_three,
            bleu_four,
            bleu_test, acc_test))
        # 储存结果
        with open(opt.output_path + '/test_seq_pred.txt', 'w') as p, open(opt.output_path + '/test_seq_tgt.txt',
                                                                          'w') as t:
            for line in result:
                print(line[0], file=p)
                print(line[1], file=t)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
