"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import argparse
import csv
import os
import shutil

from nltk.translate.bleu_score import sentence_bleu
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MyDataset
from dataset_t5 import MyDatasetFileSummaryT5
from src.model.Summary_Two_Level import SummaryTwoLevel
from src.utils import get_evaluation, computebleu
from tokenizer import MyTokenizer


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_data_path", type=str, default="./data/mini_test_t5.csv")
    parser.add_argument("--t5_path", type=str,
                        default="/home/LAB/guanz/gz_graduation/code_embedding_pretrained_model/google-flan-t5-large")
    parser.add_argument("--output_path", type=str, default="predictions_t5")
    parser.add_argument("--max_token_length", type=int, default=600)
    parser.add_argument("--max_summary_length", type=int, default=30)
    parser.add_argument("--lang", type=str, default="java")
    parser.add_argument("--checkpoint", type=int, default="-1")
    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # 用于将生成的id转换成text
    tokenizer = T5Tokenizer.from_pretrained(opt.t5_path)

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    test_set = MyDatasetFileSummaryT5(opt.test_data_path, opt.max_token_length,
                          opt.max_summary_length, tokenizer)
    test_generator = DataLoader(test_set, **test_params)
    if os.path.isdir(opt.output_path):
        shutil.rmtree(opt.output_path)
    os.makedirs(opt.output_path)
    load_model_path = "./trained_models_t5/java/checkpoint_{}.pkl".format(opt.checkpoint)  # 读取模型位置

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = T5ForConditionalGeneration.from_pretrained(opt.t5_path)
    model.to(device)

    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Load model done.')

    with torch.no_grad():
        model.eval()
        # criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id).to(device)
        # 测试模型
        bleu_test = 0.0
        bleu_one, bleu_two, bleu_three, bleu_four = 0.0, 0.0, 0.0, 0.0
        n = 0
        result = []
        for data in tqdm(test_generator):
            target_ids = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            batch_size = target_ids.size(0)
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=opt.max_summary_length,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0
            )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            targets = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                       target_ids]

            # 记录验证集结果
            for pred, target in zip(preds, targets):
                # 计算 Bleu Score
                bleu_one += sentence_bleu([target.split()], pred.split(), weights=(1, 0, 0, 0))
                bleu_two += sentence_bleu([target.split()], pred.split(), weights=(0, 1, 0, 0))
                bleu_three += sentence_bleu([target.split()], pred.split(), weights=(0, 0, 1, 0))
                bleu_four += sentence_bleu([target.split()], pred.split(), weights=(0, 0, 0, 1))
                bleu_test += computebleu(pred, target)
                result.append((pred, target))
            n += batch_size
        bleu_test = bleu_test / n
        bleu_one = bleu_one / n
        bleu_two = bleu_two / n
        bleu_three = bleu_three / n
        bleu_four = bleu_four / n
        print("@@@@@@ Test: Bleu1 {} Bleu2 {} Bleu3 {} Bleu4 {} Bleu-4 score: {}".format(
            bleu_one,
            bleu_two,
            bleu_three,
            bleu_four,
            bleu_test))
        # 储存结果
        with open(opt.output_path + '/test_pred.txt', 'w') as p, open(opt.output_path + '/test_tgt.txt', 'w') as t:
            for line in result:
                print(line[0], file=p)
                print(line[1], file=t)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
