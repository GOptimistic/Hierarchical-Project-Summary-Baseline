"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import os
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

import matplotlib.pyplot as plt

from dataset_t5 import MyDatasetFileSummaryT5
from src.model.Summary_Two_Level import SummaryTwoLevel
from src.utils import schedule_sampling, computebleu
from dataset import MyDataset
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from torchinfo import summary as infoSummary

from tokenizer import MyTokenizer


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model: Hierarchical Project Summary Baseline""")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--token_hidden_size", type=int, default=256)
    parser.add_argument("--file_hidden_size", type=int, default=256)
    parser.add_argument("--package_hidden_size", type=int, default=256)
    parser.add_argument("--decoder_hidden_size", type=int, default=256)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=10,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_data_path", type=str, default="./data/mini_train_t5.csv")
    parser.add_argument("--valid_data_path", type=str, default="./data/mini_valid_t5.csv")
    parser.add_argument("--valid_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--t5_path", type=str,
                        default="/home/LAB/guanz/gz_graduation/code_embedding_pretrained_model/google-flan-t5-large")
    parser.add_argument("--log_path", type=str, default="./logs_t5")
    parser.add_argument("--saved_path", type=str, default="./trained_models_t5")
    # parser.add_argument("--repo_base_path", type=str, default="./data")
    parser.add_argument("--max_token_length", type=int, default=500)
    parser.add_argument("--max_summary_length", type=int, default=100)
    parser.add_argument("--lang", type=str, default="java")
    parser.add_argument("--checkpoint", type=int, default="-1")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    return args


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    lang = opt.lang
    # opt.repo_base_path = opt.repo_base_path + os.sep + lang
    opt.saved_path = opt.saved_path + os.sep + lang
    opt.log_path = opt.log_path + os.sep + lang
    if not os.path.exists(opt.log_path):
        os.mkdir(opt.log_path)
    bleu_path = opt.saved_path + os.sep + 'bleu'
    if not os.path.exists(bleu_path):
        os.mkdir(bleu_path)

    # 用于将生成的id转换成text
    tokenizer = T5Tokenizer.from_pretrained(opt.t5_path)


    print("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    valid_params = {"batch_size": opt.batch_size,
                    "shuffle": False,
                    "drop_last": False}

    training_set = MyDatasetFileSummaryT5(opt.train_data_path, opt.max_token_length,
                                          opt.max_summary_length, tokenizer)
    training_generator = DataLoader(training_set, **training_params)
    valid_set = MyDatasetFileSummaryT5(opt.valid_data_path, opt.max_token_length,
                                       opt.max_summary_length, tokenizer)
    valid_generator = DataLoader(valid_set, **valid_params)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = T5ForConditionalGeneration.from_pretrained(opt.t5_path)
    model.to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    # infoSummary(model,
    #             [(opt.batch_size, opt.max_package_length, opt.max_file_length, opt.max_token_length),
    #              (opt.batch_size, opt.max_summary_length),
    #              (1,)],
    #             dtypes=[torch.long, torch.long, torch.float])
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    # criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    best_bleu = 1e9
    best_epoch = 0
    epoch_finished = -1
    if opt.checkpoint > 0:
        checkpoint_path = opt.saved_path + os.sep + "checkpoint_{}.pkl".format(opt.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_finished = checkpoint['epoch']
        best_bleu = checkpoint['best_bleu']
        best_epoch = checkpoint['best_epoch']

    # torch.autograd.set_detect_anomaly(True)
    model.train()
    model.zero_grad()
    num_iter_per_epoch = len(training_generator)
    # total_steps = num_iter_per_epoch * opt.num_epoches
    # kk = np.argmin([np.abs(total_steps / 2 - x * np.log(x)) for x in range(1, total_steps)])
    train_losses, val_acces, val_bleu_scores = [], [], []
    for epoch in range(opt.num_epoches):
        if epoch + 1 <= epoch_finished:
            print("###### Epoch {} has archived".format(epoch + 1))
            continue
        print("###### Epoch {} start:".format(epoch + 1))
        iter_index = 0
        model.train()
        for data in tqdm(training_generator, 0):
            iter_index = iter_index + 1
            y = data['target_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            preds = outputs[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y = y[:, 1:]
            accuracy = torch.eq(preds.reshape(-1, preds.size(2)).argmax(1), y.reshape(-1)).float().mean().item()
            print("###### Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter_index,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss.item(),
                accuracy))
            writer.add_scalar('Train/Loss', loss.item(), epoch * num_iter_per_epoch + iter_index)
            writer.add_scalar('Train/Accuracy', accuracy, epoch * num_iter_per_epoch + iter_index)
            train_losses.append(loss.item())
        if (epoch + 1) % opt.valid_interval == 0:
            with torch.no_grad():
                model.eval()

                bleu_val, acc_val = 0.0, 0
                n = 0
                token_n = 0
                result_val = []
                for data in tqdm(valid_generator):
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
                    print(generated_ids.shape)
                    print(target_ids.shape)
                    accuracy = torch.eq(generated_ids.reshape(-1), target_ids.reshape(-1)).float().mean().item()
                    acc_val += accuracy * target_ids.size(0) * target_ids.size(1)
                    token_n += target_ids.size(0) * target_ids.size(1)
                    # print('generated_ids {}'.format(generated_ids))
                    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                             generated_ids]
                    targets = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                              target_ids]
                    # 记录验证集结果
                    for pred, target in zip(preds, targets):
                        # 计算 Bleu Score
                        bleu_val += computebleu(pred, target)
                        result_val.append((pred, target))
                    n += batch_size
                bleu_val = bleu_val / n
                acc_val = acc_val / token_n
                val_acces.append(acc_val)
                val_bleu_scores.append(bleu_val)
                # 储存结果
                with open(bleu_path + os.sep + "valid_result_{}.txt".format(epoch + 1), 'w') as f:
                    for line in result_val:
                        print(line, file=f)
                if bleu_val < best_bleu:
                    best_bleu = bleu_val
                    best_epoch = epoch
                print("@@@@@@ Epoch Valid Test: {}/{}, Acc: {} Bleu-4 score: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    acc_val,
                    bleu_val))
                # writer.add_scalar('Valid/Loss', loss_val, epoch)
                writer.add_scalar('Valid/Accuracy', acc_val, epoch)
                writer.add_scalar('Valid/Bleu-4', bleu_val, epoch)

                # 保存模型
                checkpoint = {"model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch + 1,
                              "best_bleu": best_bleu,
                              "best_epoch": best_epoch}
                path_checkpoint = opt.saved_path + os.sep + "checkpoint_{}.pkl".format(epoch + 1)
                torch.save(checkpoint, path_checkpoint)

    writer.close()
    # 绘图
    plt.figure(1)
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.savefig('./train_loss.png')

    plt.figure(2)
    plt.plot(range(1, len(val_acces) + 1), val_acces)
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Valid Accuracy')
    plt.savefig('./valid_acc.png')

    plt.figure(3)
    plt.plot(range(1, len(val_bleu_scores) + 1), val_bleu_scores)
    plt.xlabel('Epoch')
    plt.ylabel('Bleu-4 Score')
    plt.title('Valid Bleu-4 Score')
    plt.savefig('./valid_bleu.png')
    # torch.autograd.set_detect_anomaly(False)
    print("###### Train done. Best loss {}. Best epoch {}".format(best_bleu, best_epoch))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
