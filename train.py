"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from src.model.Project2Seq import Project2Seq
from src.model.Project2Seq_three_level import Project2Seq_three_level
from src.utils import get_max_lengths, get_evaluation
from dataset import MyDataset
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model: Hierarchical Project Summary Baseline""")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epoches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--token_hidden_size", type=int, default=64)
    parser.add_argument("--method_hidden_size", type=int, default=64)
    parser.add_argument("--file_hidden_size", type=int, default=64)
    parser.add_argument("--package_hidden_size", type=int, default=64)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=10,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_dir_path", type=str, default="../clone_github_repo_data/java/hdf5_no_compress_data/")
    parser.add_argument("--train_data_path_prefix", type=str, default="train_")
    parser.add_argument("--valid_data_path_prefix", type=str, default="valid_")
    parser.add_argument("--valid_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--pretrained_model", type=str, default="./src/pretrained/codebert-base")
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--saved_path", type=str, default="./trained_models")
    # parser.add_argument("--repo_base_path", type=str, default="./data")
    parser.add_argument("--max_length_package", type=int, default=30)
    parser.add_argument("--max_length_file", type=int, default=10)
    parser.add_argument("--max_length_method", type=int, default=20)
    parser.add_argument("--max_length_token", type=int, default=350)
    parser.add_argument("--max_length_summary", type=int, default=40)
    parser.add_argument("--lang", type=str, default="java")
    parser.add_argument("--checkpoint", type=int, default="-1")
    parser.add_argument("--model_level", type=int, default="5")
    parser.add_argument("--train_part_size", type=int, default="3200")
    parser.add_argument("--train_total_length", type=int, default="24357")
    parser.add_argument("--valid_part_size", type=int, default="400")
    parser.add_argument("--valid_total_length", type=int, default="3045")
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
    opt.train_data_path_prefix = opt.train_data_path_prefix + lang + "_"
    opt.valid_data_path_prefix = opt.valid_data_path_prefix + lang + "_"
    opt.saved_path = opt.saved_path + os.sep + lang
    opt.log_path = opt.log_path + os.sep + lang
    if not os.path.exists(opt.log_path):
        os.mkdir(opt.log_path)
    bleu_path = opt.saved_path + os.sep + 'bleu'
    if not os.path.exists(bleu_path):
        os.mkdir(bleu_path)

    # 用于将生成的id转换成text
    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model)
    pretrained_model = AutoModel.from_pretrained(opt.pretrained_model)
    bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    print("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    valid_params = {"batch_size": opt.batch_size,
                    "shuffle": False,
                    "drop_last": False}

    training_set = MyDataset(opt.data_dir_path, opt.train_data_path_prefix, opt.train_part_size, opt.train_total_length)
    training_generator = DataLoader(training_set, **training_params)
    valid_set = MyDataset(opt.data_dir_path, opt.valid_data_path_prefix, opt.valid_part_size, opt.valid_total_length)
    valid_generator = DataLoader(valid_set, **valid_params)

    if opt.model_level == 3:
        model = Project2Seq_three_level(opt)
    else:
        model = Project2Seq(opt, pretrained_model, bos_token_id)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model,
    #                  (torch.zeros(opt.batch_size, opt.max_length_package, opt.max_length_file, opt.max_length_method, opt.max_length_token).long().to(device),
    #                  torch.zeros(opt.batch_size, opt.max_length_package, opt.max_length_file, opt.max_length_method).long().to(device),
    #                  torch.zeros(opt.batch_size, opt.max_length_summary).long().to(device)))


    criterion = nn.NLLLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    epoch_finished = -1
    if opt.checkpoint > 0:
        checkpoint_path = opt.saved_path + os.sep + "checkpoint_{}.pkl".format(opt.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_finished = checkpoint['epoch']

    best_loss = 1e9
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        if epoch + 1 <= epoch_finished:
            print("###### Epoch {} has archived".format(epoch + 1))
            continue
        print("###### Epoch {} start:".format(epoch + 1))
        iter_index = 0
        for repo_info, repo_valid_len, summary, summary_valid_len in tqdm(training_generator):
            iter_index = iter_index + 1
            if torch.cuda.is_available():
                repo_info = repo_info.cuda()
                repo_valid_len = repo_valid_len.cuda()
                summary = summary.cuda()
                summary_valid_len = summary_valid_len.cuda()
            optimizer.zero_grad()
            predictions = model(repo_info, repo_valid_len, summary)[0]
            # 将向量变成[batch_size*max_length_summary, vocab_size]方便计算损失值，可参考torch官方api文档
            loss = criterion(predictions.view(opt.batch_size * opt.max_length_summary, -1), summary.view(-1))
            loss.backward()
            optimizer.step()
            # TODO：使用bleu-4作为评估指标 训练时不展示bleu指标，只在测试时对测试集展示bleu指标，将每一条的输出写入文件中
            training_metrics = get_evaluation(summary.view(-1).cpu().numpy(),
                                              predictions.view(opt.batch_size * opt.max_length_summary,
                                                               -1).cpu().detach().numpy(), list_metrics=["accuracy"])
            print("###### Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter_index,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter_index)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter_index)
        if epoch % opt.test_interval == 0:
            model.eval()
            target_bleu_file = open(bleu_path + os.sep + "target_bleu_{}.txt".format(epoch + 1), "w")
            pred_bleu_file = open(bleu_path + os.sep + "pred_bleu_{}.txt".format(epoch + 1), "w")
            loss_ls = []
            te_target_ls = []
            te_pred_ls = []
            print("Epoch {} start valid test".format(epoch + 1))
            for te_repo_info, te_repo_valid_len, te_summary, te_summary_valid_len in tqdm(valid_generator):
                if torch.cuda.is_available():
                    te_repo_info = te_repo_info.cuda()
                    te_repo_valid_len = te_repo_valid_len.cuda()
                    te_summary = te_summary.cuda()
                    te_summary_valid_len = te_summary_valid_len.cuda()
                with torch.no_grad():
                    te_predictions = model.evaluation(te_repo_info, te_repo_valid_len)
                # 最后一个batch的size不一定是opt.batch_size,所以用第一维大小代替
                te_loss = criterion(te_predictions.view(te_repo_info.shape[0] * opt.max_length_summary, -1),
                                    te_summary.view(-1))
                loss_ls.append(te_loss)
                te_target_ls.extend(te_summary.clone().cpu().numpy())
                te_pred_ls.extend(te_predictions.clone().cpu().numpy())
            te_loss = sum(loss_ls)
            te_pred = np.array(te_pred_ls)
            te_target = np.array(te_target_ls)
            # 将prediction和target的每一行内容写在两个文件中 pred.txt target.txt，再调bleu.py得到结果，每个epoch做一次测试
            te_pred_text = tokenizer.batch_decode(np.argmax(te_pred, -1))
            te_target_text = tokenizer.batch_decode(te_target)

            # 写入文本文件
            for i in range(len(te_target_text)):
                target_bleu_file.write(te_target_text[i] + "\n")
                pred_bleu_file.write(te_pred_text[i] + "\n")
            valid_metrics = get_evaluation(te_target.reshape(-1),
                                          te_pred.reshape(len(te_pred) * opt.max_length_summary, -1),
                                          list_metrics=["accuracy"])

            print("@@@@@@ Epoch Valid Test: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, valid_metrics["accuracy"]))
            writer.add_scalar('Valid/Loss', te_loss, epoch)
            writer.add_scalar('Valid/Accuracy', valid_metrics["accuracy"], epoch)
            # 保存模型
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch + 1}
            path_checkpoint = opt.saved_path + os.sep + "checkpoint_{}.pkl".format(epoch + 1)
            torch.save(checkpoint, path_checkpoint)
            # torch.save(model.state_dict(), opt.saved_path + os.sep + "checkpoint_{}.pt".format(epoch + 1))
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
            #
            # # Early stopping
            # if epoch - best_epoch > opt.es_patience > 0:
            #     print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
            #     break

    writer.close()
    print("###### Train done. Best loss {}. Best epoch {}".format(best_loss, best_epoch))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
