"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model.Project2Seq import Project2Seq
from src.utils import get_max_lengths, get_evaluation
from dataset import MyDataset
from src.model.encoder.hierarchical_att_model import HierAttNet
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model: Hierarchical Project Summary Baseline""")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--token_hidden_size", type=int, default=25)
    parser.add_argument("--method_hidden_size", type=int, default=25)
    parser.add_argument("--file_hidden_size", type=int, default=25)
    parser.add_argument("--package_hidden_size", type=int, default=25)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="data/train.csv")
    parser.add_argument("--test_set", type=str, default="data/test.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--pretrained_model", type=str, default="./pretrained/codebert-base")
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--saved_path", type=str, default="./trained_models")
    parser.add_argument("--repo_base_path", type=str, default="./data")
    parser.add_argument("--max_length_package", type=int, default=5)
    parser.add_argument("--max_length_file", type=int, default=5)
    parser.add_argument("--max_length_method", type=int, default=5)
    parser.add_argument("--max_length_token", type=int, default=30)
    parser.add_argument("--max_length_summary", type=int, default=30)
    parser.add_argument("--lang", type=str, default="java")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    lang = opt.lang
    opt.repo_base_path = opt.repo_base_path + os.sep + lang
    opt.saved_path = opt.saved_path + os.sep + lang
    opt.log_path = opt.log_path + os.sep + lang
    bleu_path = opt.saved_path + os.sep + 'bleu'
    if not os.path.exists(bleu_path):
        os.mkdir(bleu_path)

    # 用于将生成的id转换成text
    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model)

    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")

    output_file.write("Model's parameters: {}".format(vars(opt)))
    print("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    training_set = MyDataset(opt.train_set, opt.repo_base_path, opt.pretrained_model, opt.max_length_package, opt.max_length_file, opt.max_length_method,
                 opt.max_length_token, opt.max_length_summary)
    training_generator = DataLoader(training_set, **training_params)
    test_set = MyDataset(opt.test_set, opt.repo_base_path, opt.pretrained_model, opt.max_length_package, opt.max_length_file, opt.max_length_method,
                 opt.max_length_token, opt.max_length_summary)
    test_generator = DataLoader(test_set, **test_params)

    model = Project2Seq(opt)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    best_loss = 1e9
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        print("###### Epoch {} start:".format(epoch + 1))
        iter_index = 0
        for feature, target in tqdm(training_generator):
            iter_index = iter_index + 1
            if torch.cuda.is_available():
                feature = feature.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            predictions = model(feature, target)[0]
            # 将向量变成[batch_size*max_length_summary, vocab_size]方便计算损失值，可参考torch官方api文档
            loss = criterion(predictions.view(opt.batch_size*opt.max_length_summary, -1), target.view(-1))
            loss.backward()
            optimizer.step()
            # TODO：使用bleu-4作为评估指标 训练时不展示bleu指标，只在测试时对测试集展示bleu指标，将每一条的输出写入文件中
            training_metrics = get_evaluation(target.view(-1).cpu().numpy(), predictions.view(opt.batch_size*opt.max_length_summary, -1).cpu().detach().numpy(), list_metrics=["accuracy"])
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
            print("Epoch {} start test".format(epoch + 1))
            for te_feature, te_target in tqdm(test_generator):
                if torch.cuda.is_available():
                    te_feature = te_feature.cuda()
                    te_target = te_target.cuda()
                with torch.no_grad():
                    te_predictions = model.evaluation(te_feature)
                te_loss = criterion(te_predictions.view(opt.batch_size*opt.max_length_summary, -1), te_target.view(-1))
                loss_ls.append(te_loss)
                te_target_ls.extend(te_target.clone().cpu())
                te_pred_ls.extend(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = np.array(te_pred_ls)
            te_target = np.array(te_target_ls)
            # 将prediction和target的每一行内容写在两个文件中 pred.txt target.txt，再调bleu.py得到结果，每个epoch做一次测试
            te_pred_text = tokenizer.batch_decode(te_pred)
            te_target_text = tokenizer.batch_decode(te_target)

            # 写入文本文件
            target_bleu_file.write(te_target_text)
            pred_bleu_file.write(te_pred_text)
            test_metrics = get_evaluation(te_target, te_pred.numpy(), list_metrics=["accuracy"])
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \n\n".format(
                    epoch + 1, opt.num_epoches,
                    te_loss,
                    test_metrics["accuracy"]))
            print("@@@@@@ Epoch Test: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            writer.add_scalar('Test/Loss', te_loss, epoch)
            writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            # if te_loss + opt.es_min_delta < best_loss:
            #     best_loss = te_loss
            #     best_epoch = epoch
            #     torch.save(model, opt.saved_path + os.sep + "whole_model_han")
            #
            # # Early stopping
            # if epoch - best_epoch > opt.es_patience > 0:
            #     print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
            #     break


if __name__ == "__main__":
    opt = get_args()
    train(opt)
