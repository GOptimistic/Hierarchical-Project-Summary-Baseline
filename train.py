"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel

from src.model.Project2Seq import Project2Seq
from src.model.Project2Seq_three_level import Project2Seq_three_level
from src.model.Project2Seq_two_level import Project2Seq_Two_Level
from src.utils import get_max_lengths, get_evaluation, schedule_sampling, computebleu
from dataset import MyDataset
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from torchinfo import summary as infoSummary


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model: Hierarchical Project Summary Baseline""")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epoches", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--token_hidden_size", type=int, default=32)
    parser.add_argument("--method_hidden_size", type=int, default=32)
    parser.add_argument("--file_hidden_size", type=int, default=32)
    parser.add_argument("--package_hidden_size", type=int, default=32)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=10,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_dir_path", type=str, default="../clone_github_repo_data/java/hdf5_no_compress_data")
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
    parser.add_argument("--max_length_token", type=int, default=20)
    parser.add_argument("--max_length_summary", type=int, default=40)
    parser.add_argument("--lang", type=str, default="java")
    parser.add_argument("--checkpoint", type=int, default="-1")
    parser.add_argument("--model_level", type=int, default="5")
    parser.add_argument("--train_part_size", type=int, default="3200")
    parser.add_argument("--train_total_length", type=int, default="24357")
    parser.add_argument("--valid_part_size", type=int, default="400")
    parser.add_argument("--valid_total_length", type=int, default="3045")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ddp setting
    if opt.local_rank != -1:
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    lang = opt.lang
    # opt.repo_base_path = opt.repo_base_path + os.sep + lang
    opt.train_data_path_prefix = opt.train_data_path_prefix + lang + "_"
    opt.valid_data_path_prefix = opt.valid_data_path_prefix + lang + "_"
    opt.saved_path = opt.saved_path + os.sep + lang
    opt.log_path = opt.log_path + os.sep + lang + os.sep + "gpu-{}".format(opt.local_rank)
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
                       "drop_last": True}
    valid_params = {"batch_size": opt.batch_size,
                    "shuffle": False,
                    "drop_last": False}

    training_set = MyDataset(opt.data_dir_path, opt.train_data_path_prefix, opt.train_part_size, opt.train_total_length, opt.max_length_token)
    train_sampler = DistributedSampler(training_set)
    training_generator = DataLoader(training_set, sampler=train_sampler, pin_memory=True, num_workers=opt.num_workers, **training_params)
    valid_set = MyDataset(opt.data_dir_path, opt.valid_data_path_prefix, opt.valid_part_size, opt.valid_total_length, opt.max_length_token)
    valid_generator = DataLoader(valid_set, **valid_params)



    if opt.model_level == 2:
        model = Project2Seq_Two_Level(opt, pretrained_model, bos_token_id, device)
    elif opt.model_level == 3:
        model = Project2Seq_three_level(opt)
    else:
        model = Project2Seq(opt, pretrained_model, bos_token_id, device)
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    # print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")
    # infoSummary(model,
                 # [(opt.batch_size, opt.max_length_package, opt.max_length_file, opt.max_length_method, opt.max_length_token),
                 # (opt.batch_size, opt.max_length_package, opt.max_length_file, opt.max_length_method),
                 # (opt.batch_size, opt.max_length_summary),
                 # (1,)],
                # dtypes=[torch.long, torch.long, torch.long, torch.float])

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank],
                                                    output_device=opt.local_rank)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model,
    #                  (torch.zeros(opt.batch_size, opt.max_length_package, opt.max_length_file, opt.max_length_method, opt.max_length_token).long().to(device),
    #                  torch.zeros(opt.batch_size, opt.max_length_package, opt.max_length_file, opt.max_length_method).long().to(device),
    #                  torch.zeros(opt.batch_size, opt.max_length_summary).long().to(device)))


    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id).to(device)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    best_loss = 1e9
    best_epoch = 0
    epoch_finished = -1
    if opt.checkpoint > 0:
        checkpoint_path = opt.saved_path + os.sep + "checkpoint_{}.pkl".format(opt.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_finished = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_epoch = checkpoint['best_epoch']

    # torch.autograd.set_detect_anomaly(True)
    model.train()
    model.zero_grad()
    num_iter_per_epoch = len(training_generator)
    total_steps = num_iter_per_epoch * opt.num_epoches
    kk = np.argmin([np.abs(total_steps / 2 - x * np.log(x)) for x in range(1, total_steps)])
    train_losses, val_losses, val_bleu_scores = [], [], []
    for epoch in range(opt.num_epoches):
        if epoch + 1 <= epoch_finished:
            print("###### Epoch {} has archived".format(epoch + 1))
            continue
        print("###### Epoch {} start:".format(epoch + 1))
        iter_index = 0
        model.train()
        train_sampler.set_epoch(epoch) #shuffle
        for repo_info, repo_valid_len, summary, summary_valid_len in tqdm(training_generator):
            iter_index = iter_index + 1
            repo_info = repo_info.to(device)
            repo_valid_len = repo_valid_len.to(device)
            summary = summary.to(device)
            # summary_valid_len = summary_valid_len.to(device)
            # print(repo_info.shape)
            # print(repo_valid_len.shape)
            optimizer.zero_grad()
            outputs, preds = model(repo_info,
                                   repo_valid_len,
                                   summary,
                                   schedule_sampling(epoch * num_iter_per_epoch + iter_index, total_steps, c=0, k=kk))
            # predictions = model(repo_info, repo_valid_len, summary)[0]
            # 将向量变成[batch_size*max_length_summary, vocab_size]方便计算损失值，可参考torch官方api文档
            outputs = outputs.reshape(-1, outputs.size(2))
            summary = summary.reshape(-1)
            loss = criterion(outputs, summary)
            loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1)
            optimizer.step()
            accuracy = torch.eq(outputs.argmax(1), summary).float().mean().item()
            print("### GPU: {} Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                opt.local_rank,
                epoch + 1,
                opt.num_epoches,
                iter_index,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss,
                accuracy))
            writer.add_scalar('Train/Loss-{}'.format(opt.local_rank), loss, epoch * num_iter_per_epoch + iter_index)
            writer.add_scalar('Train/Accuracy-{}'.format(opt.local_rank), accuracy, epoch * num_iter_per_epoch + iter_index)
            train_losses.append(loss)
        torch.distributed.barrier()
        if (epoch + 1) % opt.valid_interval == 0 and opt.local_rank in [-1, 0]:
            model.eval()

            loss_val, bleu_val, acc_val = 0.0, 0.0, 0.0
            n = 0
            result_val = []
            for te_repo_info, te_repo_valid_len, te_summary, te_summary_valid_len in tqdm(valid_generator):
                te_repo_info = te_repo_info.to(device)
                te_repo_valid_len = te_repo_valid_len.to(device)
                te_summary = te_summary.to(device)
                # te_summary_valid_len = te_summary_valid_len.to(device)
                batch_size = te_repo_info.size(0)
                # print(batch_size)
                outputs_val, preds_val = model.module.evaluation(te_repo_info, te_repo_valid_len)
                # targets 的第一个 token 是 '<BOS>' 所以忽略
                outputs_val = outputs_val.reshape(-1, outputs_val.size(2))
                te_summary = te_summary.reshape(-1)
                loss = criterion(outputs_val, te_summary)
                loss_val += loss.item()
                acc_val += torch.eq(outputs_val.argmax(1), te_summary).float().mean().item()

                # 将预测结果转为文字
                te_summary = te_summary.view(te_repo_info.size(0), -1)
                preds_val = tokenizer.batch_decode(preds_val)
                targets_val = tokenizer.batch_decode(te_summary)

                # 记录验证集结果
                for pred, target in zip(preds_val, targets_val):
                    result_val.append((pred, target))
                # 计算 Bleu Score
                bleu_val += computebleu(preds_val, targets_val)
                n += batch_size
            loss_val = loss_val / len(valid_generator)
            acc_val = acc_val / len(valid_generator)
            bleu_val = bleu_val / n
            val_losses.append(loss_val)
            val_bleu_scores.append(bleu_val)
            # 储存结果
            with open(bleu_path + os.sep + "valid_result_{}.txt".format(epoch + 1), 'w') as f:
                for line in result_val:
                    print(line, file=f)
            if loss_val + opt.es_min_delta < best_loss:
                best_loss = loss_val
                best_epoch = epoch

            print("@@@ GPU: {} Epoch Valid Test: {}/{}, Lr: {}, Loss: {}, Accuracy: {}, Bleu-4 score: {}".format(
                opt.local_rank,
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                loss_val,
                acc_val,
                bleu_val))
            writer.add_scalar('Valid/Loss-{}'.format(opt.local_rank), loss_val, epoch)
            writer.add_scalar('Valid/Accuracy-{}'.format(opt.local_rank), acc_val, epoch)
            writer.add_scalar('Valid/Bleu-4-{}'.format(opt.local_rank), bleu_val, epoch)

            # 保存模型
            checkpoint = {"model_state_dict": model.module.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch + 1,
                          "best_loss": best_loss,
                          "best_epoch": best_epoch}
            path_checkpoint = opt.saved_path + os.sep + "checkpoint_{}.pkl".format(epoch + 1)
            torch.save(checkpoint, path_checkpoint)

        torch.distributed.barrier()
        model.train()

    writer.close()
    # torch.autograd.set_detect_anomaly(False)
    print("###### Train done. Best loss {}. Best epoch {}".format(best_loss, best_epoch))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
