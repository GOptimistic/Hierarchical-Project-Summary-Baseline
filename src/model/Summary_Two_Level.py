import torch
import torch.nn as nn
import random

from src.model.attention.NormalAttention import NormalAttention
from src.model.decoder.gru_decoder import GruDecoder
from src.model.encoder.hierarchical_att_model_two_level import HierAttEncoderTwoLevel


class SummaryTwoLevel(nn.Module):
    def __init__(self, opt, pretrained_model, device):
        super(SummaryTwoLevel, self).__init__()
        self.encoder = HierAttEncoderTwoLevel(opt.token_hidden_size,
                                      opt.file_hidden_size,
                                      opt.package_hidden_size,
                                      pretrained_model,
                                      opt.n_layers,
                                      opt.dropout)
        self.attention = NormalAttention(opt.package_hidden_size)
        self.decoder = GruDecoder(opt.package_hidden_size,
                                  opt.n_layers, opt.dropout,
                                  self.attention,
                                  pretrained_model)
        self.device = device

    def forward(self, file_summaryies, repo_summary, teacher_forcing_ratio):
        # file_summaryies (batch_size, package_size, file_size, token_size),两层循环得到三维向量
        # repo_summary = [batch size, max_length_summary]
        # teacher_forcing_ratio 是使用正解训练的概率
        batch_size = repo_summary.shape[0]
        target_len = repo_summary.shape[1]
        vocab_size = self.decoder.vocab_size

        # 准备一个tensor存储输出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # encoder_outputs用来计算attention，s 用来初始化 Decoder
        encoder_outputs, s = self.encoder(file_summaryies)

        decoder_input = repo_summary[:, 0]  # [bos_id * batch_size]
        preds = []
        for t in range(1, target_len):
            output, s = self.decoder(decoder_input, s, encoder_outputs)
            outputs[:, t] = output
            # 决定是否用正解来训练
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出输出概率最大的词
            top1 = output.argmax(1)
            # teacher force 为 True 用正解训练，否则用预测到的最大概率的词训练
            decoder_input = repo_summary[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def evaluation(self, file_summaryies, repo_summary):
        # 测试模型
        batch_size = repo_summary.shape[0]
        target_len = repo_summary.shape[1]
        vocab_size = self.decoder.vocab_size

        # 准备一个tensor存储输出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # encoder_outputs用来计算attention，s 用来初始化 Decoder
        encoder_outputs, s = self.encoder(file_summaryies)

        decoder_input = repo_summary[:, 0]  # [bos_id * batch_size]
        preds = []
        for t in range(1, target_len):
            output, s = self.decoder(decoder_input, s, encoder_outputs)
            outputs[:, t] = output
            # 取出输出概率最大的词
            top1 = output.argmax(1)
            # 用预测到的最大概率的词进行下一步预测
            decoder_input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds
