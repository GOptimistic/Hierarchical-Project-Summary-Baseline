import torch
import torch.nn as nn
import random

from src.model.attention.NormalAttention import NormalAttention
from src.model.decoder.gru_decoder import GruDecoder
from src.model.encoder.hierarchical_att_model import HierAttNet
from src.model.encoder.hierarchical_att_model_two_level import HierAttNet_Two_Level


class Project2Seq_Two_Level(nn.Module):
    def __init__(self, opt, pretrained_model, bos_token_id, device):
        super(Project2Seq_Two_Level, self).__init__()
        self.encoder = HierAttNet_Two_Level(opt.token_hidden_size,
                                            pretrained_model,
                                            opt.n_layers,
                                            opt.dropout)
        self.attention = NormalAttention(opt.token_hidden_size)
        self.decoder = GruDecoder(opt.token_hidden_size,
                                  opt.n_layers, opt.dropout,
                                  self.attention,
                                  pretrained_model)
        self.bos_token_id = bos_token_id
        self.device = device
        self.target_len = opt.max_length_summary

    def forward(self, repo_info, repo_valid_len, target, teacher_forcing_ratio):
        # repo_info = [batch size, max_length_token]
        # repo_valid_len = [batch size]
        # target = [batch size, max_length_summary]
        # teacher_forcing_ratio 是使用正解训练的概率
        batch_size = repo_info.shape[0]
        vocab_size = self.decoder.vocab_size

        # 准备一个tensor存储输出
        outputs = torch.zeros(batch_size, self.target_len, vocab_size).to(self.device)
        # encoder_outputs用来计算attention，s 用来初始化 Decoder
        encoder_outputs, s = self.encoder(repo_info, repo_valid_len)

        decoder_input = torch.LongTensor([[self.bos_token_id]] * batch_size).to(self.device)
        preds = []
        for t in range(self.target_len):
            output, s = self.decoder(decoder_input, s, encoder_outputs)
            outputs[:, t] = output
            # 决定是否用正解来训练
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出输出概率最大的词
            top1 = output.argmax(1)
            # teacher force 为 True 用正解训练，否则用预测到的最大概率的词训练
            decoder_input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def evaluation(self, repo_info, repo_valid_len):
        # 测试模型
        batch_size = repo_info.shape[0]
        vocab_size = self.decoder.vocab_size

        # 准备一个tensor存储输出
        outputs = torch.zeros(batch_size, self.target_len, vocab_size).to(self.device)
        # encoder_outputs用来计算attention，s 用来初始化 Decoder
        encoder_outputs, s = self.encoder(input)

        decoder_input = torch.LongTensor([[self.bos_token_id]] * batch_size).to(self.device)
        preds = []
        for t in range(0, self.target_len):
            output, s = self.decoder(decoder_input, s, encoder_outputs)
            outputs[:, t] = output
            # 取出输出概率最大的词
            top1 = output.argmax(1)
            # 用预测到的最大概率的词进行下一步预测
            decoder_input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds


