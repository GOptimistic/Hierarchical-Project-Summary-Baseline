import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class GruDecoder(nn.Module):
    def __init__(self, max_seq_len, hidden_size, batch_size, pretrained_model):
        super(GruDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # 预训练好的词嵌入模型
        self.embedding = AutoModel.from_pretrained(pretrained_model)
        configuration = self.embedding.config
        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bos_token_id = self.pretrained_tokenizer.convert_tokens_to_ids(self.pretrained_tokenizer.bos_token)

        self.vocab_size = configuration.vocab_size
        self.embedding_dim = configuration.hidden_size

        # self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim,
        #                               padding_idx=num_sequence.PAD)
        self.gru = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True)
        self.log_softmax = nn.LogSoftmax()

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        if torch.cuda.is_available():
            self.embedding = self.embedding.cuda()

    def forward(self, encoder_hidden, target):
        # encoder_hidden [batch_size,hidden_size]
        # target [batch_size,max_len]

        # 初始的全为<s>的输入
        decoder_input = torch.LongTensor([[self.bos_token_id]] * self.batch_size)

        # 解码器的输出，用来后保存所有的输出结果
        decoder_outputs = torch.zeros(self.batch_size, self.max_seq_len, self.vocab_size)

        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        decoder_hidden = encoder_hidden  # [batch_size,hidden_size]

        for t in range(self.max_seq_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)

            # 在不同的time step上进行复制，decoder_output_t [batch_size,vocab_size]
            decoder_outputs[:, t, :] = decoder_output_t

            # 在训练的过程中，使用 teacher forcing，进行纠偏
            use_teacher_forcing = random.random() > 0.5
            if use_teacher_forcing:
                # 下一次的输入使用真实值
                decoder_input = target[:, t].unsqueeze(1)  # [batch_size,1]
            else:
                # 使用预测值，topk中k=1，即获取最后一个维度的最大的一个值
                value, index = torch.topk(decoder_output_t, 1)  # index [batch_size,1]
                decoder_input = index
        # decoder_outputs:[batch_size, max_seq_len, vocab_size]
        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden):
        """
        :param decoder_input:[batch_size,1]
        :param decoder_hidden: [1,batch_size,hidden_size]
        :return: out:[batch_size,vocab_size],decoder_hidden:[1,batch_size,hidden_size]
        """
        embeded = self.embedding(decoder_input)[0]  # embeded: [batch_size,1 , embedding_dim]

        out, decoder_hidden = self.gru(embeded.float(), decoder_hidden)  # out [1, batch_size, hidden_size]

        out = out.squeeze(0)  # 去除第0维度的1
        # 进行全连接形状变化，同时进行求取log_softmax
        out = F.log_softmax(self.fc(out), dim=-1)  # out [batch_Size,1, vocab_size]
        out = out.squeeze(1)
        return out, decoder_hidden

    def evaluation(self, encoder_hidden):
        # batch_size = encoder_hidden.size(0)  # 评估的时候和训练的batch_size不同，不适用config的配置

        decoder_input = torch.LongTensor([[self.bos_token_id]] * self.batch_size)
        decoder_outputs = torch.zeros(self.batch_size, self.max_seq_len, self.vocab_size)  # [batch_size，seq_len,vocab_size]
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        decoder_hidden = encoder_hidden

        # 评估，不再使用teacher forcing，完全使用预测值作为下一次的输入
        for t in range(self.max_seq_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output_t
            value, index = torch.topk(decoder_output_t, 1)
            decoder_input = index

        # 获取输出的id
        # decoder_indices = []
        # for i in range(self.max_seq_len):
        #     value, index = torch.topk(decoder_outputs[:, i, :], k=1, dim=-1)    # index: [batch_size, 1]
        #     decoder_indices.append(index.view(-1).numpy())
        # # transpose 调整为按句子输出 decoder_indices[max_seq_len, batch_size]
        # decoder_indices = np.array(decoder_indices).transpose() # decoder_indices[batch_size, max_seq_len]

        #   [batch_size, max_seq_len, vocal_size]
        return decoder_outputs

