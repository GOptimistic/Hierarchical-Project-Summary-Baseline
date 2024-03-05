import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class GruDecoder(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout, attention, pretrained_model=None):
        super(GruDecoder, self).__init__()
        self.hidden_size = hidden_size

        # 预训练好的词嵌入模型
        if pretrained_model is None:
            raise Exception('Pretrained_model is none!')
        pretrained_embedding = pretrained_model.transformer.embeddings.word_embeddings.weight.data
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)
        configuration = pretrained_model.config

        self.vocab_size = configuration.padded_vocab_size
        self.embedding_dim = configuration.hidden_size
        self.attention = attention
        # self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim,
        #                               padding_idx=num_sequence.PAD)
        self.rnn = nn.GRU(input_size=self.embedding_dim + self.hidden_size * 2,
                          hidden_size=self.hidden_size * 2,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.embedding2vocab = nn.Linear(self.hidden_size * 2 + self.hidden_size * 2 + self.embedding_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, s, enc_outputs):
        # input = [batch size] 一开始都是<BOS>
        # s = [num_layers, batch_size, hidden_size * 2]
        # enc_outputs = [batch_size, package_size, hidden_size * 2]
        # Decoder 是单向，所以 directions=1
        input = input.unsqueeze(1)
        # embedded = [batch size, 1, embedding_dim]
        embedded = self.dropout(self.embedding(input))
        # a = [batch_size, 1, package_size]
        a = self.attention(enc_outputs, s).unsqueeze(1)
        # c = [batch_size, 1, hidden_size * 2]
        c = torch.bmm(a, enc_outputs)
        # rnn_input = [batch_size, 1, embedding_dim + hidden_size * 2]
        # print(embedded.shape)
        # print(c.shape)
        rnn_input = torch.cat((embedded, c), dim=2)
        # dec_output = [batch_size, 1, hidden_size * 2]
        # s = [num_layers, batch_size, hidden_size * 2]
        dec_output, s = self.rnn(rnn_input, s)

        embedded = embedded.squeeze(1)
        dec_output = dec_output.squeeze(1)
        c = c.squeeze(1)

        # 将 RNN 的输出向量的维数转换到target语言的字典大小
        # output = [batch size, vocab size]
        output = self.embedding2vocab(torch.cat((dec_output, c, embedded), dim=1))
        # output = self.embedding2vocab2(output)

        return output, s



if __name__ == '__main__':
    pretrained_tokenizer = AutoTokenizer.from_pretrained('../../pretrained/codebert-base')
    bos_token_id = pretrained_tokenizer.convert_tokens_to_ids(pretrained_tokenizer.bos_token)
    print(bos_token_id)
    pad_token_id = pretrained_tokenizer.convert_tokens_to_ids(pretrained_tokenizer.pad_token)
    print(pad_token_id)
    batch_size = 8
    decoder_input = torch.LongTensor([[bos_token_id]] * batch_size)
    print(decoder_input.shape)
