import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.model.attention.NormalAttention import NormalAttention


class GruDecoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, dropout, embedding_size, vocab_size, pad_id):
        super(GruDecoder, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)

        self.vocab_size = vocab_size
        self.attention = NormalAttention(encoder_hidden_size, decoder_hidden_size)

        self.rnn = nn.GRU(encoder_hidden_size * 2 + embedding_size,
                          decoder_hidden_size,
                          batch_first=True)
        self.embedding2vocab = nn.Linear(encoder_hidden_size * 2 + decoder_hidden_size + embedding_size, self.vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, s, enc_outputs):
        # inputs = [batch size] 一开始都是<sos>
        # s = [batch_size, decoder_hidden_size]
        # enc_outputs = [batch_size, package_size, encoder_hidden_size * 2]
        # Decoder 是单向，所以 directions=1
        inputs = inputs.unsqueeze(1)
        # embedded = [batch size, 1, embedding_dim]
        embedded = self.dropout(self.embedding(inputs))
        # a = [batch_size, 1, package_size]
        a = self.attention(enc_outputs, s).unsqueeze(1)
        # c = [batch_size, 1, encoder_hidden_size * 2]
        c = torch.bmm(a, enc_outputs)
        # rnn_input = [batch_size, 1, embedding_dim + encoder_hidden_size * 2]
        # print(embedded.shape)
        # print(c.shape)
        rnn_input = torch.cat((embedded, c), dim=2)
        # dec_output = [batch_size, 1, decoder_hidden_size]
        # s = [1, batch_size, decoder_hidden_size]
        dec_output, s = self.rnn(rnn_input, s.unsqueeze(0))

        embedded = embedded.squeeze(1)
        dec_output = dec_output.squeeze(1)
        c = c.squeeze(1)

        # 将 RNN 的输出向量的维数转换到target语言的字典大小
        # output = [batch size, vocab size]
        output = self.embedding2vocab(torch.cat((dec_output, c, embedded), dim=1))
        # output = self.embedding2vocab2(output)

        return output, s.squeeze(0)



if __name__ == '__main__':
    pretrained_tokenizer = AutoTokenizer.from_pretrained('../../pretrained/codebert-base')
    bos_token_id = pretrained_tokenizer.convert_tokens_to_ids(pretrained_tokenizer.bos_token)
    print(bos_token_id)
    pad_token_id = pretrained_tokenizer.convert_tokens_to_ids(pretrained_tokenizer.pad_token)
    print(pad_token_id)
    batch_size = 8
    decoder_input = torch.LongTensor([[bos_token_id]] * batch_size)
    print(decoder_input.shape)
