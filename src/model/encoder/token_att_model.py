"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import torch.nn as nn



class TokenAttNet(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, dropout, pad_id):
        super(TokenAttNet, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)

        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        # self.token_attention = NormalAttention(hidden_size)

    def forward(self, token_input):
        # token_input:[batch_size, token_size]
        embedded = self.embedding(token_input)  # [batch_size, token_size, embedding_size]
        # print('embedded')
        # print(embedded)
        # outputs [batch_size, token_size, 2*hidden_size] hidden [2, batch_size, hidden_size]
        outputs, hidden = self.rnn(self.dropout(embedded))
        # 因为 Encoder 是双向RNN，所以需要对同一层两个方向的 hidden state 进行拼接
        # hidden = [2, batch size, hidden_size]
        # s = [batch size, hidden_size * 2]
        s = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # print('before s')
        # print(s)
        # s = torch.tanh(self.fc(s))
        # print('after s')
        # print(s)
        # # outputs [batch_size, token_size, 2*hidden_size]
        # # s = [num_layers, batch size, hidden_size * 2]
        #
        # # a = [batch_size, 1, token_size]
        # a = self.token_attention(outputs, s).unsqueeze(1)
        # print('a')
        # print(a)
        # file_embedding = torch.bmm(a, outputs)
        # print('file_embedding')
        # print(file_embedding)
        # method_embedding [batch_size, 1, 2*hidden_size]
        # s = torch.mean(s, 0).unsqueeze(0).permute(1, 0, 2)
        # s = [batch size, 1, hidden_size * 2]
        return s.unsqueeze(1)
