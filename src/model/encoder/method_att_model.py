"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import torch.nn as nn

from src.model.attention.NormalAttention import NormalAttention
from src.utils import matrix_mul, element_wise_mul, masked_softmax


class MethodAttNet(nn.Module):
    def __init__(self, method_hidden_size=128, token_hidden_size=128, n_layers=1, dropout=0.5):
        super(MethodAttNet, self).__init__()

        self.n_layers = n_layers
        self.rnn = nn.GRU(token_hidden_size*2, method_hidden_size, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(method_hidden_size * 2, method_hidden_size * 2)
        self.method_attention = NormalAttention(method_hidden_size)

    def forward(self, method_input, method_valid_len):
        # method_input: [batch_size, method_size, 2*token_hidden_size]
        # method_valid_len: [batch_size]
        batch_size = method_input.shape[0]
        # outputs [batch_size, method_size, 2*method_hidden_size] hidden [2*n_layers, method_size, method_hidden_size]
        outputs, hidden = self.rnn(self.dropout(method_input))
        # 因为 Encoder 是双向RNN，所以需要对同一层两个方向的 hidden state 进行拼接
        # hidden = [num_layers * 2, batch size, method_hidden_size] --> [num_layers, directions, batch size, method_hidden_size]
        hidden = hidden.view(self.n_layers, 2, batch_size, -1)
        # s = [num_layers, batch size, method_hidden_size * 2]
        s = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        s = torch.tanh(self.fc(s))

        # outputs [batch_size, method_size, 2*method_hidden_size]
        # s = [num_layers, batch_size, method_hidden_size * 2]

        # a = [batch_size, 1, method_size]
        a = self.method_attention(outputs, s).unsqueeze(1)
        file_embedding = torch.bmm(a, outputs)
        # file_embedding [batch_size, 1, 2*method_hidden_size]
        return file_embedding


if __name__ == "__main__":
    abc = MethodAttNet()
