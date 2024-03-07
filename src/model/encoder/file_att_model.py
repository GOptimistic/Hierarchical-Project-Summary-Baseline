"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention.NormalAttention import NormalAttention
from src.utils import matrix_mul, element_wise_mul, masked_softmax


class FileAttNet(nn.Module):
    def __init__(self, file_hidden_size=128, token_hidden_size=128, dropout=0.5):
        super(FileAttNet, self).__init__()

        self.rnn = nn.GRU(token_hidden_size * 2, file_hidden_size, batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(file_hidden_size * 2, file_hidden_size * 2)
        # self.file_attention = NormalAttention(file_hidden_size)

    def forward(self, file_input):
        # file_input: [batch_size, file_size, 2*token_hidden_size]
        # outputs [batch_size, file_size, 2*file_hidden_size] hidden [2, batch_size, file_hidden_size]
        outputs, hidden = self.rnn(self.dropout(file_input))
        # 因为 Encoder 是双向RNN，所以需要对同一层两个方向的 hidden state 进行拼接
        # s = [batch_size, file_hidden_size * 2]
        s = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # s = torch.tanh(self.fc(s))

        # outputs [batch_size, file_size, 2*file_hidden_size]
        # s = [num_layers, batch_size, file_hidden_size * 2]

        # a = [batch_size, 1, file_size]
        # a = self.file_attention(outputs, s).unsqueeze(1)
        # package_embedding = torch.bmm(a, outputs)
        # # package_embedding [batch_size, 1, 2*file_hidden_size]
        # return package_embedding
        return s.unsqueeze(1)


if __name__ == "__main__":
    abc = FileAttNet()
