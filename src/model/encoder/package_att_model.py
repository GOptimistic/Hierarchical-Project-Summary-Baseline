"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul, masked_softmax


class PackageAttNet(nn.Module):
    def __init__(self, package_hidden_size=128, file_hidden_size=128, decoder_hidden_size=128, dropout=0.5):
        super(PackageAttNet, self).__init__()

        self.rnn = nn.GRU(file_hidden_size * 2, package_hidden_size, dropout=dropout, batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(package_hidden_size * 2, decoder_hidden_size)

    def forward(self, package_input):
        # package_input: [batch_size, package_size, 2*file_hidden_size]
        # package_valid_len: [batch_size]
        # outputs [batch_size, package_size, 2*package_hidden_size] hidden [2, batch_size, package_hidden_size]
        outputs, hidden = self.rnn(self.dropout(package_input))
        # 因为 Encoder 是双向RNN，所以需要对同一层两个方向的 hidden state 进行拼接
        # s = [batch_size, package_hidden_size * 2]
        s = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # s = [batch_size, package_hidden_size]
        s = torch.tanh(self.fc(s))

        # outputs [batch_size, package_size, 2*package_hidden_size]
        # s = [batch_size, decoder_hidden_size]
        return outputs, s


if __name__ == "__main__":
    abc = PackageAttNet()
