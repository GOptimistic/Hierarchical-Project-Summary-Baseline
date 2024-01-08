"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul, masked_softmax


class PackageAttNet(nn.Module):
    def __init__(self, package_hidden_size=128, file_hidden_size=128):
        super(PackageAttNet, self).__init__()

        self.package_weight = nn.Parameter(torch.Tensor(2 * package_hidden_size, 2 * package_hidden_size))
        self.package_bias = nn.Parameter(torch.Tensor(1, 2 * package_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * package_hidden_size, 1))

        self.gru = nn.GRU(2 * file_hidden_size, package_hidden_size, bidirectional=True)
        # self.fc = nn.Linear(2 * package_hidden_size, num_classes)
        # self.package_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.package_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state, valid_len):
        # input: [package_size, batch_size, 2*file_hidden_size]
        f_output, h_output = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.package_weight, self.package_bias) # [package_size, batch_size, 2*package_hidden_size]
        output = matrix_mul(output, self.context_weight).permute(1, 0)  # [batch_size, package_size]
        output = masked_softmax(output, valid_len)  # [batch_size, package_size]
        # f_output: [package_size, batch_size, 2*package_hidden_size]
        output = element_wise_mul(f_output, output.permute(1, 0))    # [1, batch_size, 2*package_hidden_size]

        return output, h_output


if __name__ == "__main__":
    abc = PackageAttNet()
