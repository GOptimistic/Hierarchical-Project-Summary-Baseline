"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv

class TokenAttNet(nn.Module):
    def __init__(self, hidden_size=50, pretrained_model='microsoft/codebert-base'):
        super(TokenAttNet, self).__init__()

        # 预训练好的词嵌入模型
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model)
        configuration = self.pretrained_model.config
        print(configuration.vocab_size)
        embeddding_size = configuration.hidden_size

        self.token_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.token_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        # self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = nn.GRU(embeddding_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

        if torch.cuda.is_available():
            self.pretrained_model = self.pretrained_model.cuda()

    def _create_weights(self, mean=0.0, std=0.05):

        self.token_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        # inout:[token_size, batch_size]
        output = self.pretrained_model(input)[0]    # [token_size, batch_size, embeddding_size]
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.token_weight, self.token_bias)   # [token_size, batch_size, 2*token_hidden_size]
        output = matrix_mul(output, self.context_weight).permute(1, 0)  # [batch_size, token_size]
        output = F.softmax(output, dim=1)  # [batch_size, token_size]
        output = element_wise_mul(f_output, output.permute(1, 0))   # [1, batch_size, 2*token_hidden_size]

        return output, h_output


if __name__ == "__main__":
    abc = TokenAttNet("../data/glove.6B.50d.txt")
