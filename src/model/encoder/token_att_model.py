"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from src.utils import matrix_mul, element_wise_mul, masked_softmax



class TokenAttNet(nn.Module):
    def __init__(self, hidden_size=128, pretrained_model=None, n_layers=1, dropout=0.5):
        super(TokenAttNet, self).__init__()

        # 预训练好的词嵌入模型
        if pretrained_model is None:
            pretrained_model = AutoModel.from_pretrained('microsoft/codebert-base')
        pretrained_embedding = pretrained_model.embeddings.word_embeddings.weight.data
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)

        configuration = pretrained_model.config
        vocab_size = configuration.vocab_size
        embedding_size = configuration.hidden_size

        # self.token_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        # self.token_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        # self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self.n_layers = n_layers
        self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)

    def forward(self, input):
        # input:[batch_size, token_size] valid_len:[batch_size]
        # use nn.embedding.from_pretrained 替换 embedding
        batch_size = input.shape[0]
        embedding = self.embedding(input)  # [batch_size, token_size, embeddding_size]
        # outputs [batch_size, token_size, 2*hidden_size] hidden [2*n_layers, batch_size, hidden_size]
        outputs, hidden = self.rnn(self.dropout(embedding))
        # 因为 Encoder 是双向RNN，所以需要对同一层两个方向的 hidden state 进行拼接
        # hidden = [num_layers * 2, batch size, hidden_size] --> [num_layers, directions, batch size, hidden_size]
        hidden = hidden.view(self.n_layers, 2, batch_size, -1)
        # s = [num_layers, batch size, hidden_size * 2]
        s = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        s = torch.tanh(self.fc(s))
        # output = matrix_mul(f_output, self.token_weight,
        #                     self.token_bias)  # [token_size, batch_size, 2*token_hidden_size]
        # output = matrix_mul(output, self.context_weight).permute(1, 0)  # [batch_size, token_size]
        # output = masked_softmax(output, valid_len)  # [batch_size, token_size]
        # output = element_wise_mul(f_output, output.permute(1, 0))  # [1, batch_size, 2*token_hidden_size]

        return outputs, s
