"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from src.model.attention.NormalAttention import NormalAttention
from src.utils import matrix_mul, element_wise_mul, masked_softmax



class TokenAttNet(nn.Module):
    def __init__(self, hidden_size=128, pretrained_model=None, n_layers=1, dropout=0.5):
        super(TokenAttNet, self).__init__()

        # 预训练好的词嵌入模型
        if pretrained_model is None:
            raise Exception('Pretrained_model is none!')
        pretrained_embedding = pretrained_model.transformer.embedding.word_embeddings.weight.data
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)

        configuration = pretrained_model.config
        vocab_size = configuration.padded_vocab_size
        embedding_size = configuration.hidden_size

        self.n_layers = n_layers
        self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        # self.token_attention = NormalAttention(hidden_size)

    def forward(self, token_input):
        # token_input:[batch_size, token_size]
        # use nn.embedding.from_pretrained 替换 embedding
        batch_size = token_input.shape[0]
        embedding = self.embedding(token_input)  # [batch_size, token_size, embeddding_size]
        # outputs [batch_size, token_size, 2*hidden_size] hidden [2*n_layers, batch_size, hidden_size]
        outputs, hidden = self.rnn(self.dropout(embedding))
        # 因为 Encoder 是双向RNN，所以需要对同一层两个方向的 hidden state 进行拼接
        # hidden = [num_layers * 2, batch size, hidden_size] --> [num_layers, directions, batch size, hidden_size]
        hidden = hidden.view(self.n_layers, 2, batch_size, -1)
        # s = [num_layers, batch size, hidden_size * 2]
        s = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        s = torch.tanh(self.fc(s))
        # print('s')
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
        s = torch.mean(s, 0).permute(1, 0, 2)
        return s
