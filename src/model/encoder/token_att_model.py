"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from src.utils import matrix_mul, element_wise_mul, masked_softmax



class TokenAttNet(nn.Module):
    def __init__(self, hidden_size=128, pretrained_model=None):
        super(TokenAttNet, self).__init__()

        # 预训练好的词嵌入模型
        self.pretrained_model = pretrained_model
        if self.pretrained_model is None:
            self.pretrained_model = AutoModel.from_pretrained('microsoft/codebert-base')
        pretrained_embedding = self.pretrained_model.embeddings.word_embeddings.weight.data
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)

        configuration = self.pretrained_model.config
        print(configuration.vocab_size)
        embedding_size = configuration.hidden_size

        self.token_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.token_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        # self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

        # if torch.cuda.is_available():
        #     self.pretrained_model = self.pretrained_model.cuda()

    def _create_weights(self, mean=0.0, std=0.05):
        self.token_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state, valid_len):
        # input:[token_size, batch_size] hidden_state:[2, batch_size, token_hidden_size] valid_len:[batch_size]
        # use nn.embedding.from_pretrained 替换 embedding
        output = self.embedding(input)  # [token_size, batch_size, embeddding_size]
        if torch.cuda.is_available():
            output = output.cuda()
        # f_output [token_size, batch_size, 2*token_hidden_size] h_output [2, batch_size, token_hidden_size]
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.token_weight,
                            self.token_bias)  # [token_size, batch_size, 2*token_hidden_size]
        output = matrix_mul(output, self.context_weight).permute(1, 0)  # [batch_size, token_size]
        output = masked_softmax(output, valid_len)  # [batch_size, token_size]
        output = element_wise_mul(f_output, output.permute(1, 0))  # [1, batch_size, 2*token_hidden_size]

        return output, h_output
