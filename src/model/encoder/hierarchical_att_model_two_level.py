"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MyDataset
from src.model.encoder.token_att_model import TokenAttNet
from src.model.encoder.file_att_model import FileAttNet
from src.model.encoder.method_att_model import MethodAttNet
from src.model.encoder.package_att_model import PackageAttNet


class HierAttEncoderTwoLevel(nn.Module):
    def __init__(self, token_hidden_size, file_hidden_size, decoder_hidden_size,
                 embedding_size, dropout, vocab_size, pad_id):
        super(HierAttEncoderTwoLevel, self).__init__()

        self.token_att_net = TokenAttNet(vocab_size, token_hidden_size, embedding_size, dropout, pad_id)
        self.file_att_net = FileAttNet(file_hidden_size, token_hidden_size, decoder_hidden_size, dropout)

    def forward(self, file_summaryies):
        # file_summaryies (batch_size, file_size, token_size)
        file_summaryies = file_summaryies.permute(1, 0, 2)
        file_embedding_list = []
        for token_input in file_summaryies:
            # token_input[batch_size, token_size]
            # print('token_input')
            # print(token_input)
            # file_embedding [batch_size, 1, 2*token_hidden_size]
            file_embedding = self.token_att_net(token_input)
            # print('file_embedding')
            # print(file_embedding)
            file_embedding_list.append(file_embedding)
        # 将file_embedding拼接送入file层输入
        file_embedding_list = torch.cat(file_embedding_list, 1)
        # file_outputs [batch_size, file_size, 2*file_hidden_size]
        # file_hidden = [batch_size, decoder_hidden_size]
        file_outputs, file_hidden = self.file_att_net(file_embedding_list)

        return file_outputs, file_hidden


if __name__ == '__main__':
    model = HierAttEncoderTwoLevel(2, 2, 2, 4, 3, "./pretrained/codebert-base")
    test = MyDataset(data_path="./data/data_python_output_100rows.csv", repo_base_path="./data/python", max_length_package=5, max_length_file=4, max_length_method=3,
                     max_length_token=10, max_length_summary=50, pretrained_model="./pretrained/codebert-base")
    # print(test.__getitem__(index=1)[1].shape)

    training_params = {"batch_size": 3,
                       "shuffle": True,
                       "drop_last": True}
    training_generator = DataLoader(test, **training_params)
    model.train()
    for iter, (feature, label) in enumerate(training_generator):
        print("######")
        print(feature.shape)
        print(label.shape)
        model._init_hidden_state()
        predictions = model(feature)
        print(predictions.shape)
