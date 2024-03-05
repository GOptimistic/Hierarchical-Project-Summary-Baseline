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
    def __init__(self, token_hidden_size, file_hidden_size, package_hidden_size,
                 pretrained_model, n_layers, dropout):
        super(HierAttEncoderTwoLevel, self).__init__()

        self.token_att_net = TokenAttNet(token_hidden_size, pretrained_model, n_layers, dropout)

        self.file_att_net = FileAttNet(file_hidden_size, token_hidden_size, n_layers, dropout)
        self.package_att_net = PackageAttNet(package_hidden_size, file_hidden_size, n_layers, dropout)

    def forward(self, file_summaryies):
        # file_summaryies (batch_size, package_size, file_size, token_size)
        package_input = file_summaryies.permute(1, 2, 0, 3)
        package_embedding_list = []
        for file_input in package_input:
            # file_input (file_size, batch_size, token_size)
            file_embedding_list = []
            for token_input in file_input:
                # token_input[batch_size, token_size]
                # print('token_input')
                # print(token_input)
                file_embedding = self.token_att_net(token_input)
                print('file_embedding')
                print(file_embedding)
                # file_embedding [batch_size, 1, 2*token_hidden_size]
                file_embedding_list.append(file_embedding)
            # 将file_embedding拼接送入file层输入
            file_embedding_list = torch.cat(file_embedding_list, 1)
            # package_embedding [batch_size, 1, 2*file_hidden_size]
            package_embedding = self.file_att_net(file_embedding_list)
            print('package_embedding')
            print(package_embedding)
            package_embedding_list.append(package_embedding)
        # 将package_embedding拼接送入package输入
        package_embedding_list = torch.cat(package_embedding_list, 1)
        package_outputs, package_hidden = self.package_att_net(package_embedding_list)

        # package_outputs [batch_size, package_size, 2*package_hidden_size]
        # package_hidden = [num_layers, batch_size, package_hidden_size * 2]
        return package_outputs, package_hidden


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
