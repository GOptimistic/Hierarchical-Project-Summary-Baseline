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


class HierAttNet_Two_Level(nn.Module):
    def __init__(self, token_hidden_size, pretrained_model, n_layers, dropout):
        super(HierAttNet_Two_Level, self).__init__()
        self.token_att_net = TokenAttNet(token_hidden_size, pretrained_model, n_layers, dropout)

    def forward(self, input, valid_len):
        # input (batch_size, token_size),两层循环得到三维向量
        # valid_len (batch_size)

        outputs, s = self.token_att_net(input)

        # outputs [batch_size, token_size, 2*hidden_size]
        # s = [num_layers, batch size, hidden_size * 2]
        return outputs, s


if __name__ == '__main__':
    model = HierAttNet_Two_Level(2, 2, 2, 4, 3, "./pretrained/codebert-base")
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
