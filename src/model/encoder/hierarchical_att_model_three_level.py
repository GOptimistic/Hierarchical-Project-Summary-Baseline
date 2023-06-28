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


class HierAttNet_three_level(nn.Module):
    def __init__(self, token_hidden_size, method_hidden_size,
                 batch_size, pretrained_model):
        super(HierAttNet_three_level, self).__init__()
        self.batch_size = batch_size
        self.token_hidden_size = token_hidden_size
        self.method_hidden_size = method_hidden_size
        self.pretrained_model = pretrained_model

        self.token_att_net = TokenAttNet(self.token_hidden_size, self.pretrained_model)
        self.method_att_net = MethodAttNet(method_hidden_size=self.method_hidden_size, token_hidden_size=self.token_hidden_size)

        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.token_hidden_state = torch.zeros(2, batch_size, self.token_hidden_size)
        self.method_hidden_state = torch.zeros(2, batch_size, self.method_hidden_size)
        if torch.cuda.is_available():
            self.token_hidden_state = self.token_hidden_state.cuda()
            self.method_hidden_state = self.method_hidden_state.cuda()

    def forward(self, input):
        # 我的输入应该是5维向量,(batch_size, package_size, file_size, method_size, token_size),两层循环得到三维向量
        batch_size, package_size, file_size, method_size, token_size = input.size()
        input = input.view(batch_size, -1, token_size)
        input = input.permute(1, 0, 2)
        output_list = []
        for i in input:
            output, self.token_hidden_state = self.token_att_net(i.permute(1, 0), self.token_hidden_state)
            # output是每个method的向量表示
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.method_hidden_state = self.method_att_net(output, self.method_hidden_state)

        return output


if __name__ == '__main__':
    model = HierAttNet_three_level(4, 5, 3, "../../pretrained/codebert-base")
    test = MyDataset(data_path="../../data/data_python_output_100rows.csv", repo_base_path="../../data/python", max_length_package=5, max_length_file=4, max_length_method=3,
                     max_length_token=10, max_length_summary=50, pretrained_model="../../pretrained/codebert-base")
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
