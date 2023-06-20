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


class HierAttNet(nn.Module):
    def __init__(self, token_hidden_size, method_hidden_size, file_hidden_size, package_hidden_size,
                 batch_size, pretrained_model):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.token_hidden_size = token_hidden_size
        self.method_hidden_size = method_hidden_size
        self.file_hidden_size = file_hidden_size
        self.package_hidden_size = package_hidden_size
        self.pretrained_model = pretrained_model

        self.token_att_net = TokenAttNet(self.token_hidden_size, self.pretrained_model)
        self.method_att_net = MethodAttNet(method_hidden_size=self.method_hidden_size, token_hidden_size=self.token_hidden_size)
        self.file_att_net = FileAttNet(file_hidden_size=self.file_hidden_size, method_hidden_size=self.method_hidden_size)
        self.package_att_net = PackageAttNet(package_hidden_size=self.package_hidden_size, file_hidden_size=self.file_hidden_size)

        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.token_hidden_state = torch.zeros(2, batch_size, self.token_hidden_size)
        self.method_hidden_state = torch.zeros(2, batch_size, self.method_hidden_size)
        self.file_hidden_state = torch.zeros(2, batch_size, self.file_hidden_size)
        self.package_hidden_state = torch.zeros(2, batch_size, self.package_hidden_size)
        if torch.cuda.is_available():
            self.token_hidden_state = self.token_hidden_state.cuda()
            self.method_hidden_state = self.method_hidden_state.cuda()
            self.file_hidden_state = self.file_hidden_state.cuda()
            self.package_hidden_state = self.package_hidden_state.cuda()

    def forward(self, input):
        # 我的输入应该是5维向量,(batch_size, package_size, file_size, method_size, token_size),两层循环得到三维向量
        package_input = input.permute(1, 2, 3, 0, 4)
        package_embedding_list = []
        for file_input in package_input:
            file_embedding_list = []
            for method_input in file_input:
                method_embedding_list = []
                for token_input in method_input:
                    # print("###### enter token att")
                    method_embedding, self.token_hidden_state = self.token_att_net(token_input.permute(1, 0), self.token_hidden_state)
                    method_embedding_list.append(method_embedding)
                # 将method_embedding拼接送入method层输入
                method_embedding_list = torch.cat(method_embedding_list, 0)
                # print("###### enter method att")
                file_embedding, self.method_hidden_state = self.method_att_net(method_embedding_list, self.method_hidden_state)
                file_embedding_list.append(file_embedding)
            # 将file_embedding拼接送入file层输入
            file_embedding_list = torch.cat(file_embedding_list, 0)
            # print("###### enter file att")
            package_embedding, self.file_hidden_state = self.file_att_net(file_embedding_list, self.file_hidden_state)
            package_embedding_list.append(package_embedding)
        # 将package_embedding拼接送入package输入
        package_embedding_list = torch.cat(package_embedding_list, 0)
        # print("###### enter package att")
        repo_output, self.package_hidden_state = self.package_att_net(package_embedding_list, self.package_hidden_state)

        # 返回的是一个三维向量[1, batch_size, 2*package_hidden_size]
        return repo_output


if __name__ == '__main__':
    model = HierAttNet(2, 2, 2, 4, 3, "./pretrained/codebert-base")
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
