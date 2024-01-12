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
                 batch_size, local_rank, pretrained_model):
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
        if torch.cuda.is_available():
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cpu")
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.token_hidden_state = torch.zeros(2, batch_size, self.token_hidden_size).to(self.device)
        self.method_hidden_state = torch.zeros(2, batch_size, self.method_hidden_size).to(self.device)
        self.file_hidden_state = torch.zeros(2, batch_size, self.file_hidden_size).to(self.device)
        self.package_hidden_state = torch.zeros(2, batch_size, self.package_hidden_size).to(self.device)

    def forward(self, input, valid_len):
        # input (batch_size, package_size, file_size, method_size, token_size),两层循环得到三维向量
        # valid_len (batch_size, package_size, file_size, method_size)
        package_input = input.permute(1, 2, 3, 0, 4)
        package_valid_len = valid_len.permute(1, 2, 3, 0)  # (package_size, file_size, method_size, batch_size)
        package_embedding_list = []
        for file_input, file_valid_len in zip(package_input, package_valid_len):
            # file_input (file_size, method_size, batch_size, token_size)
            # file_valid_len (file_size, method_size, batch_size)
            file_embedding_list = []
            for method_input, method_valid_len in zip(file_input, file_valid_len):
                # method_input (method_size, batch_size, token_size)
                # method_valid_len (method_size, batch_size)
                method_embedding_list = []
                for token_input, token_valid_len in zip(method_input, method_valid_len):
                    # print("###### enter token att")
                    # token_input[batch_size, token_size]
                    # token_valid_len[batch_size]
                    method_embedding, self.token_hidden_state = self.token_att_net(token_input.permute(1, 0), self.token_hidden_state, token_valid_len)
                    method_embedding_list.append(method_embedding)
                # 将method_embedding拼接送入method层输入
                # (method_size, batch_size, 2*token_hidden_size)
                method_valid_len = method_valid_len.permute(1, 0)
                # 获取有效的method 长度，末尾可能有些method是padding，会是全0的元素
                method_valid_len = torch.sum(method_valid_len != 0, dim=1)
                method_embedding_list = torch.cat(method_embedding_list, 0)
                # print("###### enter method att")
                file_embedding, self.method_hidden_state = self.method_att_net(method_embedding_list, self.method_hidden_state, method_valid_len)
                file_embedding_list.append(file_embedding)
            # 将file_embedding拼接送入file层输入
            file_embedding_list = torch.cat(file_embedding_list, 0)
            # 获取有效的file 长度，末尾可能有些file是padding，会是全0的元素
            file_valid_len = file_valid_len.permute(2, 0, 1)
            file_valid_len = torch.sum(file_valid_len != 0, dim=2)
            file_valid_len = torch.sum(file_valid_len != 0, dim=1)
            # print("###### enter file att")
            package_embedding, self.file_hidden_state = self.file_att_net(file_embedding_list, self.file_hidden_state, file_valid_len)
            package_embedding_list.append(package_embedding)
        # 将package_embedding拼接送入package输入
        package_embedding_list = torch.cat(package_embedding_list, 0)
        # 获取有效的package 长度，末尾可能有些package是padding，会是全0的元素
        package_valid_len = package_valid_len.permute(3, 0, 1, 2)
        package_valid_len = torch.sum(package_valid_len != 0, dim=3)
        package_valid_len = torch.sum(package_valid_len != 0, dim=2)
        package_valid_len = torch.sum(package_valid_len != 0, dim=1)
        # print("###### enter package att")
        repo_output, self.package_hidden_state = self.package_att_net(package_embedding_list, self.package_hidden_state, package_valid_len)

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
