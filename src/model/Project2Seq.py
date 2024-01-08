import torch
import torch.nn as nn

from src.model.decoder.gru_decoder import GruDecoder
from src.model.encoder.hierarchical_att_model import HierAttNet


class Project2Seq(nn.Module):
    def __init__(self, opt, pretrained_model, bos_token_id):
        super(Project2Seq, self).__init__()
        self.encoder = HierAttNet(opt.token_hidden_size, opt.method_hidden_size, opt.file_hidden_size, opt.package_hidden_size,
                 opt.batch_size, pretrained_model=pretrained_model)
        self.decoder = GruDecoder(opt.max_length_summary, 2 * opt.package_hidden_size, opt.batch_size, bos_token_id, pretrained_model=pretrained_model)

    def forward(self, repo_info, repo_valid_len, target):
        self.encoder._init_hidden_state()
        # 进行编码
        encoder_hidden = self.encoder(repo_info, repo_valid_len)
        # 进行解码
        decoder_outputs, decoder_hidden = self.decoder(encoder_hidden, target)
        return decoder_outputs, decoder_hidden

    def evaluation(self, repo_info, repo_valid_len):
        self.encoder._init_hidden_state(last_batch_size=repo_info.shape[0])
        encoder_hidden = self.encoder(repo_info, repo_valid_len)
        decoded_sentence = self.decoder.evaluation(encoder_hidden)
        return decoded_sentence

