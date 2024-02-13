import torch
import torch.nn as nn

from src.model.decoder.gru_decoder import GruDecoder
from src.model.encoder.hierarchical_att_model_three_level import HierAttNet_three_level


class Project2Seq_three_level(nn.Module):
    def __init__(self, opt):
        super(Project2Seq_three_level, self).__init__()
        self.encoder = HierAttNet_three_level(opt.token_hidden_size, opt.method_hidden_size,
                                  opt.batch_size, pretrained_model=opt.pretrained_model)
        self.decoder = GruDecoder(opt.max_length_summary, 2 * opt.method_hidden_size, opt.batch_size,
                                  pretrained_model=opt.pretrained_model)

    def forward(self, input, target):
        self.encoder._init_hidden_state()
        # 进行编码
        encoder_hidden = self.encoder(input)
        # 进行解码
        decoder_outputs, decoder_hidden = self.decoder(encoder_hidden, target)
        return decoder_outputs, decoder_hidden

    def evaluation(self, inputs):
        self.encoder._init_hidden_state()
        encoder_hidden = self.encoder(inputs)
        decoded_sentence = self.decoder.evaluation(encoder_hidden)
        return decoded_sentence
