import torch
import torch.nn as nn


class NormalAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2 + hid_dim * 2, hid_dim * 2, bias=False)
        self.v = nn.Linear(hid_dim * 2, 1, bias=False)

    def forward(self, enc_outputs, s):
        # s = [num_layers, batch_size, hid_dim * 2]
        # enc_outputs = [batch_size, seq_len, hid_dim * 2]
        batch_size = enc_outputs.shape[0]
        seq_len = enc_outputs.shape[1]
        # s_attn = [num_layers, batch_size, seq_len, hid_dim * 2] -> [batch_size, seq_len, hid_dim * 2]
        s_attn = s.unsqueeze(2).repeat(1, 1, seq_len, 1)
        s_attn = torch.mean(s_attn, 0)
        # E = [batch_size, seq_len, hid_dim * 2]
        E = torch.tanh(self.attn(torch.cat((s_attn, enc_outputs), dim=2)))
        # attention = [batch_size, seq_len]
        attention = self.v(E).squeeze(2)
        # return result: [batch_size, seq_len]
        return nn.functional.softmax(attention, dim=1)
