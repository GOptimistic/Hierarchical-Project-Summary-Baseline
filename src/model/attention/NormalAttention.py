import torch
import torch.nn as nn


class NormalAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, enc_outputs, hidden):
        # hidden = [batch size, dec_hid_dim]
        # enc_outputs = [batch_size, seq_len, enc_hid_dim * 2]
        seq_len = enc_outputs.shape[1]
        # s_attn [batch_size, seq_len, dec_hid_dim]
        s_attn = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # E = [batch_size, seq_len, dec_hid_dim]
        E = torch.tanh(self.attn(torch.cat((s_attn, enc_outputs), dim=2)))
        # attention = [batch_size, seq_len]
        attention = self.v(E).squeeze(2)
        # return result: [batch_size, seq_len]
        return nn.functional.softmax(attention, dim=1)
