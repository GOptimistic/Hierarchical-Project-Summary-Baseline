# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from load_data import train_iter, val_iter, id2vocab, PAD_IDX, SOS_IDX
from src.model.model_seq2seq import Encoder, Decoder, Seq2Seq, Attention

device = "cuda" if torch.cuda.is_available() else 'cpu'
INPUT_DIM = len(id2vocab)
OUTPUT_DIM = len(id2vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 300
OUTPUT_MAX_LENGTH = 30
OUTPUT_PATH = './direct_summary_result'
CLIP = 1

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


def decode(ids_list):
    res = []
    for id in ids_list:
        word = id2vocab[id]
        res.append(word)
    return ' '.join(res)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=5e-5)
# we ignore the loss whenever the target token is a padding token.
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

loss_vals = []
loss_vals_eval = []
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = []
    pbar = tqdm(train_iter)
    pbar.set_description("[Train Epoch {}]".format(epoch))
    for src, trg in pbar:
        trg, src = trg.to(device), src.to(device)
        model.zero_grad()
        output = model(src, trg)
        # trg = [batch size, trg len]
        # output = [batch size, trg len, output dim]
        output_dim = output.shape[-1]
        output = output[:, 1:, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        epoch_loss.append(loss.item())
        optimizer.step()
        accuracy = torch.eq(output.argmax(1), trg).float().mean().item()
        pbar.set_postfix(loss=loss.item(), acc=accuracy)
    loss_vals.append(np.mean(epoch_loss))

    model.eval()
    epoch_loss_eval = []
    pbar = tqdm(val_iter)
    pbar.set_description("[Eval Epoch {}]".format(epoch))
    # 记录验证集结果
    result_val = []
    for src, trg in pbar:
        trg, src = trg.to(device), src.to(device)
        model.zero_grad()
        output = model.inference(src, OUTPUT_MAX_LENGTH, SOS_IDX)
        # trg = [batch size, trg len]
        # output = [batch size, trg len, output dim]
        output_dim = output.shape[-1]
        output = output[:, 1:, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        epoch_loss_eval.append(loss.item())
        accuracy = torch.eq(output.argmax(1), trg).float().mean().item()
        # 将预测结果转为文字
        trg = trg.view(src.size(0), -1)
        preds = output.argmax(1).view(src.size(0), -1)
        preds_val_result = []
        for pred in preds:
            preds_val_result.append(decode(pred.int().tolist()))
        targets_result = []
        for t in trg:
            targets_result.append(decode(t.int().tolist()))

        for pred, target in zip(preds_val_result, targets_result):
            result_val.append((pred, target))

        pbar.set_postfix(loss=loss.item(), acc=accuracy)
    loss_vals_eval.append(np.mean(epoch_loss_eval))
    # 储存结果
    with open(OUTPUT_PATH + '/test_pred_{}.txt'.format(epoch), 'w') as p, open(
            OUTPUT_PATH + '/test_tgt_{}.txt'.format(epoch), 'w') as t:
        for line in result_val:
            print(line[0], file=p)
            print(line[1], file=t)

    torch.save(model.state_dict(), 'model_{}.pt'.format(epoch))

l1, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)
l2, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals_eval)
plt.legend(handles=[l1, l2], labels=['Train loss', 'Eval loss'], loc='best')
plt.savefig('./result.png')
