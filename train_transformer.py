# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.model.model_transformer import Encoder, Decoder, Transformer
from torch.utils.data import DataLoader
from dataset_flat import MyDatasetFlat
from src.utils import computebleu
from tokenizer import MyTokenizer


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
else:
    torch.manual_seed(123)
device = "cuda" if torch.cuda.is_available() else 'cpu'
VOCAB_FILE = './w2v_vocab_flat.json'
tokenizer = MyTokenizer(VOCAB_FILE)
print(len(tokenizer.vocab))
id2vocab = tokenizer.inverse_vocab
PAD_IDX = tokenizer.pad_index
SOS_IDX = tokenizer.sos_index
VOCAB_SIZE = len(id2vocab)
print(VOCAB_SIZE)

HID_DIM = 512
ENC_LAYERS = 6
DEC_LAYERS = 6
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 2048
DEC_PF_DIM = 2048
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
N_EPOCHS = 10
CLIP = 1
OUTPUT_PATH = './flat_result_trans'
BATCH_SIZE = 128
TRAIN_DATA_PATH = './data/mini_train_flat.csv'
VALID_DATA_PATH = './data/mini_valid_flat.csv'
INPUT_MAX_LENGTH = 300
OUTPUT_MAX_LENGTH = 90

enc = Encoder(VOCAB_SIZE, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(VOCAB_SIZE, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

model = Transformer(enc, dec, PAD_IDX, device).to(device)
total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
model.apply(initialize_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)
# we ignore the loss whenever the target token is a padding token.
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# 数据加载
training_params = {"batch_size": BATCH_SIZE,
                   "shuffle": True,
                   "drop_last": True}
valid_params = {"batch_size": BATCH_SIZE,
                "shuffle": False,
                "drop_last": False}

training_set = MyDatasetFlat(TRAIN_DATA_PATH, INPUT_MAX_LENGTH,
                             OUTPUT_MAX_LENGTH, tokenizer)
training_generator = DataLoader(training_set, **training_params)
valid_set = MyDatasetFlat(VALID_DATA_PATH, INPUT_MAX_LENGTH,
                          OUTPUT_MAX_LENGTH, tokenizer)
valid_generator = DataLoader(valid_set, **valid_params)

loss_vals = []
loss_vals_eval = []
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = []
    pbar = tqdm(training_generator)
    pbar.set_description("[Train Epoch {}]".format(epoch))
    for src, trg in pbar:
        trg, src = trg.to(device), src.to(device)
        model.zero_grad()
        output, _ = model(src, trg[:, :-1])
        # trg = [batch size, trg len]
        # output = [batch size, trg len-1, output dim]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
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

    with torch.no_grad():
        model.eval()
        epoch_loss_eval = []
        pbar = tqdm(valid_generator)
        pbar.set_description("[Eval Epoch {}]".format(epoch))
        # 记录验证集结果
        result_val = []
        acc_num = 0
        bleu_val = 0.0
        n = 0
        for src, trg in pbar:
            trg, src = trg.to(device), src.to(device)
            model.zero_grad()
            batch_size = src.size(0)
            n += batch_size
            output, _ = model(src, trg[:, :-1])
            preds = output.argmax(2)
            # trg = [batch size, trg len]
            # output = [batch size, trg len-1, output dim]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            epoch_loss_eval.append(loss.item())
            accuracy = torch.eq(output.argmax(1), trg).float().mean().item()
            acc_num += accuracy * batch_size
            # 将预测结果转为文字
            trg = trg.view(batch_size, -1)
            preds_val_result = []
            for pred in preds:
                preds_val_result.append(tokenizer.decode(pred.int().tolist()))
            targets_result = []
            for t in trg:
                targets_result.append(tokenizer.decode(t.int().tolist()))

            for pred, target in zip(preds_val_result, targets_result):
                result_val.append((pred, target))
                # 计算 Bleu Score
                bleu_val += computebleu(pred, target)
            pbar.set_postfix(loss=loss.item(), acc=accuracy)
        mean_loss = np.mean(epoch_loss_eval)
        loss_vals_eval.append(mean_loss)
        acc_val = acc_num / n
        bleu_val = bleu_val / n
        print("@@@@@@ Epoch Valid Test: {}/{}, Loss: {}, Accuracy: {}, Bleu-4 score: {}".format(
            epoch + 1,
            N_EPOCHS,
            mean_loss,
            acc_val,
            bleu_val))
        # 储存结果
        with open(OUTPUT_PATH + '/pred_{}.txt'.format(epoch + 1), 'w') as p, open(
                OUTPUT_PATH + '/tgt_{}.txt'.format(epoch + 1), 'w') as t:
            for line in result_val:
                print(line[0], file=p)
                print(line[1], file=t)

        torch.save(model.state_dict(), OUTPUT_PATH + '/model_{}.pt'.format(epoch + 1))

l1, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)
l2, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals_eval)
plt.legend(handles=[l1, l2], labels=['Train loss', 'Eval loss'], loc='best')
plt.savefig('./result_transformer.png')
