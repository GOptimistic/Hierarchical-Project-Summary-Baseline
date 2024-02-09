"""
@author: Guan Zheng <wwwguanzheng@163.com>
"""
import torch
import sys
import csv
csv.field_size_limit(sys.maxsize)
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        # feature(a,b) weight(b,c) torch.mm-> result(a,c)
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)
    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)
    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]

def masked_softmax(vector, valid_len):
    """
    对两维向量进行带有有效长度掩码的 softmax 操作。

    Parameters:
    - vector: 二维向量，形状为 (batch_size, length)。
    - valid_len: 一维向量，表示每个样本的有效长度，形状为 (batch_size,)。

    Returns:
    - result: softmax 结果。
    """
    # 构造 mask，将无效部分的值设为负无穷，以便在 softmax 中变为 0
    mask = torch.arange(vector.size(1), device=vector.device)[None, :].clone() < valid_len[:, None]

    # 对 mask 应用 softmax，dim=1 表示在 length 维度上进行 softmax
    result = torch.nn.functional.softmax(vector.masked_fill(~mask, float('-inf')), dim=1)

    # 将有效长度为 0 的样本在 softmax 结果中的概率全部置为 0
    result = result.masked_fill(valid_len[:, None] == 0, 0.0)

    return result

def schedule_sampling(step, summary_steps, c, k):
    if c == 0:
        # Inverse sigmoid decay: ϵi = k/(k+exp(i/k))
        # k = np.argmin([np.abs(summary_steps / 2 - x * np.log(x)) for x in range(1, summary_steps)])
        e = k / (k + np.exp(step / k))
    elif c == 1:
        # Linear decay: ϵi = -1/k * i + 1
        e = -1 / summary_steps * step + 1
    elif c == 2:
        # Exponential decay: ϵi = k^i
        e = np.power(0.999, step)
    return e


def computebleu(sentences, targets):
    score = 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<unk>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(0.25, 0.25, 0.25, 0.25))
    return score

if __name__ == "__main__":
    word, sent = get_max_lengths("../data/test.csv")
    print (word)
    print (sent)






