import pandas as pd
from gensim.models import Word2Vec
import json

# 处理文件摘要
def handle_file_summaries_rwo(file_summaries):
    file_summaries_object = json.loads(file_summaries)
    result = []
    for package in file_summaries_object.keys():
        package_info = file_summaries_object[package]
        for file in package_info.keys():
            file_summary = package_info[file]
            result += file_summary.split()
    return result

# df = pd.read_csv('./data/mini_all.csv')
df = pd.read_csv('../src/file_summary_data/mini_all.csv')
df.head()
SOS = '<sos>'
PAD = '<pad>'
EOS = '<eos>'
UNK = '<unk>'
sent_file_summary = [handle_file_summaries_rwo(row) for row in df['file_summaries']]
sent_repo_summary = [row.split() for row in df['repo_summary']]

print(len(sent_file_summary))
print(len(sent_repo_summary))
sent = sent_file_summary + sent_file_summary
print(len(sent))
model = Word2Vec(sent, min_count=1, vector_size=128, workers=3, window=5, sg=1)
# print(model.wv[SOS])
print(len(model.wv.index_to_key))

index_to_key = [SOS, PAD, EOS, UNK] + model.wv.index_to_key
vocab = {}
for i in range(len(index_to_key)):
    vocab[index_to_key[i]] = i

json_str = json.dumps(vocab)
with open('../w2v_vocab.json', 'w') as f:
    f.write(json_str)
