import pandas as pd
from gensim.models import Word2Vec
import json

# 处理文件摘要
def handle_file_summaries_rwo(file_summaries):
    file_summaries_object = json.loads(file_summaries)
    result = []
    for file in file_summaries_object.keys():
        file_summary = file_summaries_object[file]
        result += file_summary.split()
    return result

# df = pd.read_csv('./data/mini_all.csv')
df = pd.read_csv('/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/src/file_summary_data/analyze_import_data/file_summary_all.csv')
df.head()
SOS = '<sos>'
PAD = '<pad>'
EOS = '<eos>'
UNK = '<unk>'
sent_file_summary = [handle_file_summaries_rwo(row) for row in df['file_summaries']]
sent_repo_summary = [row.split() for row in df['repo_summary']]

print(len(sent_file_summary))
print(len(sent_repo_summary))
sent = sent_file_summary + sent_repo_summary
print(len(sent))
model = Word2Vec(sent, min_count=2, vector_size=128, workers=3, window=5, sg=1)
# print(model.wv[SOS])
print(len(model.wv.index_to_key))

index_to_key = [SOS, PAD, EOS, UNK] + model.wv.index_to_key
vocab = {}
for i in range(len(index_to_key)):
    vocab[index_to_key[i]] = i

json_str = json.dumps(vocab)
with open('./w2v_vocab_import.json', 'w') as f:
    f.write(json_str)
