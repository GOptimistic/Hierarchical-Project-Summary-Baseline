import json

class MyTokenizer:
    def __init__(self, vocab_path):
        self.vocab = self.load_vocab(vocab_path)
        self.inverse_vocab = {index: word for word, index in self.vocab.items()}
        # 定义特殊词汇的索引
        self.sos_index = self.vocab["<sos>"]
        self.eos_index = self.vocab["<eos>"]
        self.pad_index = self.vocab["<pad>"]
        self.unk_index = self.vocab["<unk>"]

    def load_vocab(self, vocab_path):
        """从JSON文件中加载词表"""
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        return vocab

    def encode(self, text, max_length):
        """将文本编码为索引列表，并添加<sos>和<eos>"""
        words = text.split()

        # 对于不在词表中的词汇使用<unk>
        encoded = [self.vocab.get(word, self.unk_index) for word in words]
        encoded = [self.sos_index] + encoded + [self.eos_index]
        if len(encoded) > max_length:
            encoded = encoded[:max_length - 1]
            encoded = encoded + [self.eos_index]
        else:
            extended = [self.pad_index for _ in range(max_length - len(encoded))]
            encoded.extend(extended)
        return encoded

    def decode(self, indices):
        """将索引列表解码回文本，忽略特殊词汇"""
        words = [self.inverse_vocab.get(index, '<unk>') for index in indices if index not in [self.sos_index, self.eos_index, self.pad_index]]
        return ' '.join(words)
