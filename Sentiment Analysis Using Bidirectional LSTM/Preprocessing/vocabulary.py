from collections import Counter

class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, min_freq, max_size):
        self.min_freq = min_freq
        self.max_size = max_size
        self.freq = Counter()
        self.stoi = {}
        self.itos = {}

    def build(self, dataset, tokenizer):
        for sample in dataset:
            tokens = tokenizer.tokenize(sample["text"])
            self.freq.update(tokens)

        tokens = [t for t, f in self.freq.items() if f >= self.min_freq]
        tokens = sorted(tokens, key=lambda x: self.freq[x], reverse=True)
        tokens = tokens[:self.max_size]

        self.stoi = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1
        }

        for idx, token in enumerate(tokens, start=2):
            self.stoi[token] = idx

        self.itos = {i: t for t, i in self.stoi.items()}

    def numericalize(self, tokens):
        return [self.stoi.get(t, self.stoi[self.UNK_TOKEN]) for t in tokens]

    def __len__(self):
        return len(self.stoi)