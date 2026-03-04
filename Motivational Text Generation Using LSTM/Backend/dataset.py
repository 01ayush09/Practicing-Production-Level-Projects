
import json
import torch
from torch.utils.data import Dataset
from collections import Counter
from utils import tokenize
from config import *

class TextDataset(Dataset):
    def __init__(self):
        self.texts = self.load_data()
        self.tokens = []
        for text in self.texts:
            self.tokens.extend(tokenize(text))
        self.build_vocab()
        self.data = self.encode_tokens()

    def load_data(self):
        texts = []
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
        return texts

    def build_vocab(self):
        counter = Counter(self.tokens)
        most_common = counter.most_common(MAX_VOCAB_SIZE)
        self.itos = ["<pad>", "<unk>"] + [w for w, freq in most_common if freq >= MIN_FREQ]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode_tokens(self):
        encoded = [self.stoi.get(token, self.stoi["<unk>"]) for token in self.tokens]
        sequences = []
        for i in range(0, len(encoded) - SEQ_LENGTH):
            seq = encoded[i:i+SEQ_LENGTH]
            target = encoded[i+1:i+SEQ_LENGTH+1]
            sequences.append((seq, target))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq), torch.tensor(target)
