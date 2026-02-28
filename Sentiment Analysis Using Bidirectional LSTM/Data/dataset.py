import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, vocab, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = self.tokenizer.tokenize(sample["text"])
        tokens = tokens[:self.max_length]
        input_ids = self.vocab.numericalize(tokens)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "length": len(input_ids)
        }