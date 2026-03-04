
import torch.nn as nn
from config import *

class BiLSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = nn.LSTM(
            EMBED_DIM,
            HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(HIDDEN_DIM * 2, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        logits = self.fc(output)
        return logits, hidden
