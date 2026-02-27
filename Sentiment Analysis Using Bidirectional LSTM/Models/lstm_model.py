import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 output_dim, num_layers, dropout, pad_index):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        _, (hidden, _) = self.lstm(packed)

        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]
        hidden = torch.cat((hidden_forward, hidden_backward), dim=1)

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)

        return logits