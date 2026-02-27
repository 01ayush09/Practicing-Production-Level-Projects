import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from utils.seed import set_seed
from utils.stats import compute_max_length
from data.loaders import DatasetLoader
from data.splits import split_dataset
from preprocessing.tokenizer import Tokenizer
from preprocessing.vocabulary import Vocabulary
from data.dataset import SentimentDataset
from dataloader import collate_fn
from models.lstm_model import SentimentLSTM
from training.trainer import Trainer

DEVICE = torch.device("cpu")

def main():
    set_seed(Config.RANDOM_SEED)

    loader = DatasetLoader(Config.FILE_PATH)
    full_dataset = loader.load()

    train_data, val_data, test_data = split_dataset(
        full_dataset,
        Config.TEST_SIZE,
        Config.VAL_SIZE,
        Config.RANDOM_SEED
    )

    tokenizer = Tokenizer()
    max_length = compute_max_length(train_data, tokenizer, Config.PERCENTILE)

    vocab = Vocabulary(Config.MIN_FREQ, Config.MAX_VOCAB_SIZE)
    vocab.build(train_data, tokenizer)

    train_dataset = SentimentDataset(train_data, tokenizer, vocab, max_length)
    val_dataset = SentimentDataset(val_data, tokenizer, vocab, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=lambda x: collate_fn(x, vocab.stoi[vocab.PAD_TOKEN])
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=lambda x: collate_fn(x, vocab.stoi[vocab.PAD_TOKEN])
    )

    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.NUM_CLASSES,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        pad_index=vocab.stoi["<PAD>"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion, DEVICE)

    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc, train_f1 = trainer.train_epoch(train_loader)
        val_loss, val_acc, val_f1 = trainer.evaluate(val_loader)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Model checkpoint saved.")

if __name__ == "__main__":
    main()