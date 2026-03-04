
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TextDataset
from model import BiLSTMTextGenerator
from config import *

def train():
    print("Loading dataset...")
    dataset = TextDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BiLSTMTextGenerator(len(dataset.itos)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "stoi": dataset.stoi,
        "itos": dataset.itos
    }, MODEL_PATH)

    print("Model saved at:", MODEL_PATH)

if __name__ == "__main__":
    train()
