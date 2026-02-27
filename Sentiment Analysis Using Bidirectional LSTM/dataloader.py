import torch

def collate_fn(batch, pad_index):
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)

    input_ids = [item["input_ids"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch])

    padded = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_index
    )

    return {
        "input_ids": padded,
        "labels": labels,
        "lengths": lengths
    }