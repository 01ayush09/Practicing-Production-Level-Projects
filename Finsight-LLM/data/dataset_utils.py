"""data/dataset_utils.py — Dataset class, prompt builder, collate_fn."""
import json
from functools import partial
import torch
from torch.utils.data import Dataset

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Given a question about a company's "
    "financial performance and relevant context from financial reports, provide "
    "an accurate, step-by-step answer. For numerical questions, show your "
    "reasoning and calculations clearly."
)

def load_jsonl(path):
    out = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

def build_prompt(example, include_output=True):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['instruction']}\n\nContext:\n{example['input']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if include_output:
        prompt += f"{example['output']}<|eot_id|>"
    return prompt

def normalize_answer(text):
    import re, string
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.replace("$","").replace("£","").replace("€","").replace("%","")
    text = re.sub(r"\b(billion)\b","000000000",text)
    text = re.sub(r"\b(million)\b","000000",text)
    text = re.sub(r"\b(thousand)\b","000",text)
    text = text.replace(",","")
    text = text.translate(str.maketrans("","",string.punctuation))
    return " ".join(text.split())

class FinQADataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=2048):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex         = self.examples[idx]
        full_ids   = self._tok(build_prompt(ex, include_output=True))
        prompt_ids = self._tok(build_prompt(ex, include_output=False))
        labels     = full_ids.clone()
        labels[:min(len(prompt_ids), len(full_ids))] = -100
        return {"input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids),
                "labels": labels}

    def _tok(self, text):
        return self.tokenizer(text, truncation=True, max_length=self.max_len,
                              return_tensors="pt")["input_ids"].squeeze(0)

def collate_fn(batch, pad_token_id):
    max_len   = max(x["input_ids"].shape[0] for x in batch)
    bsz       = len(batch)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attn_mask = torch.zeros(bsz, max_len, dtype=torch.long)
    labels    = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        L = item["input_ids"].shape[0]
        input_ids[i,:L] = item["input_ids"]
        attn_mask[i,:L] = item["attention_mask"]
        labels[i,:L]    = item["labels"]
    return {"input_ids":input_ids,"attention_mask":attn_mask,"labels":labels}

def get_collate_fn(pad_token_id):
    return partial(collate_fn, pad_token_id=pad_token_id)
