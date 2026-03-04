
import torch
import torch.nn.functional as F

def tokenize(text):
    return text.lower().split()

def top_k_sampling(logits, k=40, temperature=1.0):
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    idx = torch.multinomial(probs, 1)
    return top_k_indices[idx]
