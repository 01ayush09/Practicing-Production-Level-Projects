import numpy as np

def compute_max_length(dataset, tokenizer, percentile):
    lengths = [
        len(tokenizer.tokenize(sample["text"]))
        for sample in dataset
    ]
    return int(np.percentile(lengths, percentile))