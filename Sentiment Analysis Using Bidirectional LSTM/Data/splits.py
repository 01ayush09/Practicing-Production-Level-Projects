from sklearn.model_selection import train_test_split

def split_dataset(dataset, test_size, val_size, seed):
    texts = [x["text"] for x in dataset]
    labels = [x["label"] for x in dataset]

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        test_size=test_size + val_size,
        stratify=labels,
        random_state=seed
    )

    val_ratio_adjusted = val_size / (test_size + val_size)

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=1 - val_ratio_adjusted,
        stratify=temp_labels,
        random_state=seed
    )

    def build(texts, labels):
        return [{"text": t, "label": l} for t, l in zip(texts, labels)]

    return build(train_texts, train_labels), build(val_texts, val_labels), build(test_texts, test_labels)