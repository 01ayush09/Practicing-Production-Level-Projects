import os
import json
import pandas as pd

class LabelEncoder:
    def __init__(self):
        self.label_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }

    def encode(self, label):
        return self.label_map[label.strip().lower()]


class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.label_encoder = LabelEncoder()

    def load(self):
        ext = os.path.splitext(self.file_path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(self.file_path)
            records = df.to_dict("records")

        elif ext == ".json":
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data["data"]
        else:
            raise ValueError("Unsupported format")

        dataset = []
        for item in records:
            dataset.append({
                "text": item["text"],
                "label": self.label_encoder.encode(item["sentiment"])
            })

        return dataset