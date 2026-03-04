
import torch
import re
from model import BiLSTMTextGenerator
from utils import tokenize, top_k_sampling
from config import *

class MotivationEngine:
    def __init__(self):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        self.stoi = checkpoint["stoi"]
        self.itos = checkpoint["itos"]

        self.model = BiLSTMTextGenerator(len(self.itos)).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def detect_intent(self, text):
        text = text.lower()
        if "quit" in text:
            return "overcome quitting because"
        if "tired" in text:
            return "push through fatigue because"
        return "stay motivated because"

    def generate(self, user_input, max_len=40):
        prompt = self.detect_intent(user_input)
        tokens = tokenize(prompt)
        input_ids = [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]
        input_tensor = torch.tensor([input_ids]).to(DEVICE)

        hidden = None
        generated = tokens.copy()

        for _ in range(max_len):
            logits, hidden = self.model(input_tensor, hidden)
            next_logits = logits[:, -1, :]
            next_id = top_k_sampling(next_logits.squeeze(), TOP_K, TEMPERATURE)
            word = self.itos[next_id.item()]
            generated.append(word)
            input_tensor = torch.tensor([[next_id.item()]]).to(DEVICE)

        return re.sub(r'\s+', ' ', " ".join(generated)).capitalize()
