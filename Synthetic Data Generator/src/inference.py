
import torch

class InferencePipeline:
    def __init__(self, model, tokenizer, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def generate(self, prompt, max_length=100, temperature=1.0, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
