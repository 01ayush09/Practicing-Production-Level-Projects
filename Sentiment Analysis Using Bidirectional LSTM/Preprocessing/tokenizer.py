import re

class Tokenizer:
    def __init__(self):
        self.pattern = re.compile(r"\w+|[!?.,]")

    def tokenize(self, text):
        text = text.lower().strip()
        return self.pattern.findall(text)