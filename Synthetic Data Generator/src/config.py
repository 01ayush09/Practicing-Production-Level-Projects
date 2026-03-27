
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_length: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
