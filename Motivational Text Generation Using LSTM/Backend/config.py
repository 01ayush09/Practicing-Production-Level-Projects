
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "quotes.jsonl"
MODEL_PATH = BASE_DIR / "models" / "bilstm_model.pt"

MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
SEQ_LENGTH = 20

EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
CLIP = 5.0

TEMPERATURE = 0.8
TOP_K = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
