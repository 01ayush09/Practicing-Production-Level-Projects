class Config:
    FILE_PATH = "C:/Users/ayush/OneDrive/Desktop/data/sentiment_data.json"

    RANDOM_SEED = 42

    TEST_SIZE = 0.1
    VAL_SIZE = 0.1

    MIN_FREQ = 2
    MAX_VOCAB_SIZE = 20000

    BATCH_SIZE = 8
    NUM_WORKERS = 0

    PERCENTILE = 95

    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    NUM_CLASSES = 3
    LR = 0.001
    EPOCHS = 10