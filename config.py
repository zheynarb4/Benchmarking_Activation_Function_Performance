import architecture

DATASET_URL = "https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset/"
TRAIN_FOLDER = "./data/raf-db-dataset/DATASET/train/"
TEST_FOLDER = "./data/raf-db-dataset/DATASET/test/"
CLASS_NAMES = [
    "surprised",
    "fearful",
    "disgusted",
    "happy",
    "sad",
    "angry",
    "neutral"
]

MODELS = {
    "relu": architecture.ReLU(),
    "selu": architecture.SELU(),
    "sigmoid": architecture.Sigmoid()
}

TRAIN_VAL_SPLIT_RATIO = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
EPOCHS = 20
MODEL_DIR = "./model"