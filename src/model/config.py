import torch 
import os

# INPUT_DIR = '../data/blurred/'
# TARGET_DIR = '../data/processed/'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'blurred')
TARGET_DIR = os.path.join(BASE_DIR, 'data', 'processed')
print(INPUT_DIR)
NUM_CHANNELS = 3
FEATURES = [64, 128, 256, 512]
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")