import torch 
import numpy as np
import os

# INPUT_DIR = '../data/blurred/'
# TARGET_DIR = '../data/processed/'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'blurred')
TARGET_DIR = os.path.join(BASE_DIR, 'data', 'processed')
NUM_CHANNELS = 3
FEATURES = [64, 128, 256, 512]
NUM_EPOCHS = 40
LEARNING_RATE = 0.003
MOMENTUM = 0.9
WEIGHT_DECAY = 0
BATCH_SIZE = 4
TEXT_WEIGHT = 1

# cbam unet
WIDTH = 32
DW_EXPAND = 1
FFN_EXPAND = 2
ENC_BLKS = [1, 1, 1, 2]
MIDDLE_BLK_NUM = 1
DEC_BLKS = [1, 1, 1, 1]
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rescale_and_convert_image(image): 
    return (image.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)

 





