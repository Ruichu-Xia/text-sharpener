import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class LocalImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, indices, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.indices = indices 
        self.transform = transform

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image_name = f"Image_{self.indices[idx]}.png"
        input_path = os.path.join(self.input_dir, image_name)
        target_path = os.path.join(self.target_dir, image_name)

        input_image = self._load_image(input_path)
        target_image = self._load_image(target_path)

        input_image = torch.tensor(np.array(input_image, dtype=np.float32))
        target_image = torch.tensor(np.array(target_image, dtype=np.float32))

        if input_image.dim() == 2:
            input_image = input_image.unsqueeze(0) 
            target_image = target_image.unsqueeze(0) 
        else:
            input_image =  input_image.permute(2, 0, 1) 
            target_image = target_image.permute(2, 0, 1) 
        
        input_image = input_image / 255.0 
        target_image = target_image / 255.0

        return input_image, target_image
    
    def _load_image(self, image_path):
        image = Image.open(image_path)
        return image


def get_split_indices(num_images, val_ratio=0.2, test_ratio=0.1, seed=42): 
    indices = np.arange(0, num_images)
    train_val_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=seed)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_ratio / (1 - test_ratio), random_state=seed)
    return train_indices, val_indices, test_indices