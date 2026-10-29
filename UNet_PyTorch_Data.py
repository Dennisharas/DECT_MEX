import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BreastCTDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.slices = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        # Example loading logic for binary or numpy files
        slice_path = os.path.join(self.data_dir, self.slices[idx])
        # Load your image array and normalize to [0, 1]
        image = np.load(slice_path).astype(np.float32) 
        image = torch.tensor(image).unsqueeze(0) # Add channel dim: (1, H, W)
        return image, image # Replace second item with your target if doing supervised learning