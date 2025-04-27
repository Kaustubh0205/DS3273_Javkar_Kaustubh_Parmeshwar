# dataset.py

import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from config import resize_x, resize_y, input_channels, batchsize

class ShipsNetDataset(Dataset):  
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            shipsnet = json.load(f)
        self.X = np.array(shipsnet['data']) / 255.0
        self.X = self.X.reshape([-1, input_channels, resize_x, resize_y])
        self.Y = np.array(shipsnet['labels'])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        image = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.Y[idx], dtype=torch.long)
        return image, label

def ShipsNetLoader(json_file):
    dataset = ShipsNetDataset(json_file)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader
