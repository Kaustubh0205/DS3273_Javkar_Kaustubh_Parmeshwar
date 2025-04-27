import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from model import CNNModel
from dataset import ShipsNetLoader, ShipsNetDataset
from config import *

class ShipsNetDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.shipsnet = json.load(f)
        self.X = np.array(self.shipsnet['data']) / 255.0
        self.X = self.X.reshape([-1, 3, 80, 80])
        self.Y = np.array(self.shipsnet['labels'])

        # FAST: Only use a smaller subset (optional)
        self.X = self.X[:2000]   # Load only 2000 samples
        self.Y = self.Y[:2000]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        image = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.Y[idx], dtype=torch.long)
        return image, label

def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

def main():
    dataset_path = 'Data\shipsnet.json'
    model_save_path = 'checkpoint\shipsnet_model.pth'

    # Dataset and DataLoader
    train_dataset = ShipsNetDataset(dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # smaller batch size

    # Model, Loss, Optimizer
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    train(model, train_loader, criterion, optimizer, num_epochs=10)

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully at {model_save_path}")

if __name__ == "__main__":
    main()
