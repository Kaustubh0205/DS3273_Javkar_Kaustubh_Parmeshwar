# predict.py

import torch
from torchvision import transforms
from PIL import Image
from config import resize_x, resize_y
from model import CNNModel  # So we can load model if needed

# Label mapping (Change if needed)
label_names = ["no_ship", "ship"]  # 0 -> no ship, 1 -> ship

def inferloader(list_of_img_paths):
    """Helper to load and preprocess a batch of images."""
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
    ])

    images = []
    for img_path in list_of_img_paths:
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        images.append(image)
    
    batch = torch.stack(images)  # Create a single batch tensor
    return batch

def cryptic_inf_f(model, list_of_img_paths):
    """Predict labels for a batch of images."""
    model.eval()
    batch = inferloader(list_of_img_paths)

    with torch.no_grad():
        logits = model(batch)
        predictions = torch.argmax(logits, dim=1)

    labels = [label_names[pred.item()] for pred in predictions]
    return labels
