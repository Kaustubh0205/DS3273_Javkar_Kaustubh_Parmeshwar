# 🚢 ShipsNet Classification with PyTorch

## 📚 Project Description

This project is aimed at building a Convolutional Neural Network (CNN) classifier to detect the presence of ships in satellite imagery using the **ShipsNet dataset**.  
We train a deep learning model from scratch on preprocessed satellite images and evaluate its performance using a subset of the dataset.

The key objectives are:
- Implement a modular, well-structured machine learning project.
- Train a custom CNN to classify images into "ship" or "no-ship."
- Create clean interfaces for training and inference to allow easy grading and testing.
- Standardize hyperparameters and data loading across files.

## 🧠 Choice of Model

We chose a **simple Convolutional Neural Network (CNN)** for the following reasons:
- CNNs are highly effective at capturing spatial hierarchies in images.
- The ShipsNet dataset is relatively small (~4000 images), making a lightweight CNN a better fit than heavyweight models like ResNet or EfficientNet.
- Simpler architectures train faster and reduce overfitting on small datasets.

The model architecture (`model.py`) consists of:
- Two convolutional layers followed by ReLU activations and max pooling.
- Two fully connected layers.
- A final output layer with two classes (ship, no-ship).

## 📥 Dataset

We use the  Ships in Satellite Imagery from kaggle.

- we will use the Ships in Satellite imagery Dataset, available at kaggle. The dataset consists of images extracted from Planet satellite imagery collected over the San Francisco Bay and San Pedro Bay areas of California. It includes 4000, 80x80 RGB images labeled with either a "ship" or "no-ship" classification.

Each sample consists of:
- A 3-channel (RGB) 80x80 pixel satellite image.
- A binary label (1 = ship, 0 = no-ship).

### 🔗 Dataset Download

You can download the ShipsNet dataset from:

> [ShipsNet Dataset](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery/data))

**Important**:
1. To use shipsnet.json for training, which was over 300Mb, could not upload it into repositery.
2. I have upload it to my Drive
3. Download the `shipsnet.json` file from the given Drive link https://drive.google.com/drive/folders/1WNFpSBXZIitDZE_w03bzseEz5gVvE937?usp=drive_link.
4. Place `shipsnet.json` inside your data of project directory.


