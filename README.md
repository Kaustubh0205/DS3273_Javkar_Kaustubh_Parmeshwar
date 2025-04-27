Project Description
This project is aimed at building a Convolutional Neural Network (CNN) classifier to detect the presence of ships in satellite imagery using the ShipsNet dataset.
We train a deep learning model from scratch on preprocessed satellite images and evaluate its performance using a subset of the dataset.

The key objectives are:

Implement a modular, well-structured machine learning project.

Train a custom CNN to classify images into "ship" or "no-ship."

Create clean interfaces for training and inference to allow easy grading and testing.

Standardize hyperparameters and data loading across files.

🧠 Choice of Model
We chose a simple Convolutional Neural Network (CNN) for the following reasons:

CNNs are highly effective at capturing spatial hierarchies in images.

The ShipsNet dataset is relatively small (~4000 images), making a lightweight CNN a better fit than heavyweight models like ResNet or EfficientNet.

Simpler architectures train faster and reduce overfitting on small datasets.

The model architecture (model.py) consists of:

Two convolutional layers followed by ReLU activations and max pooling.

Two fully connected layers.

A final output layer with two classes (ship, no-ship).

📥 Dataset
We use the ShipsNet Dataset from Planet Labs.

Each sample consists of:

A 3-channel (RGB) 80x80 pixel satellite image.

A binary label (1 = ship, 0 = no-ship).

🔗 Dataset Download
You can download the ShipsNet dataset from:

https://github.com/planetlabs/shipsnet

Instructions:

Go to the repository above.

Download the shipsnet.json file.

Place shipsnet.json inside your project directory (same level as train.py, dataset.py, etc.).

Or you can directly download using:

bash
Copy
Edit
wget https://raw.githubusercontent.com/planetlabs/shipsnet/master/shipsnet.json
🛠 Installation
Make sure you have Python 3.7+ installed.

Then install required libraries:

bash
Copy
Edit
pip install torch torchvision numpy
(Optional) You can also use the provided requirements.txt if available:

bash
Copy
Edit
pip install -r requirements.txt
⚙️ How to Run the Project
1. Train the Model
bash
Copy
Edit
python train.py
This script loads the dataset, initializes the model, and trains it for a specified number of epochs (default: 10).

The trained model is saved as shipsnet_model.pth inside the project directory.

2. Predict on New Data
bash
Copy
Edit
python predict.py
This script loads the trained model and makes predictions on new images inside the /data directory.

You must place a few test images into the data/ folder for inference.

3. Interface Check (for Grading)
bash
Copy
Edit
python interface.py
This will import everything according to a standardized naming convention.

No errors during this run = your submission will be gradeable automatically.

📁 Project Structure
bash
Copy
Edit
.
├── config.py             # Hyperparameters like batch size, number of epochs, resize info
├── dataset.py            # Custom Dataset class and DataLoader
├── model.py              # CNN Model Architecture
├── train.py              # Training function
├── predict.py            # Inference function
├── interface.py          # Standardized imports for grading
├── shipsnet.json         # Training dataset
├── data/                 # Folder containing images for prediction
└── README.md             # (This file)
🔧 Configuration Details
All hyperparameters are centralized in config.py for easier experimentation:

python
Copy
Edit
# config.py
batchsize = 32
epochs = 10
resize_x = 80
resize_y = 80
input_channels = 3
Other scripts like dataset.py, train.py, and predict.py import settings from config.py to keep everything synchronized.# DS3273_Javkar_Kaustubh_Parmeshwar
