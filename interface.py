# interface.py

from model import CNNModel as Themodel
from train import main as the_trainer
from predict import cryptic_inf_f as the_predictor
from dataset import ShipsNetDataset as TheDataset
from dataset import ShipsNetLoader as the_dataloader
from config import batchsize as the_batch_size
from config import epochs as total_epochs
