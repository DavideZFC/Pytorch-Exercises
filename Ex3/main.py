import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from data_loader import get_data

X_train, y_train = get_data()

# convert numpy dataset into pytorch tensor
dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))