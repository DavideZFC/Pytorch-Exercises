import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from classes.CustomDataset import CustomDataset
from classes.CNN_general import CNN
from data_loader import get_data
from functions.train_on_data import train_on_data
from functions.plot_figure import plot_figure
import numpy as np

X_train, y_train = get_data()


net = train_on_data(X_train, y_train)

idx = 2
plot_figure(X_train, idx)
print(net.numpy_predict(X_train[np.newaxis,idx:idx+1,:,:]))