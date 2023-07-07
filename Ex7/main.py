from data_loader import get_data
from functions.train_autoencoder import train_autoencoder
import numpy as np

# export dataset
X_train, y_train = get_data()

# train the network on the dataset
net = train_autoencoder(X_train)
