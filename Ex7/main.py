from data_loader import get_data
from functions.train_autoencoder import train_autoencoder
from functions.plot_figure import *
import numpy as np

# export dataset
X_train, y_train = get_data()

# train the network on the dataset
net = train_autoencoder(X_train, epochs=2, short=False)

idx = 7
test_sample = X_train[np.newaxis,idx:idx+1,:,:]

test_autoencoder(test_sample, net)
