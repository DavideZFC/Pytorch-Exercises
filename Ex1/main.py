import numpy as np
import matplotlib.pyplot as plt
from classes.Net import Net
import torch

# number of training data
N_data = 5000

# noise to add to the data
sigma = 0.2

# generate synthetic data (training set)
x = np.random.rand(N_data) * 2 * np.pi
y = np.sin(x) + np.random.normal(0, scale=sigma, size=N_data)

# generate network and train it for given number of epochs
net = Net()
epochs = 300
net.train_net(x, y, n_epochs=epochs)

# generate test set
N_test = 100
x_test = np.linspace(0, 2 * np.pi, N_test)

# compute prediction on test set
y_test = np.sin(x_test)
y_pred = net(torch.Tensor(x_test).reshape(-1, 1)).detach().numpy().reshape(N_test)

# show the results
print('MSE: '+str(np.mean((y_test-y_pred)**2)))
plt.plot(x_test, y_pred, label='predicted curve')
plt.plot(x_test, y_test, 'o', label='real values')
plt.show()