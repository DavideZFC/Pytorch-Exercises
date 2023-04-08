import numpy as np
import matplotlib.pyplot as plt
from classes.Net import Net
from classes.Experiment import Experiment
from tqdm import tqdm

# generate test set
N_test = 1000
x_test = np.linspace(0, 2 * np.pi, N_test)
y_test = np.sin(x_test)
exp = Experiment(x_test, y_test)

# generate synthetic data (training set)
N_data = 5000
x = np.random.rand(N_data) * 2 * np.pi
y = np.sin(x)

# epochs
epochs_list = [10, 100, 1000, 10000]

# network structure
layers_list = [[1, 50, 50, 1], [1, 100, 100, 1], [1, 50, 50, 50, 1]]

# store results in a matrix
result_matrix = np.zeros((len(epochs_list), len(layers_list)))

# perform the experiments
for i in tqdm(range(len(epochs_list))):
    for j in range(len(layers_list)):
        net = Net(layers_list[j])
        result_matrix[i,j] = exp.make_experiment(net, x_train=x, y_train=y, epochs=epochs_list[j])

# generate network and train it for given number of epochs
plt.imshow(result_matrix, cmap='gray')
plt.colorbar()
plt.show()