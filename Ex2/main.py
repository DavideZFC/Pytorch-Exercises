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
epochs = 20
n_max = 10

# network structure
layers_list = [[1, 2*n, 2*n, 1] for n in range(1,n_max)]

# store results of the training
result = np.zeros(10)


# perform the experiments
for j in tqdm(range(len(layers_list))):
    net = Net(layers_list[j])
    result[j] = exp.make_experiment(net, x_train=x, y_train=y, epochs=epochs)

plt.plot(result)
plt.show()

