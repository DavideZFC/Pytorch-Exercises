import numpy as np
import matplotlib.pyplot as plt
import torch
from classes.LSTMpredictor import LSTMpredictor
from misc.plot_traj import plot_traj

# N corresponds to the number of data
N = 1000

# L corresponds to the number of total time steps to be used
L = 200

# T corresponds to the frequency of the wave. The total number of waves can be easily computed as L/T
T = 10

x = np.zeros((N,L))
x[:] = np.array(range(L)) + np.random.normal(0, 10*T, N).reshape(-1,1)

# crucial: if is not everyithing stored as float32 it does not work!
y = np.sin(x/T).astype(np.float32)

plot_traj(x=np.array(range(L)), y=y)

train_input = torch.from_numpy(y[:,:-1])
train_target = torch.from_numpy(y[:,1:])

model = LSTMpredictor(input_size=1, num_layers=2)

model.train('LBFGS', train_input, train_target, n_steps=5, lr=0.5, show_every=10)

####
# Test trained model on given data

n_test = 1
sample_data = train_input[:n_test,:]
future = 1000
pred = model.predict(sample_data, future=future).reshape(-1)

colors = ['C{}'.format(i) for i in range(n_test)]

for i in range(n_test):
    y = sample_data.numpy()[i,:]
    plt.plot(np.array(range(len(y))), y, color = colors[i], label='target {}'.format(i))
    plt.plot(np.array(range(len(pred))), pred, color = colors[i], linestyle='--', label='prediction {}'.format(i))

plt.legend()
plt.show()



