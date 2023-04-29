import numpy as np
import matplotlib.pyplot as plt
import torch
from classes.LSTMpredictor import LSTMpredictor


N = 100
L = 1000
T = 20

x = np.zeros((N,L))
x[:] = np.array(range(L)) + np.random.normal(0, 10*T, N).reshape(-1,1)

# crucial: if is not everyithing stored as float32 it does not work!
y = np.sin(x/T).astype(np.float32)

train_input = torch.from_numpy(y[:,:-1])
train_target = torch.from_numpy(y[:,1:])

model = LSTMpredictor(input_size=1, num_layers=2)

model.train(train_input, train_target)

####
# Test trained model on given data

n_test = 3
sample_data = train_input[:n_test,:]
future = 1000
pred = model.predict(sample_data, future=future)

colors = ['C{}'.format(i) for i in range(n_test)]

for i in range(n_test):
    y = sample_data.numpy()[i,:]
    plt.plot(np.array(range(len(y))), y, color = colors[i])

for i in range(n_test):
    y = pred.numpy()[i,:]
    plt.plot(np.array(range(len(y))), y, color = colors[i], linestyle='--')

plt.show()



