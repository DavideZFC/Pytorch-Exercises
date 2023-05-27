import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self, neurons_per_layer):
        super(Net, self).__init__()

        # build list of layers
        self.layers = nn.ModuleList()

        # cycle over the number of neuron per layers adding the respective number of neurons
        for i in range(len(neurons_per_layer)-1):

            # add linear layer of neurons
            self.layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i+1]))

            # if we are not in the first or last layer, add activation function
            if i>0 and i<len(neurons_per_layer)-2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train_net(self, x, y, n_epochs, lr=0.01, batch_size=32):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        N = len(x)
        x_tensor = torch.Tensor(x)
        y_tensor = torch.Tensor(y)

        for epoch in range(n_epochs*int(N/batch_size)):
            optimizer.zero_grad()

            # sample batch
            idx = np.random.choice(N, size=batch_size)

            x_ = x_tensor[idx]
            y_ = y_tensor[idx]

            output = self(x_.reshape(-1, 1))
            target = y_.reshape(-1, 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()