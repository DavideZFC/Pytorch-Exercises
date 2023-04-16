import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    
    def train_net(self, x, y, n_epochs):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self(torch.Tensor(x).reshape(-1, 1))
            target = torch.Tensor(y).reshape(-1, 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()