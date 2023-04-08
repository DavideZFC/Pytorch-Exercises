import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, neurons_per_layer):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(neurons_per_layer)-1):
            self.layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i+1]))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x