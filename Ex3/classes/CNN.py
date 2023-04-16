import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, neurons_per_layer, kernel_sizes):
        super(CNN, self).__init__()

        # build list of layers
        self.layers = nn.ModuleList()

        # cicle over the number of neuron per layers adding the respective number of neurons
        for i in range(len(neurons_per_layer)-1):
            
            # add linear layer of neurons
            self.layers.append(nn.Conv2d(in_channels=neurons_per_layer[i], out_channels=neurons_per_layer[i+1], kernel_size=kernel_sizes[i], padding=1))

            # at each step, we apply a maxpooling operator to reduce the image size
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            # if we are not in the first or last layer, add activation function
            if i>0 and i<len(neurons_per_layer)-2:
                self.layers.append(nn.ReLU())

        input = torch.rand(1, neurons_per_layer[0], 28, 28)
                
        # Definiamo i layer lineari
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        
        # Definiamo il layer di dropout
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # Applichiamo i layer convoluzionali
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        
        # Reshape per la linearizzazione
        x = x.view(-1, 64*4*4)
        
        # Applichiamo i layer lineari e il dropout
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x