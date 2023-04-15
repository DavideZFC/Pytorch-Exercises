import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Definiamo i layer convoluzionali
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
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