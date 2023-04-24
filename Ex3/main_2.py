import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from classes.CustomDataset import CustomDataset
from classes.CNN_general import CNN
from data_loader import get_data

X_train, y_train = get_data()
y_train = torch.from_numpy(y_train.to_numpy()).long()


# Definisci la trasformazione da applicare alle immagini
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Crea il dataset personalizzato
dataset = CustomDataset(X_train, y_train, transform=transform)

# Crea il dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Inizializza la CNN e l'ottimizzatore
conv_layers = [32, 64]
fc_layers = [128, 10]
input_dim_x = 28
input_dim_y = 28
net = CNN(conv_layers, fc_layers, input_dim_x, input_dim_y)

# Allenamento della CNN
net.train(dataloader, epochs=1)

# Valutazione della CNN
net.eval_net(dataloader)