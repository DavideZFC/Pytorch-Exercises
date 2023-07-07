import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from classes.CustomDataset import CustomDataset
from classes.AE_general import Autoencoder
from data_loader import get_data


def train_autoencoder(X_train):


    # Definisci la trasformazione da applicare alle immagini
    # Ogni canale viene normalizzato in modo da avere media e deviazione standard date
    # Qui c'è un solo canale e quindi la tupla
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # alzare la media peggiora il training, ma solo all'inizio, poi la rete si adatta

    # Crea il dataset personalizzato
    dataset = CustomDataset(X_train, X_train, transform=transform)

    # Crea il dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Inizializza la CNN e l'ottimizzatore
    encode_dim = 10
    net = Autoencoder(encoded_space_dim=encode_dim)

    # Allenamento della CNN
    net.train(dataloader, epochs=2)

    # Valutazione della CNN
    # net.eval_net(dataloader)

    return net