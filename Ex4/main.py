import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment by creating mini batches etc.
from classes.CustomDataset import CustomDataset
from data_loader import get_data
from classes.RNN import RecNN

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# if you want the 28x28 images to be trated as 784 vectors, put False
reshape = False

if reshape:
    input_size = 28
    hidden_size = 5
    sequence_length = 28
    num_epochs = 3
else:
    input_size = 1
    hidden_size = 5
    sequence_length = 784
    num_epochs = 1


# other Hyperparameters
num_layers = 1
num_classes = 10
batch_size = 64





X_train, y_train = get_data(reshape=reshape)
y_train = torch.from_numpy(y_train.to_numpy()).long()


# Definisci la trasformazione da applicare alle immagini
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Crea il dataset personalizzato
dataset = CustomDataset(X_train, y_train, transform=transform)

# Crea il dataloader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

recnet = RecNN(input_size, hidden_size, num_layers, num_classes, sequence_length)
recnet.train(train_loader, num_epochs=num_epochs)



# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, recnet.model)*100:2f}")