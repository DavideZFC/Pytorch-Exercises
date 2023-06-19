import torch
import torch.nn as nn
from torch import optim  # For optimizers like SGD, Adam, etc.
from tqdm import tqdm  # For a nice progress bar!

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x, device='cpu'):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with GRU (many-to-one)
class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x, device='cpu'):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with LSTM (many-to-one)
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x, device='cpu'):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out
    

class RecNN:
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
        # model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
        # self.model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes, sequence_length)
        self.model = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length)
        print('Built NN with {} parameters'.format(self.count_parameters()))
        print('Created model:')
        print(self.model)

    def count_parameters(self):
        '''
        hypothesis on number of parameters for LSTM num layers = 1
        4*hidden_size*(input_size+1)
        (seq_len*hidden_size+1)*num_classes
        '''
        for p in self.model.parameters():
            if p.requires_grad:
                print('detected layer with {} parameters'.format(p.numel()))
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self, train_loader, num_epochs=1, learning_rate = 0.005, device='cpu'):

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Train Network
        for epoch in range(num_epochs):
            for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
                # Get data to cuda if possible
                data = data.to(device=device).squeeze(1)
                targets = targets.to(device=device)

                # forward
                scores = self.model(data)
                loss = criterion(scores, targets)

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent update step/adam step
                optimizer.step()
