import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Inizializziamo lo stato nascosto
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Passiamo gli input attraverso la RNN
        out, hn = self.rnn(x, h0)

        # Passiamo l'ultimo stato nascosto attraverso il layer fully connected
        out = self.fc(hn[-1])

        return out
    
    def fit(self, data):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Alleniamo la rete neurale
        num_epochs = 1000
        for epoch in range(num_epochs):
            # Convertiamo i dati in tensori PyTorch
            inputs = torch.from_numpy(data).float()
            targets = torch.from_numpy(data[:, :, 0]).float()

            # Forward pass
            outputs = self(inputs)

            # Calcoliamo la loss
            loss = criterion(outputs, targets)

            # Backward pass e aggiornamento dei pesi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stampa della loss
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')