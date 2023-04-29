import torch
import torch.nn as nn
from torch import optim  # For optimizers like SGD, Adam, etc.


class LSTMpredictor(nn.Module):
    def __init__(self, input_size, hidden_size=51, num_layers=1):
        super(LSTMpredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTMCell(input_size, hidden_size, num_layers)

        # one since we are doing regression
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, future=0):
        outputs = []

        h0 = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float32)
        c0 = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float32)

        for input_t in x.split(1, dim=1):
            h0, c0 = self.lstm(input_t, (h0, c0))
            output = self.fc(h0)
            outputs.append(output)

        for i in range(future):
            h0, c0 = self.lstm(output, (h0, c0))
            output = self.fc(h0)
            outputs.append(output)

        return torch.cat(outputs, dim=1)
    
    def train(self, train_input, train_target):
        criterion = nn.MSELoss()
        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.8)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.005)

        n_steps = 5


        for i in range(n_steps):

            print('----------- {} steps done -------------'.format(i))

            def closure():
                optimizer.zero_grad()
                out = self(train_input)
                loss = criterion(out, train_target)

                loss.backward()
                print("train_loss",loss.item())
                return loss
            
            optimizer.step(closure=closure)


        with torch.no_grad():
            pred = self(train_input)
            loss = criterion(pred, train_target)
            print("test loss", loss.item())

    def predict(self, seq, future=1000):
        pred = self(seq, future)
        return pred.detach()