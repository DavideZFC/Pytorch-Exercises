import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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