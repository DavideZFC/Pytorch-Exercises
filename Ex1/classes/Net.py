import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    '''
    class to build a simple pytorch neural network with one single
    hidden layer of 100 neurons and relu activation function
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_net(self, x, y, n_epochs, lr=0.001, batch_size=32, show_every=1000):
        '''
        choose the MSE criterion as training loss
        for the optimizer, use the simple stochastic gradient descent
        '''
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)
        N = x.shape[0]


        x_tensor = torch.Tensor(x)
        y_tensor = torch.Tensor(y)

        # training cycle for each epoch
        for epoch in range(n_epochs*int(N/batch_size)):
            if epoch % show_every == 1:
                print('Epoch: {} loss: {}'.format(epoch, loss.item()))

            optimizer.zero_grad()

            # sample batch
            idx = np.random.choice(N, size=batch_size)

            x_ = x_tensor[idx]
            y_ = y_tensor[idx]

            output = self(x_.reshape(-1, 1))
            target = torch.Tensor(y_).reshape(-1, 1)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()