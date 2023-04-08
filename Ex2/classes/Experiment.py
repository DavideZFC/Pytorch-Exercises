import torch
import numpy as np

class Experiment:
    def __init__(self, x_test, y_test):

        self.x_test = x_test
        self.y_test = y_test
        self.N = len(y_test)

    def make_experiment(self, net, x_train, y_train, epochs):

        net.train_net(x_train, y_train, n_epochs=epochs)
        y_pred = net(torch.Tensor(self.x_test).reshape(-1, 1)).detach().numpy().reshape(self.N)

        return np.mean((y_pred-self.y_test)**2)

