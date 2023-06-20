import torch
import torch.nn as nn
import numpy as np


class LSTMpredictor(nn.Module):
    def __init__(self, input_size, hidden_size=51, num_layers=1):
        super(LSTMpredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTMCell(input_size, hidden_size, num_layers)

        # one since we are doing regression
        self.fc = nn.Linear(hidden_size, 1)
        self.count_parameters()

    def count_parameters(self):
        '''
        hypothesis on number of parameters for LSTM num layers = 1
        4*hidden_size*(input_size+1)
        (seq_len*hidden_size+1)*num_classes
        '''
        for p in self.parameters():
            if p.requires_grad:
                print('detected layer with {} parameters'.format(p.numel()))
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
    
    def train(self, mode, train_input, train_target, n_steps = 2, lr=0.001, batch_size=32, show_every=100):
        criterion = nn.MSELoss()

        if mode == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=0.8)

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

        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
            N = train_input.shape[0]

            # training cycle for each epoch
            for epoch in range(n_steps*int(N/batch_size)):
                if epoch % show_every == 1:
                    print('Batch: {} loss: {}'.format(epoch, loss.item()))

                optimizer.zero_grad()

                # sample batch
                idx = np.random.choice(N, size=batch_size)

                x_ = train_input[idx]
                y_ = train_target[idx]

                output = self(x_.reshape(-1, 1))
                target = torch.Tensor(y_).reshape(-1, 1)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()



        with torch.no_grad():
            pred = self(train_input)
            loss = criterion(pred, train_target)
            print("test loss", loss.item())


    def predict(self, seq, future=1000):
        pred = self(seq, future)
        return pred.detach()