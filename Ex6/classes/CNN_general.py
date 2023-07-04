import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN architecture

class Net(nn.Module):
    def __init__(self, conv_layers, fc_layers, input_dim_x, input_dim_y):
        super(Net, self).__init__()

        self.conv_layers_list = nn.ModuleList()
        self.conv_layers_list.append(nn.Conv2d(1, conv_layers[0], 3, padding=1))

        # build list of convolutional layers
        for i in range(len(conv_layers)-1):
            self.conv_layers_list.append(nn.Conv2d(conv_layers[i], conv_layers[i+1], 3, padding=1))

        self.pool = nn.MaxPool2d(2, 2)

        # compute dimension of the input after the poolings
        for i in range(len(conv_layers)):
            input_dim_x /= 2
            input_dim_y /= 2

        # make int
        input_dim_x = int(input_dim_x)
        input_dim_y = int(input_dim_y)

        # start adding the fc layers
        self.fc_layers_list = nn.ModuleList()
        self.fc_layers_list.append(nn.Linear(conv_layers[-1] * input_dim_x * input_dim_y, fc_layers[0]))

        # collect all the fully connected layers
        for i in range(len(fc_layers)-1):
            self.fc_layers_list.append(nn.Linear(fc_layers[i], fc_layers[i+1]))
            
        

    def forward(self, x):

        # convolutional part
        for lay in self.conv_layers_list:
            x = self.pool(nn.functional.relu(lay(x)))
        
        # flatten layer
        x = torch.flatten(x, 1)

        # fully connected part
        for i,lay in enumerate(self.fc_layers_list):
            if i<len(self.fc_layers_list)-1:
                x = nn.functional.relu(lay(x))
            else:
                x = lay(x)

        # here I apply relu also to the last layer, not the best thing to do
        return x
    
class CNN:
    def __init__(self, conv_layers, fc_layers, input_dim_x, input_dim_y):
        self.net = Net(conv_layers, fc_layers, input_dim_x, input_dim_y)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def train(self, dataloader, epochs=10):
        # Allenamento della CNN
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

    def eval_net(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                print(labels.shape)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy on the dataset: %d %%' % (100 * correct / total))

    def numpy_predict(self, X):
        return self.net(torch.from_numpy(X)).detach().numpy()
    
    def adversarial_attack(self, X, label, iter=1):
        X = torch.from_numpy(X)
        X.requires_grad = True

        label = torch.tensor(label).unsqueeze(0)

        dark_criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD([X], lr=0.001, momentum=0.9)

        for i in range(iter):
            optimizer.zero_grad()
            outputs = self.net(X)
            loss = dark_criterion(outputs, label)
            loss.backward()
            self.optimizer.step()

            print('[%d] loss: %.3f' % (i + 1, loss.item()))

            if loss.item() < 0.5:
                return X.detach().numpy()


    
