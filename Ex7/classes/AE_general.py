import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN autoencoder architecture

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    

class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
        
            
        


class Autoencoder:
    def __init__(self, encoded_space_dim):
        self.encoder = Encoder(encoded_space_dim=encoded_space_dim)
        self.decoder = Decoder(encoded_space_dim=encoded_space_dim)
        self.criterion = nn.MSELoss()

        params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]
        
        # improves a lot with respect to just using SGD
        self.optimizer = optim.Adam(params_to_optimize, lr=0.001, weight_decay=1e-05)

    def train(self, dataloader, epochs=10, short=False):
        # Allenamento della CNN
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, target = data

                target = target.unsqueeze(1)

                self.optimizer.zero_grad()

                encode = self.encoder(inputs)
                output = self.decoder(encode)

                loss = self.criterion(output, target)

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
                
                if short and i>99:
                    break

    def numpy_predict(self, img):
        input = torch.tensor(img)

        encode = self.encoder(input)
        output = self.decoder(encode)

        return output.detach().numpy()

    
