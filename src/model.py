import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(256 * 256, 150 * 150, bias=False)
        self.conv_block = nn.Sequential(nn.Conv2d(1, 3, kernel_size=7),
                                        nn.ReLU(),
        
                                        nn.Conv2d(3, 5, kernel_size=5),
                                        nn.ReLU(),
        
                                        nn.Conv2d(5, 7, kernel_size=5),
                                        nn.ReLU(),
        
                                        nn.Conv2d(7, 5, kernel_size=7),
                                        nn.ReLU(),
        
                                        nn.Conv2d(5, 3, kernel_size=5),
                                        nn.ReLU(),
        
                                        nn.Conv2d(3, 1, kernel_size=5),
                                        nn.Tanh())        

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.flatten()
        x = self.relu(self.fc1(x))

        x = x.reshape(-1, 1, 150, 150)
        x = self.conv_block(x)

        return x


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(128 * 128, 4096)
        self.fc2 = nn.Linear(4069, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.flatten()

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x