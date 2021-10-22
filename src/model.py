import torch.nn as nn

class Generator(nn.Module):

    def __inti__(self):

        self.fc1 = nn.Linear(128 * 128, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        
        self.fc3 = nn.Linear(2048, 4096)
        self.fc4 = nn.Linear(4096, 128 * 128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        
        x = self.relu(self.dropout(self.fc3(x)))
        x = self.relu(self.dropout(self.fc4(x)))
        
        x = self.relu(self.dropout(self.fc5(x)))
        x = self.fc6(x)

        return x

class Descriminator(nn.Module):

    def __inti__(self):

        self.fc1 = nn.Linear(128 * 128, 4096)
        self.fc2 = nn.Linear(4069, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x