# Define a new neural network architecture
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

        # Add batch normalization layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)

        # Add dropout layers
        self.dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.relu(self.bn4(self.dropout(self.fc4(x))))
        x = self.fc5(x)
        return x
