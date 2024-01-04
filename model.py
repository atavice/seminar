import torch
import torch.nn as nn
import torch.optim as optim


# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input size: 28 * 28, Output size: 128
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)  # Input size: 128, Output size: 64
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)  # Input size: 64, Output size: 10 (number of classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# Instantiate the model
model = SimpleNN()

# Print the model architecture
print(model)
