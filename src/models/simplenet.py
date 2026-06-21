import torch.nn as nn

class SimpleNet(nn.Module):

    """
    A basic MLP

    Reference:
    https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(

            # First hidden layer: 784 -> 256
            nn.Linear(in_features=28 * 28, out_features=256),
            nn.Sigmoid(),
            
            # Second hidden layer: 256 -> 128
            nn.Linear(in_features=256, out_features=128),
            nn.Sigmoid(),
            
            # Output layer: 128 -> 10 feature
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x