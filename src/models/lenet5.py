import torch.nn as nn


class LeNet5(nn.Module):
    """
    Standard LeNet-5 architecture adapted for the MNIST dataset.

    Reference:
    https://medium.com/@siddheshb008/lenet-5-architecture-explained-3b559cb2d52b

    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # C1: Convolutional Layer, channels: 1 -> 6, 6 filters, 5x5 kernel
            # padding: 2, because MNIST has 28x28 images and LeNet5 requires 32x32
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            # ReLU instead of Tanh activation (used more often)
            nn.ReLU(),
            # S2: Pooling Layer, 2x2 kernel, stride: 2
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3: Convolutional Layer, channels: 6 -> 16, 6 filters, 5x5 kernel
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            # ReLU instead of Tanh activation (used more often)
            nn.ReLU(),
            # S4: Pooling Layer, 2x2 kernel, stride: 2
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C5: Fully Connected Layer, features: 5x5x16 -> 120
            nn.Flatten(),
            nn.Linear(in_features=(5 * 5 * 16), out_features=120),
            # ReLU instead of Tanh activation (used more often)
            nn.ReLU(),
            # F6: Fully Connected Layer, features: 120 -> 84
            nn.Linear(in_features=120, out_features=84),
            # ReLU instead of Tanh activation (used more often)
            nn.ReLU(),
            # Output Layer
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.model(x)
        return x
