import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    """
    Standard ResNet architecture adapted for the MNIST dataset.

    Reference:
    https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/

    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Main path
        self.main_path = nn.Sequential(
            # Conv1: Convolutional Layer, kernel: 3x3, stride: variable, padding: 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # BatchNorm1: Batch Normalization Layer
            nn.BatchNorm2d(out_channels),
            # ReLU activation function
            nn.ReLU(),
            # Conv2: Convolutional Layer, kernel: 3x3, stride: 1, padding: 1
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # BatchNorm2: Batch Normalization Layer
            nn.BatchNorm2d(out_channels),
        )

        # Skip connection
        self.shortcut = nn.Sequential()

        # Projection Shortcut: Adjusts dimensions and channels to match main path output
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # Conv1x1: Convolutional Layer, kernel: 1x1, stride: variable
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                # BatchNorm: Batch Normalization Layer
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        # Pass input through the main convolutional path
        out = self.main_path(x)
        # Add the original input to the output
        out += self.shortcut(x)
        # ReLU activation function applied after addition
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Prep: Initial preparation block for MNIST (channels: 1 -> 4, 28x28 images)
        self.prep = nn.Sequential(
            # Conv0: Convolutional Layer, channels: 1 -> 4, kernel: 3x3, stride: 1, padding: 1
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False),
            # BatchNorm: Batch Normalization Layer
            nn.BatchNorm2d(4),
            # ReLU activation function
            nn.ReLU(),
        )

        # ResNet Layers: Each contains 2 Residual Blocks
        # Layer 1: channels 4 -> 4, stride: 1 (dimensions unchanged)
        self.layer1 = self.make_layer(in_channels=4, out_channels=4, stride=1)
        # Layer 2: channels 4 -> 8, stride: 2 (halves spatial dimensions)
        self.layer2 = self.make_layer(in_channels=4, out_channels=8, stride=2)
        # Layer 3: channels 8 -> 16, stride: 2 (halves spatial dimensions)
        self.layer3 = self.make_layer(in_channels=8, out_channels=16, stride=2)
        # Layer 4: channels 16 -> 32, stride: 2 (halves spatial dimensions)
        self.layer4 = self.make_layer(in_channels=16, out_channels=32, stride=2)

        # Pool: Adaptive Average Pooling Layer, ensures 1x1 spatial output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Flatten: Flattens features into a vector
        self.flatten = nn.Flatten()
        # Fully Connected Layer, features: 32 -> 10 (one for each digit)
        self.fc = nn.Linear(in_features=32, out_features=10)

    def make_layer(self, in_channels, out_channels, stride):
        # Combines two blocks sequentially to form a full ResNet layer stage
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x):
        # Input passes through the 1-channel preparation block
        x = self.prep(x)
        # Sequential flow through all four residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Pooled down to a 1x1 spatial resolution
        x = self.avgpool(x)
        # Flattened into a 1D vector
        x = self.flatten(x)
        # Linear classifier yields the final digit predictions
        x = self.fc(x)
        return x