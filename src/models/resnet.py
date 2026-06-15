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
            # First convolutional layer of the block, padding ensures that the dimensions do not change
            # We deifen the stride to be 1, but it can be increased to reduce run time
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            # Normalizing the output after conv so the training is faster
            nn.BatchNorm2d(out_channels),
            # ReLu does what is does
            nn.ReLU(),
            # Second convolutional layer, same as before just with a fixed stride of 1 so the dim is not modified
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Skip connection
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            # If dimensions or channels change we need to match them up again so we do a 1 x 1 conv
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    # We define how the data flows
    def forward(self, x):

        # Puts X through the main path consisting of the convs
        out = self.main_path(x)
        # Add the original input back to the output while also modifying dimension if needed
        out += self.shortcut(x)
        # ReLu does something
        out = torch.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Setting up for the MNIST (1 channel, 28x28)
        self.prep = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )

        # Each layer contains 2 Residual Blocks, we halve the dimension and double the channels after the first
        self.layer1 = self._make_layer(in_channels=2, out_channels=2, stride=1)
        self.layer2 = self._make_layer(in_channels=2, out_channels=4, stride=2)
        self.layer3 = self._make_layer(in_channels=4, out_channels=8, stride=2)
        self.layer4 = self._make_layer(in_channels=8, out_channels=16, stride=2)

        # We make sure the output is 1X1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Then flatten it into a vector
        self.flatten = nn.Flatten()
        # Then we transform the 16 output from layer 4 into 10, one for each digit
        self.fc = nn.Linear(in_features=16, out_features=10)

    def _make_layer(self, in_channels, out_channels, stride):
        # Combines two blocks to form a full ResNet layer block
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, stride=1),
        )

    # An image x enters the network, passes through the 1-channel preparation block,
    # flows sequentially through all four residual stages,
    # is pooled down to 1x1,
    # flattened into a vector,
    # and finally pushed through the linear classifier to yield the predictions.
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
