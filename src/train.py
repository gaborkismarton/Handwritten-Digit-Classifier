import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# LeNet5 model
# Used resource: https://medium.com/@siddheshb008/lenet-5-architecture-explained-3b559cb2d52b
class LeNet5(nn.Module):
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

# RESNet model
# Used resource: https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        # Main path
        self.main_path = nn.Sequential(
            # First convolutional layer of the block, padding ensures that the dimensions do not change
            # We deifen the stride to be 1, but it can be increased to reduce run time
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # Normalizing the output after conv so the training is faster
            nn.BatchNorm2d(out_channels),
            # ReLu does what is does
            nn.ReLU(),
            # Second convolutional layer, same as before just with a fixed stride of 1 so the dim is not modified
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Skip connection
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
             # If dimensions or channels change we need to match them up again so we do a 1 x 1 conv
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
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
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        # Each layer contains 2 Residual Blocks, we halve the dimension and double the channels after the first
        self.layer1 = self._make_layer(in_channels=4, out_channels=4, stride=1)
        self.layer2 = self._make_layer(in_channels=4, out_channels=8, stride=2)
        self.layer3 = self._make_layer(in_channels=8, out_channels=16, stride=2)
        self.layer4 = self._make_layer(in_channels=16, out_channels=32, stride=2)

        # We make sure the output is 1X1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #Then flatten it into a vector
        self.flatten = nn.Flatten()
        # Then we transform the 32 output from layer 4 into 10, one for each digit
        self.fc = nn.Linear(in_features=32, out_features=10)

    def _make_layer(self, in_channels, out_channels, stride):
        # Combines two blocks to form a full ResNet layer block
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, stride = 1)
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

LEARNING_RATE = 0.001
NUM_EPOCHS = 5


def train():
    # Data normalization
    # Used resource: https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    model = LeNet5()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Save the weights
    model_filename = "lenet5_mnist.pth"
    torch.save(model.state_dict(), model_filename)
    print("Model saved as", model_filename)


if __name__ == "__main__":
    train()
