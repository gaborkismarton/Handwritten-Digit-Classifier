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
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {running_loss / len(train_loader):.2f}"
        )

    # Save the weights
    model_filename = "lenet5_mnist.pth"
    torch.save(model.state_dict(), model_filename)
    print("Model saved as", model_filename)


if __name__ == "__main__":
    train()
