import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.lenet5 import LeNet5
from models.resnet import ResNet

LEARNING_RATE = 0.001
NUM_EPOCHS = 5
BATCH_SIZE = 100


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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Models, Loss, and Optimizer
    models_to_train = [
        (LeNet5(), "lenet5_mnist.pth", "LeNet5"),
        (ResNet(), "resnet_mnist.pth", "ResNet"),
    ]

    for model, model_filename, model_name in models_to_train:
        print(f"\nStarting training for {model_name}")

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
                    print(
                        f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}"
                    )
                    running_loss = 0.0

        # Save the weights
        torch.save(model.state_dict(), model_filename)
        print("Model saved as", model_filename)


if __name__ == "__main__":
    train()
