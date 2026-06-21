import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.lenet5 import LeNet5
from models.resnet import ResNet
from models.simplenet import SimpleNet

LEARNING_RATE = 0.001
NUM_EPOCHS = 15
BATCH_SIZE = 256


def train():
    # Data normalization
    # Used resource: https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST Training Data
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Load MNIST Test Data
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Initialize Models
    models_to_train = [
        (SimpleNet(), "simplenet_mnist.pth", "SimpleNet"),
        (LeNet5(), "lenet5_mnist.pth", "LeNet5"),
        (ResNet(), "resnet_mnist.pth", "ResNet"),
    ]

    # List to store metrics for Pandas DataFrame
    all_history_data = []

    for model, model_filename, model_name in models_to_train:

        print(f"\nStarting training for {model_name}")

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training Loop
        for epoch in range(NUM_EPOCHS):

            # Start measuring the time required for this epoch
            start_time = time.time()

            # Set the model to training mode
            model.train()

            # Stores the total loss accumulated over all batches
            running_loss = 0.0

            # Iterates over the dataset in small batches
            for images, labels in train_loader:

                # Clear old gradients from the previous step
                optimizer.zero_grad()

                # Pass the images through the network to get predictions
                outputs = model(images)

                # Compute the loss
                loss = loss_func(outputs, labels)

                # Calculate the gradients of the loss with respect to the model parameters
                loss.backward()

                # Update the model weights using the calculated gradients
                optimizer.step()

                # Add the current batch loss to the running total
                running_loss += loss.item()

            # Compute the average training loss for this epoch
            avg_loss = running_loss / len(train_loader)

            # Switch the model to evaluation mode
            model.eval()

            # Variables used to calculate classification accuracy
            correct = 0
            total = 0

            # Disable gradient calculations during evaluation
            with torch.no_grad():
                 # Iterate over the test dataset
                for test_images, test_labels in test_loader:

                    # Pass the test images through the network
                    test_outputs = model(test_images)

                    # Get the predicted class for each image
                    _, predicted = torch.max(test_outputs, 1)

                    # Count the total number of test samples
                    total += test_labels.size(0)

                    # Count how many predictions were correct
                    correct += (predicted == test_labels).sum().item()

            # Compute the test accuracy as a percentage
            accuracy = 100 * correct / total

            # Calculate the total time required for this epoch
            epoch_time = time.time() - start_time

            # Store the collected metrics for later analysis and plotting
            all_history_data.append({
                "Model": model_name,
                "Epoch": epoch + 1,
                "Loss": round(avg_loss, 4),
                "Accuracy (%)": round(accuracy, 2),
                "Time (s)": round(epoch_time, 2)
            })

            # Print the training statistics for this epoch
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
                f"Loss: {avg_loss:.4f} | "
                f"Accuracy: {accuracy:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )

        # Save the weights
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    # Export history to CSV
    df = pd.DataFrame(all_history_data)
    df.to_csv("training_history.csv", index=False)

    print("\nTraining history saved to 'training_history.csv'")


if __name__ == "__main__":
    train()