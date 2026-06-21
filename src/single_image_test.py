import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# Import all three models
from models.simplenet import SimpleNet
from models.lenet5 import LeNet5
from models.resnet import ResNet

# Data normalization
# Used resource: https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Load Test Data
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

INDEX = 67

image, label = test_dataset[INDEX]
# Convert image to model input
input_tensor = image.unsqueeze(0)

# Selecting Model
print("Select a model to test:")
print("1: SimpleNet (Baseline)")
print("2: LeNet5")
print("3: ResNet")
choice = input("Enter 1, 2, or 3: ").strip()

if choice == "1":
    # Load SimpleNet
    model = SimpleNet()
    model.load_state_dict(torch.load("simplenet_mnist.pth"))
    model_name = "SimpleNet"

elif choice == "2":
    # Load LeNet5
    model = LeNet5()
    model.load_state_dict(torch.load("lenet5_mnist.pth"))
    model_name = "LeNet5"

elif choice == "3":
    # Load ResNet
    model = ResNet()
    model.load_state_dict(torch.load("resnet_mnist.pth"))
    model_name = "ResNet"

else:
    print("Invalid choice.")
    exit()

model.eval()

with torch.no_grad():
    logits = model(input_tensor)
    # Apply Softmax to get probabilities (0.0 to 1.0)
    probs = torch.softmax(logits, dim=1)
    # Get the highest probability and its index
    conf, prediction = torch.max(probs, 1)

# Print a nice table of probabilities
print(f"\nResults for {model_name}:")
print(f"{'Digit':<10} | {'Probability':<10}")
print("-" * 25)
for i, p in enumerate(probs[0]):
    # Format as a percentage with 4 decimal places
    print(f"{i:<10} | {p.item() * 100:>8.4f}%")

print(f"\nActual Label: {label}")
print(f"Model Prediction: {prediction.item()}")
print(f"Confidence: {conf[0] * 100:.2f}%")

# Show the image, remove extra dimension
plt.imshow(image.squeeze(), cmap="gray")
plt.axis("off")
plt.show()