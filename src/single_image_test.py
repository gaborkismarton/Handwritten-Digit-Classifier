import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from train import LeNet5

# Data normalization
# Used resource: https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Load Test Data
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

INDEX = 2

image, label = test_dataset[INDEX]
# Convert image to model input
input_tensor = image.unsqueeze(0)

# Load Model
model = LeNet5()
model.load_state_dict(torch.load("lenet5_mnist.pth"))
# Set to evaluation mode
model.eval()

with torch.no_grad():
    logits = model(input_tensor)
    # Apply Softmax to get probabilities (0.0 to 1.0)
    probs = torch.softmax(logits, dim=1)
    # Get the highest probability and its index
    conf, prediction = torch.max(probs, 1)

# Print a nice table of probabilities
print(f"{'Digit':<10} | {'Probability':<10}")
print("-" * 25)
for i, p in enumerate(probs[0]):
    # Format as a percentage with 4 decimal places
    print(f"{i:<10} | {p.item() * 100:>8.4f}%")

print(f"Actual Label: {label}")
print(f"Model Prediction: {prediction.item()}")
print(f"Confidence: {conf[0] * 100:.2f}%")

# Show the image, remove extra dimension
plt.imshow(image.squeeze(), cmap="gray")
plt.axis("off")
plt.show()
