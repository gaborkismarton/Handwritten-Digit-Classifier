import pandas as pd
import matplotlib.pyplot as plt

# Load the training history
df = pd.read_csv("training_history.csv")

# Models present in the CSV
models = df["Model"].unique()

# Training Loss vs Epoch
plt.figure(figsize=(12, 8))

for model in models:
    model_data = df[df["Model"] == model]
    plt.plot(
        model_data["Epoch"],
        model_data["Loss"],
        marker="o",
        linewidth=2,
        label=model
    )

plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("training_loss.png", dpi=300)
plt.show()


# Test Accuracy vs Epoch
plt.figure(figsize=(12, 8))

for model in models:
    model_data = df[df["Model"] == model]
    plt.plot(
        model_data["Epoch"],
        model_data["Accuracy (%)"],
        marker="o",
        linewidth=2,
        label=model
    )

plt.title("Test Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("test_accuracy.png", dpi=300)
plt.show()


# Training Time vs Epoch
plt.figure(figsize=(12, 8))

for model in models:
    model_data = df[df["Model"] == model]
    plt.plot(
        model_data["Epoch"],
        model_data["Time (s)"],
        marker="o",
        linewidth=2,
        label=model
    )

plt.title("Training Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("training_time.png", dpi=300)
plt.show()


# Final comparison table
print("\nFinal Model Comparison")

comparison = (
    df.groupby("Model")
      .agg(
          Final_Loss=("Loss", "last"),
          Final_Accuracy=("Accuracy (%)", "last"),
          Average_Epoch_Time=("Time (s)", "mean")
      )
      .round(3)
)

print(comparison)

print("\nSaved:")
print("training_loss.png")
print("test_accuracy.png")
print("training_time.png")
