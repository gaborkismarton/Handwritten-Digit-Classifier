import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from train import LeNet5

# Load the trained model
model = LeNet5()
model.load_state_dict(torch.load("lenet5_mnist.pth"))
model.eval()

# Data normalization
# Used resource: https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def predict_digit(image_dict):
    # Ensure the user actually drew something
    if not image_dict or not image_dict.get("composite"):
        return "Please draw a number in the canvas first."

    img = image_dict["composite"]

    # Ensure it's in grayscale
    img = img.convert("L")

    # Invert the image: MNIST expects a black background (0) and white digits (255)
    import PIL.ImageOps

    img = PIL.ImageOps.invert(img)

    # Resize to the 28x28 dimension expected by the network
    img = img.resize((28, 28), Image.Resampling.NEAREST)

    img.save("debug_input.png")

    # Convert image to model input tensor
    input_tensor = transform(img).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, prediction = torch.max(probs, 1)

    pred_val = prediction.item()
    conf_val = conf.item() * 100

    # Format the output message to the user
    result_text = f"### Detected Number: **{pred_val}**\n"
    result_text += f"**Confidence:** {conf_val:.2f}%\n\n"
    result_text += "#### Detailed Probabilities:\n"
    for i, p in enumerate(probs[0]):
        result_text += f"* **Digit {i}:** {p.item() * 100:.2f}%\n"

    return result_text


# 4. Build the Gradio Interface
with gr.Blocks(title="Handwritten Digit Classifier") as demo:
    gr.Markdown("# Handwritten Digit Classifier")
    gr.Markdown(
        "Draw a digit (0-9) on the canvas below and click **Classify** to see the model's prediction."
    )

    with gr.Row():
        with gr.Column():
            canvas = gr.ImageEditor(
                type="pil",
                image_mode="L",  # Grayscale mode
                sources=(),  # No upload dialog
                label="Draw your digit here",
            )
            submit_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            output_text = gr.Markdown(label="Results")

    # Link the button to the prediction function
    submit_btn.click(fn=predict_digit, inputs=canvas, outputs=output_text)

if __name__ == "__main__":
    demo.launch(footer_links=["settings"])
