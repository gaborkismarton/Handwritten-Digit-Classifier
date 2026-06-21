import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from models.simplenet import SimpleNet
from models.lenet5 import LeNet5
from models.resnet import ResNet

# Load the trained models into memory
simplenet_model = SimpleNet()
simplenet_model.load_state_dict(torch.load("simplenet_mnist.pth"))
simplenet_model.eval()

lenet_model = LeNet5()
lenet_model.load_state_dict(torch.load("lenet5_mnist.pth"))
lenet_model.eval()

resnet_model = ResNet()
resnet_model.load_state_dict(torch.load("resnet_mnist.pth"))
resnet_model.eval()

# Data normalization
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def predict_all_models(image_dict):
    # Ensure the user actually drew something
    if not image_dict or not image_dict.get("composite"):
        error_msg = "Please draw a number in the canvas first."
        return error_msg, error_msg, error_msg

    img = image_dict["composite"]

    # Ensure it's in grayscale
    img = img.convert("L")

    # Invert the image: MNIST expects a black background (0) and white digits (255)
    import PIL.ImageOps
    img = PIL.ImageOps.invert(img)

    # Resize to the 28x28 dimension expected by the network
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img.save("debug_input.png")

    # Convert image to model input tensor
    input_tensor = transform(img).unsqueeze(0)

    models_to_run = {
        "SimpleNet": simplenet_model,
        "LeNet5": lenet_model,
        "ResNet": resnet_model
    }

    formatted_outputs = []

    # Run inference for all models
    with torch.no_grad():
        for model_name, active_model in models_to_run.items():
            logits = active_model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, prediction = torch.max(probs, 1)

            pred_val = prediction.item()
            conf_val = conf.item() * 100

            # Format the output message for this specific model
            result_text = f"### {model_name}: **{pred_val}**\n"
            result_text += f"**Confidence:** {conf_val:.2f}%\n\n"
            result_text += "#### Detailed Probabilities:\n"
            for i, p in enumerate(probs[0]):
                result_text += f"* **Digit {i}:** {p.item() * 100:.2f}%\n"
            
            formatted_outputs.append(result_text)

    # Return the three distinct strings to map to the three output columns
    return tuple(formatted_outputs)


# Build the Gradio Interface
with gr.Blocks(title="Handwritten Digit Classifier") as demo:
    gr.Markdown("# Handwritten Digit Classifier")
    gr.Markdown(
        "Draw a digit (0-9) on the canvas below and click **Classify** to see predictions from all three models simultaneously."
    )

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.ImageEditor(
                type="pil",
                image_mode="L",  # Grayscale mode
                sources=(),  # No upload dialog
                label="Draw your digit here",
            )
            submit_btn = gr.Button("Classify", variant="primary")

        # Create three separate columns on the right for side-by-side comparison
        with gr.Column(scale=2):
            with gr.Row():
                out_simplenet = gr.Markdown(label="SimpleNet")
                out_lenet = gr.Markdown(label="LeNet5")
                out_resnet = gr.Markdown(label="ResNet")

    # Link the button to the prediction function, outputting to all three markdown blocks
    submit_btn.click(
        fn=predict_all_models, 
        inputs=[canvas], 
        outputs=[out_simplenet, out_lenet, out_resnet]
    )

if __name__ == "__main__":
    demo.launch(footer_links=["settings"])