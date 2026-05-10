# Handwritten Digit Classifier

### Introduction to Machine Learning Project Work
### Ferenc Juhász, Gábor Kismárton

## Overall Project Plan

- Train a CNN (ex. LeNet5, UNet or compare multiple ones if we have time) with the MNIST dataset
- Develop a Gradio application where the user can draw numbers and the trained classifier will detect them: https://www.gradio.app/docs/gradio/imageeditor

## User Guide

- Train LeNet5 model by running **src/train.py**:
```bash
uv run src/train.py
```
- You can test the model on one image of the MNIST Dataset using **src/single_image_test.py**
  - You can change the INDEX variable to select different images from the MNIST Dataset
```bash
uv run src/single_image_test.py
```
