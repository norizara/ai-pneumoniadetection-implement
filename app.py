import torch
import torch.nn as nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
from PIL import Image
import gradio as gr

# Download the pretrained weights from Hugging Face
repo_id = "izeeek/resnet18_pneumonia_classifier"
filename = "resnet18_pneumonia_classifier.pth"
path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load model architecture
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Prediction function
def predict(image: Image.Image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img)
        predicted = torch.argmax(logits, dim=1).item()
    return "Pneumonia" if predicted == 1 else "Normal"

# Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Pneumonia Detection (ResNet-18)",
    description="Upload a chest X-ray image to detect Pneumonia using a PyTorch ResNet18 model trained on the Paul Mooney dataset."
).launch()
