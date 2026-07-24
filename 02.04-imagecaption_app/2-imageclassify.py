import torch
from torchvision import transforms, models

# Load ResNet18 directly from torchvision to avoid torch.hub version conflicts.
try:
    # Newer torchvision versions use the `weights` enum
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
except Exception:
    # Fallback for older torchvision versions
    model = models.resnet18(pretrained=True).eval()

import requests
# Download human-readable labels for ImageNet
response = requests.get("https://git.io/JJkYN")
labels = [l.strip() for l in response.text.split("\n") if l.strip()]
# Define image preprocessing (IMPORTANT for ResNet)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
def predict(inp):
    # preprocess image
    inp = transform(inp).unsqueeze(0)
    # ensure model runs in inference mode
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    # map predictions to labels
    confidences = {
        labels[i]: float(prediction[i]) 
        for i in range(len(labels))
    }
    return confidences

import gradio as gr
gr.Interface(fn=predict,
       inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3),
       examples=["/content/lion.jpg", "/content/cheetah.jpg"]).launch()
