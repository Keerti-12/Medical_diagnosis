"""
inference.py
Beginner-friendly script for loading a trained DenseNet-121 model and making predictions on chest X-ray images.
"""
import torch
from torchvision import models, transforms
from PIL import Image
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'densenet121_chestxray.pth'

# Define the same transforms as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Class names
CLASS_NAMES = ['Normal', 'Pneumonia']

def predict(image_path):
    """
    Predicts the class and probability for a given X-ray image.
    """
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        prob, pred = torch.max(probs, 1)
    return CLASS_NAMES[pred.item()], prob.item(), probs.cpu().numpy()[0]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    else:
        test_img = 'test_image.png'  # Default test image
    label, prob, all_probs = predict(test_img)
    print(f"Prediction: {label}, Probability: {prob:.4f}")
