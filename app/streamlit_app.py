"""
streamlit_app.py
Beginner-friendly Streamlit demo for chest X-ray classification and Grad-CAM visualization.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from utils.attention import GradCAM, overlay_cam_on_image
from model.inference import predict, MODEL_PATH, DEVICE

st.title('Chest X-ray Classification (NORMAL vs PNEUMONIA)')
st.write('Upload a chest X-ray image to get a diagnosis and see the attention map.')

uploaded_file = st.file_uploader('Choose an X-ray image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded X-ray', use_column_width=True)
    # Use same transforms as train.py
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    # Load model as in train.py
    model = models.densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    gradcam = GradCAM(model, model.features)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        prob, pred = torch.max(probs, 1)
    cam = gradcam.generate(img_tensor, class_idx=pred.item())
    gradcam.remove_hooks()
    # Overlay CAM
    img.save('temp_img.jpg')
    overlay = overlay_cam_on_image('temp_img.jpg', cam)
    st.image(overlay, caption='Grad-CAM Attention Map', use_column_width=True)
    class_names = ['NORMAL', 'PNEUMONIA']
    st.write(f'Prediction: {class_names[pred.item()]}')
    st.write(f'Probability: {float(prob.item()):.4f}')
    st.write(f'Probabilities: NORMAL={float(probs[0][0]):.4f}, PNEUMONIA={float(probs[0][1]):.4f}')
