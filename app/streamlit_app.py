"""
streamlit_app.py
Beginner-friendly Streamlit demo for chest X-ray classification and Grad-CAM visualization.
"""
import sys
import os
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Add parent directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.attention import GradCAM, overlay_cam_on_image
from model.inference import MODEL_PATH, DEVICE

# --- UI Layout ---
st.set_page_config(page_title="Chest X-ray Classifier", layout="centered")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #e3f0ff 0%, #f8f9fa 100%) !important;}
h1, h2, h3, h4, h5, h6, p, label, .stFileUploader, .stButton>button, .stInfo {color: #222 !important; font-size: 18px !important;}
.stFileUploader {background-color: #f5faff !important; border: 1px solid #b0c4de !important;}
.stButton>button {background-color: #0072B5 !important; color: #fff !important; font-weight: bold; border-radius: 6px !important;}
.stFileUploader label {color: #222 !important;}
.stFileUploader .css-1y4p8pa {background-color: #f5faff !important; color: #222 !important;}
.stFileUploader .css-1y4p8pa button {background-color: #0072B5 !important; color: #fff !important; font-weight: bold;}
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stFileUploader,
[data-testid="stSidebar"] .stButton>button {
    color: #fff !important;  /* White color */
}
/* Make 'Browse files' text and upload icon white in file uploader */
[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] svg,
.stFileUploader label span,
.stFileUploader label,
.stFileUploader span,
.stFileUploader button {
    color: #fff !important;
    fill: #fff !important;
}
button[aria-label="Browse files"] {
    color: #fff !important;
    background: #0072B5 !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/8/8a/Chest_Xray_PA_3-8-2010.png")
st.sidebar.markdown("<h2 style='color:#fff; font-size:1.5rem;'>Chest X-ray Classifier</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#fff; font-size:1.1rem;'>Classifies X-rays as <b>NORMAL</b> or <b>PNEUMONIA</b> and shows attention map.</p>", unsafe_allow_html=True)

# Moderately enlarged Title with diagonal gradient background
st.markdown("""
<div style='
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 32px;
'>
    <div style='
        background: linear-gradient(135deg, #6dd5fa 0%, #e3f0ff 60%, #f8f9fa 100%);
        border-radius: 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.07);
        padding: 28px 40px;
        display: inline-block;
    '>
        <span style='
            color: #222;
            font-size: 2.5rem;
            font-weight: 800;
            text-align: center;
            display: block;
        '>
            Chest X-ray Classification
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='display:flex; justify-content:center; align-items:center;'>
    <p style='color:#222; font-size:1.25rem; margin-bottom:32px; text-align:center;'>
        Upload a chest X-ray image below. The model will predict the diagnosis and show which regions influenced its decision.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload X-ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded X-ray', use_container_width=True)
    with st.spinner('Analyzing image...'):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        # Robust model loading
        model = models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except FileNotFoundError:
            st.error(f"Model file not found: {MODEL_PATH}. Please train and save the model first.")
            st.stop()
        model = model.to(DEVICE)
        model.eval()
        gradcam = GradCAM(model, model.features)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            prob, pred = torch.max(probs, 1)
        cam = gradcam.generate(img_tensor, class_idx=pred.item())
        gradcam.remove_hooks()
        img.save('temp_img.jpg')
        overlay = overlay_cam_on_image('temp_img.jpg', cam)
    st.markdown(f"<h2 style='color:#0072B5; font-size:2rem; text-align:center;'>Prediction: {'NORMAL' if pred.item() == 0 else 'PNEUMONIA'}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1.2rem; text-align:center;'><b>Confidence:</b> {float(prob.item()):.2%}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1.2rem; text-align:center;'><b>Probabilities:</b> NORMAL = {float(probs[0][0]):.2%}, PNEUMONIA = {float(probs[0][1]):.2%}</p>", unsafe_allow_html=True)
    st.image(overlay, caption='Grad-CAM Attention Map', use_container_width=True)
else:
    st.info("Please upload a chest X-ray image to begin.")