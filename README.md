# Chest X-ray Classification with DenseNet-121

Beginner-friendly project for classifying chest X-ray images (Normal vs Pneumonia) using PyTorch, DenseNet-121, Grad-CAM, Flask, and Streamlit.

## Project Structure
```
model/
  train.py         # Train or fine-tune DenseNet-121
  inference.py     # Inference script for predictions
utils/
  attention.py     # Grad-CAM attention map utility
app/
  flask_app.py     # Flask API for prediction and Grad-CAM
  streamlit_app.py # Streamlit demo for uploading and visualizing
README.md          # This file
```

## Setup Instructions

### 1. Create and Activate Python Virtual Environment
Open PowerShell and run:
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 2. Install Required Packages
```powershell
# Install PyTorch and torchvision with CUDA support for GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install flask streamlit pillow numpy opencv-python
```

### 3. Download the Dataset
- Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
- Extract it and place the folders as follows:
  - `data/train/`
  - `data/val/`
  - `data/test/`

### 4. Training the Model
Edit paths in `model/train.py` if needed, then run:
```powershell
python model/train.py
```

### 5. Inference
Test predictions on a single image:
```powershell
python model/inference.py
```

### 6. Run Flask API
```powershell
python app/flask_app.py
```
- Use a tool like Postman or curl to POST an X-ray image to `http://127.0.0.1:5000/predict`.

### 7. Run Streamlit Demo
```powershell
streamlit run app/streamlit_app.py
```
- Open the provided local URL in your browser.

## Notes
- All code is beginner-friendly and well-commented.
- Uses GPU if available (PyTorch + CUDA).
- Grad-CAM visualizes model attention on X-ray images.
- For any issues, check file paths and ensure your virtual environment is activated.

## File Overview
- `model/train.py`: Training script for DenseNet-121.
- `model/inference.py`: Loads model and predicts on new images.
- `utils/attention.py`: Grad-CAM implementation and overlay utility.
- `app/flask_app.py`: REST API for predictions and attention maps.
- `app/streamlit_app.py`: Interactive demo for uploading and visualizing results.

---
**Enjoy learning and exploring deep learning for medical imaging!**
