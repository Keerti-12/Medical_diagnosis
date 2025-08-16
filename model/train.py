"""
train.py
Beginner-friendly script to train or fine-tune DenseNet-121 for chest X-ray classification (Normal vs Pneumonia).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar

# Set device to GPU if available
# This allows PyTorch to use your local GPU for faster training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Paths to data (updated for your dataset structure)
TRAIN_DIR = 'd:/Medical_POC/chest_xray/train'
VAL_DIR = 'd:/Medical_POC/chest_xray/val'
MODEL_PATH = 'densenet121_chestxray.pth'


# Data augmentation for training: helps model generalize better
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images
    transforms.RandomRotation(10),      # Randomly rotate images
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Randomly change brightness/contrast
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation transform: only resize and normalize
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load datasets with augmentation for training
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)


# Data loaders: load data in batches for training
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Load DenseNet-121 pre-trained on ImageNet
model = models.densenet121(pretrained=True)
# Modify the classifier for binary classification (Normal vs Pneumonia)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model = model.to(DEVICE)

# Class names based on folder names
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
EPOCHS = 50  # Increased for better accuracy; adjust as needed

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    # Progress bar for training
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for images, labels in train_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        # Update progress bar with batch loss
        train_bar.set_postfix({
            'GPU_mem': f"{torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else 'CPU',
            'batch_loss': f"{loss.item():.4f}"
        })
    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc = correct_train / total_train
    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            val_bar.set_postfix({
                'batch_loss': f"{loss.item():.4f}"
            })
    val_acc = correct / total
    val_loss = val_loss / len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | GPU_mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else f"Epoch {epoch+1}/{EPOCHS} | CPU", end=' ')
    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


# Save the trained model for later use in Streamlit demo
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Note: For demo and prediction, use only Streamlit (see app/streamlit_app.py)
