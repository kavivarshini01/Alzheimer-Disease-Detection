import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib
import torch.nn as nn
import cv2
import sqlite3
from src.gradcam import generate_heatmap

# Load trained models
cnn_model_path = "models/cnn_model.pth"
svm_model_path = "models/svm_model.pkl"
scaler_path = "models/scaler.pkl"
label_encoder_path = "models/label_encoder.pkl"

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load models
cnn_model = CNNModel()
cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device("cpu")))
cnn_model.eval()

svm_model = joblib.load(svm_model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# SQLite database setup
DB_PATH = "patient_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 image_path TEXT, prediction TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

def save_to_db(image_path, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO history (image_path, prediction, confidence) VALUES (?, ?, ?)", 
              (image_path, prediction, confidence))
    conn.commit()
    conn.close()

# Image Preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Prediction Function
def predict(image_path):
    image_tensor = preprocess_image(image_path)

    # Extract CNN feature vector
    with torch.no_grad():
        cnn_features = cnn_model(image_tensor).numpy()

    # Normalize features before SVM prediction
    cnn_features = scaler.transform(cnn_features)

    # Predict with SVM
    prediction = svm_model.predict(cnn_features)[0]
    confidence = np.max(svm_model.predict_proba(cnn_features))

    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    # Save to database
    save_to_db(image_path, predicted_label, confidence)
    
    return predicted_label, confidence

# Grad-CAM Heatmap Function
def generate_heatmap(image_path):
    image_tensor = preprocess_image(image_path)
    image_tensor.requires_grad_()

    # Forward pass
    output = cnn_model(image_tensor)
    class_idx = output.argmax().item()

    # Backpropagation
    cnn_model.zero_grad()
    output[0, class_idx].backward()

    # Extract gradients from the last convolutional layer
    gradients = cnn_model.conv2.weight.grad.mean(dim=[0, 2, 3]).detach().numpy()

    # Convert image tensor to numpy for visualization
    image_np = image_tensor.detach().squeeze().permute(1, 2, 0).numpy()

    # Normalize and apply heatmap
    heatmap = np.maximum(gradients, 0)
    heatmap = cv2.resize(heatmap, (128, 128))

    if heatmap.max() > heatmap.min():  
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Overlay heatmap on the image
    overlay = cv2.addWeighted(cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

    # Save and return heatmap
    heatmap_path = "heatmap.jpg"
    cv2.imwrite(heatmap_path, overlay)
    return heatmap_path

# Retrain Model Function
def retrain_model():
    os.system("python src/train.py")
