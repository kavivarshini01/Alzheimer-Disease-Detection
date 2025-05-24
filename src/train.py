import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 output classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset Preparation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset_path = "D:\\PROJECT\\Alzheimer\\dataset"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Apply Label Encoding for class labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform([dataset.classes[i] for i in dataset.targets])

# Split dataset
train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=encoded_labels, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Dataset uploaded properly")

# Train CNN model
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training CNN model...")

def train_cnn():
    model.train()
    for epoch in range(10):  
        for images, labels in train_loader:
            images = images.to(torch.float32)  # Ensure correct dtype
            labels = labels.to(torch.long)  

            optimizer.zero_grad()
            outputs = model(images)  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    torch.save(model.state_dict(), "models/cnn_model.pth")

# Extract CNN Features & Train SVM
def extract_features():
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in train_loader:
            images = images.to(torch.float32)
            outputs = model(images)
            features.extend(outputs.numpy())
            labels.extend(lbls.numpy())
    return np.array(features), np.array(labels)

def train_svm():
    features, labels = extract_features()
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    svm = SVC(kernel='linear', probability=True)
    svm.fit(features, labels)

    # Save models & encoders
    joblib.dump(svm, "models/svm_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

if __name__ == "__main__":
    train_cnn()
    print("CNN Training Completed")

    print("Training SVM...")
    train_svm()
    print("Training Complete!")
