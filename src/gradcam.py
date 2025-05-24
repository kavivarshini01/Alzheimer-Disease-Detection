import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
from src.model import CNNModel  # Ensure the correct import

# Load Trained CNN Model
cnn_model_path = "models/cnn_model.pth"
cnn_model = CNNModel()
cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device("cpu")))
cnn_model.eval()

# Grad-CAM Hooks
gradients = None
activations = None

def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]  # Store gradients

def save_activation(module, input, output):
    global activations
    activations = output  # Store activations

# Register hooks on the last convolutional layer
cnn_model.conv2.register_forward_hook(save_activation)
cnn_model.conv2.register_backward_hook(save_gradient)

# Preprocess MRI Image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Generate Grad-CAM Heatmap
def generate_heatmap(image_path):
    global gradients, activations
    image_tensor = preprocess_image(image_path)
    image_tensor.requires_grad_()

    # Forward Pass
    output = cnn_model(image_tensor)
    class_idx = output.argmax().item()

    # Backward Pass
    cnn_model.zero_grad()
    output[0, class_idx].backward()

    # Compute Grad-CAM heatmap
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # Average across channels
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]  # Apply weights to activations

    heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0)  # Apply ReLU

    # **Fixing Stripes Issue**
    heatmap = cv2.resize(heatmap, (128, 128), interpolation=cv2.INTER_LINEAR)  
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # **Proper Overlay on MRI Image**
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (128, 128), interpolation=cv2.INTER_LINEAR)

    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)  # Adjust transparency

    # Save Heatmap
    heatmap_path = "heatmap_corrected.jpg"
    cv2.imwrite(heatmap_path, overlay)

    return heatmap_path
