# Alzheimer Disease Detection using CNN and SVM

This project classifies MRI brain images into four stages of Alzheimer’s Disease:
- Non-Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

It uses Convolutional Neural Networks (CNN) for feature extraction and a Support Vector Machine (SVM) for classification.

## Folder Structure

- `src/` – Source code for training and prediction
- `models/` – Trained model files
- `app.py` – Main app for prediction
- `uploads/` – Temporary uploaded images
- `*.db` – SQLite database for storing results

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt