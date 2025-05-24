import streamlit as st
import os
import time
from src.user import predict, generate_heatmap, retrain_model, DB_PATH
import sqlite3
import pandas as pd

# Set Page Config
st.set_page_config(page_title="Alzheimer’s MRI Detector", page_icon="🧠", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Predict", "History", "Retrain Model"])

# Theme Toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

# Apply custom dark mode style
if dark_mode:
    st.markdown("""
        <style>
            body { background-color: #121212; color: white; }
            .stApp { background-color: #121212; }
        </style>
    """, unsafe_allow_html=True)

# Ensure necessary directories exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")
 
# Introduction Page
if page == "Introduction":
    st.title("🧠 Alzheimer’s Disease Detection Using MRI")
    st.markdown("""
    ### About This Project
    Alzheimer's disease is a progressive neurological disorder that affects memory and cognitive function. Early detection is crucial for managing and slowing its progression.
    
    This application uses **Convolutional Neural Networks (CNN)** and **Support Vector Machines (SVM)** to analyze MRI scans and classify them into different stages of Alzheimer's disease.

    **Key Features:**
    - 📊 MRI-based classification of Alzheimer's stages
    - 🔍 Grad-CAM heatmap visualization for explainability
    - 🛠 Real-time model retraining for improved accuracy

    Upload an MRI scan in the **Predict** section to get started.
    """)

    # Display Sample MRI Image
    sample_img_path = "D:/PROJECT/Alzheimer/dataset/NonDemented/1 (10).jpg"


    if os.path.exists(sample_img_path):
        st.image(sample_img_path, caption="Sample MRI Scan", use_container_width=300)
    else:
        st.warning("Sample MRI image not found! Please place an image at `assets/sample_mri.jpg`.")

# Prediction Page
elif page == "Predict":
    st.title("📊 MRI Scan Prediction")
    
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image_path = os.path.join("uploads", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(image_path, caption="Uploaded MRI", width=300)

        if st.button("🔍 Analyze MRI"):
            with st.spinner("Processing..."):
                prediction, confidence = predict(image_path)
                heatmap_path = generate_heatmap(image_path)

            st.success(f"Prediction: **{prediction}**")
            st.info(f"Confidence Score: **{confidence:.2f}**")

            st.image(heatmap_path, caption="Grad-CAM Heatmap", use_container_width=300)

# History Page
elif page == "History":
    st.title("📜 Prediction History")

    # Fetch history from database
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()

    if not df.empty:
        st.dataframe(df)
    else:
        st.warning("No history available.")

# Retrain Model Page
elif page == "Retrain Model":
    st.title("🔄 Retrain Model")
    st.write("Use this feature to retrain the model with new data.")

    if st.button("🛠 Retrain Now"):
        with st.spinner("Retraining Model... This may take some time."):
            retrain_model()
            time.sleep(5)  # Simulate training time
        st.success("✅ Model retrained successfully!")
