# app.py
import streamlit as st
from PIL import Image
from predict import predict_tumor

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection Web App")

st.write("Upload a brain MRI image, and the AI model will predict whether it shows a brain tumor.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("🔍 Predict"):
        result = predict_tumor(uploaded_file)
        st.success(f"🧠 Prediction Result: **{result}**")

