%%writefile app.py
import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("tumor_classifier.h5")

# Labels and Recommendations
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No_Tumor']
recommendations = {
    "Glioma": "🧠 Consult a neuro-oncologist. MRI and biopsy recommended. Surgical removal may be needed.",
    "Meningioma": "🧠 Usually benign. Consult neurologist. Monitor or remove surgically.",
    "Pituitary": "🩺 May affect hormones. Consult endocrinologist. MRI & hormone test required.",
    "No_Tumor": "✅ No tumor detected. If symptoms persist, consult a neurologist for further evaluation."
}

# Prediction Function
def predict(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    return class_names[idx], preds[0][idx], recommendations[class_names[idx]]

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.markdown(
    """
    <style>
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #4B0082;
        }
        .footer {
            margin-top: 30px;
            font-size: 13px;
            color: #888;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="title">🧠 Brain Tumor Detection App</div>', unsafe_allow_html=True)
st.markdown("Upload an MRI scan to predict tumor type and receive medical recommendations.")

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_path = os.path.join("uploaded", uploaded_file.name)
    os.makedirs("uploaded", exist_ok=True)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(img_path, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing..."):
        tumor_type, confidence, advice = predict(img_path)

    st.success(f"✅ Prediction: **{tumor_type}** ({confidence*100:.2f}%)")
    st.info(f"📋 Recommendation: {advice}")

st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)

