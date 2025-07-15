# predict.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("model/brain_tumor_model.h5")

def predict_tumor(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    image = image.reshape(1, 100, 100, 1) / 255.0
    prediction = model.predict(image)[0][0]
    return "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"
