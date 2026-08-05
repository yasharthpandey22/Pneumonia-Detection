import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Load model
MODEL_PATH = "pneumonia_model.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1K1A378amAI-kNVsUh9zoiQm486XvL-RU"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

st.title("🫁 Pneumonia Detection using Chest X-ray")

uploaded_file = st.file_uploader(
    "Upload a Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((224, 224))

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.error(f"Prediction: PNEUMONIA\n\nConfidence: {confidence*100:.2f}%")
    else:
        st.success(f"Prediction: NORMAL\n\nConfidence: {(1-confidence)*100:.2f}%")