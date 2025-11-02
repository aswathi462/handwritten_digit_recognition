import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload a 28x28 grayscale image of a digit (0–9) to predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    img_array = np.array(image)
    st.image(image, caption='Uploaded Digit', use_container_width=True)

    # Normalize and predict
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    model = keras.models.load_model("digit_model.h5")
    pred = np.argmax(model.predict(img_array))
    st.success(f"Predicted Digit: **{pred}**")
