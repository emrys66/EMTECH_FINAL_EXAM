import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.h5")

model = load_model()

# CIFAR-10 class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Streamlit UI
st.title("Image Classifier using CIFAR 10 Dataset")
st.markdown("Upload a 32x32 color image to check the classification in within the CIFAR 10 Dataset")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### Prediction: **{predicted_class}**")
