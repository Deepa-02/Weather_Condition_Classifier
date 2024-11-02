import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json

# Load the trained model and label map
model = load_model('weather_classifier.keras')
with open('label_map.json', 'r') as file:
    label_map = json.load(file)
inv_label_map = {v: k for k, v in label_map.items()}  # Reverse label map

# Define a function to predict the class of an image
def predict_image(image):
    img = image.resize((150, 150))  # Resize the image to match model input
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    result_message = f"The picture you provided represents: **{inv_label_map[predicted_class]}**"
    return result_message

# Streamlit app layout
st.title("Weather Condition Classifier")
st.write("Upload an image of weather conditions to classify.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load the image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    prediction_message = predict_image(image)
    
    # Display the result with bold formatting
    st.write(prediction_message)
