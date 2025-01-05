import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Define class names corresponding to indices
class_names = [
    'Moon JellyFish',  # Replace with actual class name for index 0
    'Barrel JellyFish',  # Replace with actual class name for index 1
    'Blue JellyFish',  # Replace with actual class name for index 2
    'Compass JellyFish',  # Replace with actual class name for index 3
    'Lions mane JellyFish',  # Replace with actual class name for index 4
    'Mauve Stinger JellyFish'   # Replace with actual class name for index 5
]

def load_and_preprocess_image(image):
    
    img = load_img(image, target_size=(224, 224))  # Resize image to match model input shape
    img_array = img_to_array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(image):
   
    processed_image = load_and_preprocess_image(image) 
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    
    # Get the corresponding class name
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name

# Streamlit UI
st.title("Jellyfish Type Classification")
st.write("Upload an image of a jellyfish to classify its type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    if st.button('Classify'):
        predicted_class_name = predict_image(uploaded_file)
        st.write(f'Predicted Class: {predicted_class_name}')
