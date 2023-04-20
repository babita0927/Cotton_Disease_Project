import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Load the trained model
MODEL_PATH = 'model_resnet152V2.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define a function for model prediction
def model_predict(img_path, model):
    img = Image.open(img_path).resize((224, 224))
    x = np.array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The leaf is diseased cotton leaf"
    else:
        preds = "The leaf is fresh cotton leaf"
    return preds

# Define the Streamlit app
def app():
    st.set_page_config(page_title='Cotton Plant Disease Classification')
    st.title('Cotton Plant Disease Classification')

    # Add file uploader
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    # Make a prediction on the uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('')
        with st.spinner('Classifying...'):
            prediction = model_predict(uploaded_file, model)
        st.success(f'Prediction: {prediction}')


if __name__ == '__main__':
    app()
