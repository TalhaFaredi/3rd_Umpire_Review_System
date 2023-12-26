import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model_path = 'path.h5'  # Replace with the actual path to your saved model
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = {0: 'not out', 1: 'out'}

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model_path = 'path.h5'  # Replace with the actual path to your saved model
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = {0: 'not out', 1: 'out'}

# Streamlit app
st.title("Cricket Web App")

# Navigation bar
st.markdown('''''')
# Hero Section
st.header("Your Inspiring Headline")
st.write("A captivating description of your business or passion.")
st.button("Explore More")

# Image Grid
st.subheader("Image Grid")
col1, col2 = st.columns(2)
with col1:
    st.image("not4.jpg", caption="Image 1", use_column_width=True)
    st.write("Title 1")
    st.write("Short description.")

with col2:
    st.image("out5.jpg", caption="Image 2", use_column_width=True)
    st.write("Title 2")
    st.write("Short description.")

# CEO Section
st.subheader("CEO's Message")
ceo_col1, ceo_col2 = st.columns(2)
with ceo_col1:
    st.image('f.JPG', caption="CEO Image", use_column_width=True)

with ceo_col2:
    st.write("Inspiring quote from your CEO.")
    st.button("Read More")

# Testimonials Section
st.subheader("What People Say")
st.image("afridi.jpg", use_column_width=True)
st.write("\"Fantastic service! Highly recommend them!\"")
st.write("- John Doe, CEO of Company X")

# Footer

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class = int(round(prediction[0][0]))
        predicted_label = class_labels[predicted_class]

        st.empty()
        st.subheader("Prediction Result:")
        st.write(f"The model predicts: {predicted_label} (Class {predicted_class})")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

st.write("&copy; Talha Fareedi | [Privacy Policy](#) | [Terms & Conditions](#)")
