import streamlit as st
import pickle
import numpy as np

# Load the trained model
filename = 'iris_model.pkl'
try:
    with open(filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: The file '{filename}' was not found. Make sure it's in the same directory as this app.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Title of the web application
st.title('Iris Flower Prediction App')
st.subheader('Enter the flower features to get a prediction')

# Input fields for the features
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.5)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.3)

# Prediction button
if st.button('Predict Flower'):
    # Prepare the input data as a NumPy array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make the prediction
    prediction = model.predict(features)

    # Map the prediction to the class name
    class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    predicted_class = class_names[prediction[0]]

    # Display the prediction
    st.subheader(f'The predicted Iris flower is: **{predicted_class}**')