import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("./models/knn_model.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("./models/scaler.pkl", "rb"))

# Load encoder
encoder = pickle.load(open("./models/encoder.pkl", "rb"))

st.title("Iris Flower Classification using KNN")

st.write("Enter flower measurements:")

# Inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Prediction
if st.button("Predict"):

    features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)

    flower_name = encoder.inverse_transform(prediction)

    st.success(f"Predicted Flower: {flower_name[0]}")