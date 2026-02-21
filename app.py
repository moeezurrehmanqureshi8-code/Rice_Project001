
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Rice Demand Prediction", layout="centered")

st.title("🌾 Rice Demand Prediction App")
st.write("Predict rice demand using trained ML model")

# Load model
@st.cache_resource
def load_model():
    with open("rice_demand_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Sidebar Inputs
st.sidebar.header("Input Features")

# NOTE:
# Replace these inputs according to your actual model features.
# If you tell me your feature names, I’ll customize this exactly.

price = st.sidebar.number_input("Rice Price", min_value=0.0, value=50.0)
income = st.sidebar.number_input("Average Income", min_value=0.0, value=30000.0)
population = st.sidebar.number_input("Population", min_value=0.0, value=100000.0)

# Prediction Button
if st.sidebar.button("Predict Demand"):
    
    input_data = np.array([[price, income, population]])
    
    prediction = model.predict(input_data)

    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Rice Demand: {prediction[0]:,.2f}")

st.markdown("---")
st.write("Built with Streamlit 🚀")