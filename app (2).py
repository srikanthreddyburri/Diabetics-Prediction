import streamlit as st
import pickle
import numpy as np


# opening the pickle files
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('selected_features.pkl', 'rb') as file:
    selected_features = pickle.load(file)

# UserInterface using Streamlit
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

st.title("🔍 Diabetes Prediction System")
st.write("Enter the following details to check if a person is likely to have diabetes.")

# Inputs
glucose = st.number_input("Glucose Level", min_value=0.0, step=0.1, format="%.1f")
bmi = st.number_input("BMI", min_value=0.0, step=0.1, format="%.1f")
age = st.number_input("Age", min_value=1, step=1, format="%d")

# Prediction function
def predict_diabetes(glucose, bmi, age):
    input_data = np.array([glucose, bmi, age]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return "🛑 Diabetes Detected" if prediction[0] == 1 else "✅ No Diabetes"

# Prediction Button
if st.button("Predict Diabetes"):
    if glucose > 0 and bmi > 0 and age > 0:
        result = predict_diabetes(glucose, bmi, age)
        st.success(result)
    else:
        st.error("Please enter valid positive values.")
