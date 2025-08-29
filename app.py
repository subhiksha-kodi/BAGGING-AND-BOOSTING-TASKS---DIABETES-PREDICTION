import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details below to predict the likelihood of diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1, format="%d")
glucose = st.number_input("Glucose", min_value=0, step=1, format="%d")
blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1, format="%d")
skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1, format="%d")
insulin = st.number_input("Insulin", min_value=0, step=1, format="%d")
bmi = st.number_input("BMI", min_value=0.0, step=0.1, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01, format="%.2f")
age = st.number_input("Age", min_value=0, step=1, format="%d")

# Prediction button
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Patient is likely Diabetic (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Patient is Non-Diabetic (Probability: {probability:.2f})")