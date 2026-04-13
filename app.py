import streamlit as st
import numpy as np
import pickle

st.title("🎓 Student Performance Predictor")

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Inputs
studytime = st.slider("Study Time", 1, 4)
failures = st.slider("Failures", 0, 4)
absences = st.slider("Absences", 0, 100)

if st.button("Predict"):
    input_data = np.array([[studytime, failures, absences]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ PASS")
    else:
        st.error("❌ FAIL")
