# app.py
import streamlit as st
import numpy as np
import joblib

# Load trained model and data
model = joblib.load("model.pkl")
n_features = joblib.load("n_features.pkl")

st.title("🧬 Cancer Cell Classifier")
st.write("Enter cell features to predict whether it's cancerous or noncancerous.")

# Input features
input_features = []
for i in range(n_features):
    val = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    input_features.append(val)

input_array = np.array(input_features).reshape(1, -1)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]

    # Result
    if prediction == 1:
        st.success("🧪 Prediction: Cancerous Cell")
    else:
        st.success("🧪 Prediction: Noncancerous Cell")

    # Confidence
    st.markdown("### 📊 Confidence")
    st.progress(int(proba[prediction] * 100))
    st.write(f"**Confidence Score:** {proba[prediction]*100:.2f}%")

# Clear button
if st.button("Clear"):
    st.experimental_rerun()
