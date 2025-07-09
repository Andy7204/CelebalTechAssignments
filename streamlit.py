# app.py
import streamlit as st
import numpy as np
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load trained model and data
model = joblib.load("model.pkl")
n_features = joblib.load("n_features.pkl")
X_train = joblib.load("train_data_X.pkl")

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

    if prediction == 1:
        st.success("🧪 Prediction: Cancerous Cell")
    else:
        st.success("🧪 Prediction: Noncancerous Cell")

    st.markdown("### 📊 Confidence")
    st.progress(int(proba[prediction] * 100))

    st.markdown("### 🔍 Why did the model say that? (SHAP Explanation)")

    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # SHAP expects a DataFrame
    X_df = pd.DataFrame(X_train, columns=[f"Feature {i+1}" for i in range(n_features)])
    input_df = pd.DataFrame(input_array, columns=X_df.columns)

    # Force plot or waterfall plot (we use summary + force here)
    shap.initjs()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.write("#### 🔥 Feature impact on this specific prediction")
    shap_values_input = explainer.shap_values(input_df)
    shap.force_plot(explainer.expected_value[1], shap_values_input[1][0], input_df, matplotlib=True, show=False)
    st.pyplot(bbox_inches='tight', dpi=300)

# Clear button
if st.button("Clear"):
    st.experimental_rerun()
