# frontend/app.py
# Streamlit UI that sends selected symptoms to the FastAPI backend for prediction

import streamlit as st
import requests

st.set_page_config(page_title="Symptom → Disease Predictor", page_icon="🩺")

st.title("🩺 Symptom → Disease Predictor")
st.write("Select your symptoms and get a predicted disease from the model.")

# Make sure this matches FEATURE_NAMES in the backend
ALL_SYMPTOMS = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue",
    # ... add all features you used during training
]

backend_url = st.text_input("Backend URL", value="http://127.0.0.1:8000", help="FastAPI server URL")

selected = st.multiselect("Select your symptoms:", options=ALL_SYMPTOMS)

if st.button("Predict"):
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        try:
            resp = requests.post(f"{backend_url}/predict", json={"symptoms": selected}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Predicted disease: **{data.get('predicted_disease','N/A')}**")
                probas = data.get("probas")
                if probas:
                    st.write("Probabilities:")
                    st.json(probas)
            else:
                st.error(f"Backend error: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
