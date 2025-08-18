# backend/main.py
# FastAPI backend for Symptom → Disease prediction
# NOTE: Replace the FEATURE_NAMES and model loading path to match your trained model.

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title="Symptom Disease Predictor", version="1.0")

# ======== LOAD YOUR TRAINED MODEL ========
# Put your model file (e.g., model.joblib) in backend/ and update the name below.
# You must export it from your training notebook using joblib.dump(...)
MODEL_PATH = "backend/model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"WARNING: Could not load model at {MODEL_PATH}. Error: {e}")

# ======== EDIT THIS TO MATCH YOUR FEATURES ========
# Example: symptoms converted to 0/1 feature vector
FEATURE_NAMES = [
    # Replace these with the exact features your model expects (order matters!)
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue",
    # ... add all features you used during training
]

class PredictRequest(BaseModel):
    symptoms: List[str]  # e.g., ["fever", "headache"]

class PredictResponse(BaseModel):
    predicted_disease: str
    probas: Dict[str, float] | None = None

@app.get("/")
def root():
    return {"status": "ok", "message": "Symptom Disease Predictor API"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        return PredictResponse(predicted_disease="MODEL_NOT_LOADED", probas=None)

    # Build a 0/1 vector over FEATURE_NAMES
    bin_vec = np.zeros(len(FEATURE_NAMES), dtype=int)
    user_syms = set([s.strip().lower() for s in req.symptoms])

    for i, feat in enumerate(FEATURE_NAMES):
        if feat.lower() in user_syms:
            bin_vec[i] = 1

    X = np.array([bin_vec])

    # Predict
    try:
        y_pred = model.predict(X)[0]
        # If your model supports predict_proba:
        probas = None
        if hasattr(model, "predict_proba"):
            proba_vec = model.predict_proba(X)[0]
            # If you have class names, map them; fallback to index
            if hasattr(model, "classes_"):
                probas = {str(cls): float(p) for cls, p in zip(model.classes_, proba_vec)}
            else:
                probas = {str(i): float(p) for i, p in enumerate(proba_vec)}
        return PredictResponse(predicted_disease=str(y_pred), probas=probas)
    except Exception as e:
        return PredictResponse(predicted_disease=f"ERROR: {e}", probas=None)
