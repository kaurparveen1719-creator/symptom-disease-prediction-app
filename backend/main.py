# backend/main.py
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Symptom-Disease Predictor", version="1.0")

# Allow calls from Streamlit / browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict to your Streamlit URL if you like
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Paths (files must be in backend/) ----------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "symptom_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
SEVERITY_CSV = BASE_DIR / "Symptom-severity.csv"   # contains the feature names in column 'Symptom'

# ---------- Load artifacts at startup ----------
model = None
label_encoder = None
FEATURE_NAMES: List[str] = []  # exact order used for encoding

class PredictIn(BaseModel):
    symptoms: List[str]  # e.g. ["fever", "headache"]

def _load_feature_names() -> List[str]:
    """Read symptom feature names (order!) from Symptom-severity.csv -> 'Symptom' column."""
    if not SEVERITY_CSV.exists():
        raise FileNotFoundError(f"Feature list CSV not found at {SEVERITY_CSV}")
    df = pd.read_csv(SEVERITY_CSV)
    if "Symptom" not in df.columns:
        raise ValueError("Expected column 'Symptom' in Symptom-severity.csv")
    # normalize to lowercase & strip, preserve CSV order (assumed to match training)
    names = [str(s).strip().lower() for s in df["Symptom"].tolist()]
    # remove obvious empties / NAs
    names = [n for n in names if n and n != "nan"]
    return names

@app.on_event("startup")
def _startup():
    global model, label_encoder, FEATURE_NAMES
    # Load feature names
    try:
        FEATURE_NAMES = _load_feature_names()
        print(f"[startup] Loaded {len(FEATURE_NAMES)} feature names from {SEVERITY_CSV.name}")
    except Exception as e:
        print(f"[startup][WARN] Failed to load FEATURE_NAMES: {e}")
        FEATURE_NAMES = []

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[startup] Model loaded: {MODEL_PATH.name}")
    except Exception as e:
        model = None
        print(f"[startup][ERROR] Could not load model at {MODEL_PATH}: {e}")

    # Load label encoder
    try:
        label_encoder = joblib.load(ENCODER_PATH)
        print(f"[startup] Label encoder loaded: {ENCODER_PATH.name}")
    except Exception as e:
        label_encoder = None
        print(f"[startup][ERROR] Could not load label encoder at {ENCODER_PATH}: {e}")

@app.get("/meta")
def meta() -> Dict[str, Optional[object]]:
    classes = None
    if getattr(label_encoder, "classes_", None) is not None:
        try:
            classes = label_encoder.classes_.tolist()
        except Exception:
            classes = None
    return {
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "n_features": len(FEATURE_NAMES),
        "features_preview": FEATURE_NAMES[:10],   # first 10 so response stays small
        "model_path": str(MODEL_PATH.name),
        "encoder_path": str(ENCODER_PATH.name),
        "feature_source": str(SEVERITY_CSV.name),
        "classes": classes,
    }

@app.post("/predict")
def predict(payload: PredictIn):
    if model is None or label_encoder is None:
        return {"predicted_disease": "MODEL_NOT_LOADED", "probas": None}

    if not FEATURE_NAMES:
        return {"predicted_disease": "FEATURES_NOT_LOADED", "probas": None}

    # multi-hot encode symptoms in the exact FEATURE_NAMES order
    present = {s.strip().lower() for s in payload.symptoms}
    x = np.zeros((1, len(FEATURE_NAMES)), dtype=float)
    for i, feat in enumerate(FEATURE_NAMES):
        if feat in present:
            x[0, i] = 1.0

    # Predict
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x)[0]  # shape (n_classes,)
            top_idx = int(np.argmax(prob))
            # map class index -> disease label via label_encoder
            pred_label = label_encoder.inverse_transform([top_idx])[0]
            probas = {
                label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(prob)
            }
            return {"predicted_disease": str(pred_label), "probas": probas}
        else:
            idx = int(model.predict(x)[0])
            pred_label = label_encoder.inverse_transform([idx])[0]
            return {"predicted_disease": str(pred_label), "probas": None}
    except Exception as e:
        return {"predicted_disease": f"ERROR: {e}", "probas": None}
