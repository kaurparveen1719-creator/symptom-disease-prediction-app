from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Symptom-Disease Predictor", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "symptom_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
FEATURES_CSV = BASE_DIR / "feature_names.csv"   # <- we load THIS file

model = None
label_encoder = None
FEATURE_NAMES: List[str] = []
_last_errors: Dict[str, str] = {}

class PredictIn(BaseModel):
    symptoms: List[str]

def load_feature_names() -> List[str]:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"{FEATURES_CSV} not found.")
    df = pd.read_csv(FEATURES_CSV)
    col = df.columns[0]  # supports 1-col CSV or header "symptom"
    names = [str(v).strip() for v in df[col].tolist() if str(v).strip()]
    if not names:
        raise ValueError("feature_names.csv loaded but empty.")
    return names

@app.on_event("startup")
def startup():
    global model, label_encoder, FEATURE_NAMES, _last_errors
    _last_errors = {}
    try:
        FEATURE_NAMES = load_feature_names()
    except Exception as e:
        FEATURE_NAMES = []
        _last_errors["features"] = repr(e)
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model = None
        _last_errors["model"] = repr(e)
    try:
        label_encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        label_encoder = None
        _last_errors["encoder"] = repr(e)

@app.get("/meta")
def meta():
    classes = getattr(label_encoder, "classes_", None)
    return {
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "n_features": len(FEATURE_NAMES),
        "features_preview": FEATURE_NAMES[:10],
        "classes_preview": (classes[:10].tolist() if classes is not None else None),
        "last_errors": _last_errors or None,
    }

@app.post("/predict")
def predict(payload: PredictIn):
    if model is None or label_encoder is None:
        return {"predicted_disease": "MODEL_NOT_LOADED", "probas": None}
    if not FEATURE_NAMES:
        return {"predicted_disease": "FEATURES_NOT_LOADED", "probas": None}

    present = {s.strip().lower() for s in payload.symptoms}
    x = np.zeros((1, len(FEATURE_NAMES)), dtype=float)
    for i, feat in enumerate(FEATURE_NAMES):
        if feat.lower() in present:
            x[0, i] = 1.0

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)[0]
        idx = int(np.argmax(prob))
        label = label_encoder.inverse_transform([idx])[0]
        probas = {label_encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(prob)}
        return {"predicted_disease": str(label), "probas": probas}
    else:
        idx = int(model.predict(x)[0])
        label = label_encoder.inverse_transform([idx])[0]
        return {"predicted_disease": str(label), "probas": None}

@app.get("/features")
def features():
    return {"features": FEATURE_NAMES}
