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

# Allow calls from Streamlit/browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Streamlit URL if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "symptom_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
SEVERITY_CSV = BASE_DIR / "Symptom-severity.csv"  # must contain column 'Symptom'

# ---------- Globals ----------
model = None
label_encoder = None
FEATURE_NAMES: List[str] = []
_last_errors: Dict[str, str] = {}

# ---------- Schemas ----------
class PredictIn(BaseModel):
    symptoms: List[str]  # e.g., ["itching", "skin_rash"]

# ---------- Helpers ----------
def load_feature_names(source_csv: Path) -> List[str]:
    """
    Read symptom feature names (and order) from Symptom-severity.csv.
    Uses the 'Symptom' column; normalizes to lowercase, strips spaces.
    """
    if not source_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {source_csv}")
    df = pd.read_csv(source_csv)
    if "Symptom" not in df.columns:
        raise ValueError("Expected column 'Symptom' in Symptom-severity.csv")
    names = [str(s).strip().lower() for s in df["Symptom"].tolist()]
    names = [n for n in names if n and n != "nan"]
    return names

# ---------- Startup ----------
@app.on_event("startup")
def startup_load():
    global model, label_encoder, FEATURE_NAMES, _last_errors
    _last_errors.clear()

    # Load features
    try:
        FEATURE_NAMES = load_feature_names(SEVERITY_CSV)
        print(f"[startup] Loaded {len(FEATURE_NAMES)} features from {SEVERITY_CSV.name}")
    except Exception as e:
        FEATURE_NAMES = []
        _last_errors["features"] = str(e)
        print(f"[startup][WARN] Feature load failed: {e}")

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[startup] Model loaded from {MODEL_PATH.name}")
    except Exception as e:
        model = None
        _last_errors["model"] = str(e)
        print(f"[startup][ERROR] Model load failed: {e}")

    # Load label encoder
    try:
        label_encoder = joblib.load(ENCODER_PATH)
        print(f"[startup] Label encoder loaded from {ENCODER_PATH.name}")
    except Exception as e:
        label_encoder = None
        _last_errors["encoder"] = str(e)
        print(f"[startup][ERROR] Encoder load failed: {e}")

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

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
        "features_preview": FEATURE_NAMES[:10],
        "model_path": MODEL_PATH.name,
        "encoder_path": ENCODER_PATH.name,
        "feature_source": SEVERITY_CSV.name,
        "classes_preview": (classes[:10] if classes else None),
        "last_errors": _last_errors or None,
    }

@app.post("/predict")
def predict(payload: PredictIn):
    if model is None or label_encoder is None:
        return {"predicted_disease": "MODEL_NOT_LOADED", "probas": None}

    if not FEATURE_NAMES:
        return {"predicted_disease": "FEATURES_NOT_LOADED", "probas": None}

    # Multi-hot encode symptoms in the exact FEATURE_NAMES order
    present = {s.strip().lower() for s in payload.symptoms}
    x = np.zeros((1, len(FEATURE_NAMES)), dtype=float)
    for i, feat in enumerate(FEATURE_NAMES):
        if feat in present:
            x[0, i] = 1.0

    # Predict
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x)[0]  # shape: (n_classes,)
            top_idx = int(np.argmax(prob))
            try:
                top_label = label_encoder.inverse_transform([top_idx])[0]
            except Exception:
                top_label = str(top_idx)
            probas = {}
            for i, p in enumerate(prob):
                try:
                    name = label_encoder.inverse_transform([i])[0]
                except Exception:
                    name = str(i)
                probas[str(name)] = float(p)
            return {"predicted_disease": str(top_label), "probas": probas}
        else:
            idx = int(model.predict(x)[0])
            try:
                label = label_encoder.inverse_transform([idx])[0]
            except Exception:
                label = str(idx)
            return {"predicted_disease": str(label), "probas": None}
    except Exception as e:
        return {"predicted_disease": f"ERROR: {e}", "probas": None}
