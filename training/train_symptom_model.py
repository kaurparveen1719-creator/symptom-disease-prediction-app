# training/train_symptom_model.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import re

ROOT = Path(__file__).resolve().parents[1]   # repo root
TRAIN_DIR = ROOT / "training"
BACKEND_DIR = ROOT / "backend"

DATASET = TRAIN_DIR / "dataset.csv"              # your table with symptom text columns + target

# -------- helpers --------
def norm_symptom(s: str) -> str:
    """normalize symptom strings to a consistent token: lowercase, underscores, no extra spaces"""
    s = str(s).strip().lower()
    if s in ("", "nan", "none", "null"):
        return ""
    # unify separators and remove non-word chars to underscores
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# -------- load --------
if not DATASET.exists():
    raise FileNotFoundError(f"Missing {DATASET}. Put your dataset into {TRAIN_DIR}")

df = pd.read_csv(DATASET)

# 1) find target column
possible_targets = [c for c in df.columns if c.lower() in ("prognosis", "disease", "label", "target")]
if not possible_targets:
    raise ValueError("Could not find target column (expected one of: prognosis / disease / label / target).")
target_col = possible_targets[0]

# 2) find symptom TEXT columns (common: Symptom_1 ... Symptom_17)
symptom_text_cols = [c for c in df.columns if c != target_col and df[c].dtype == object]
if not symptom_text_cols:
    # if object dtype detection failed (sometimes pandas infers category), fallback by name
    symptom_text_cols = [c for c in df.columns if c.lower().startswith("symptom_")]

if not symptom_text_cols:
    raise ValueError("Could not find symptom text columns (e.g., Symptom_1 ...).")

print(f"[info] target: {target_col}")
print(f"[info] symptom text columns ({len(symptom_text_cols)}): {symptom_text_cols[:6]}{' ...' if len(symptom_text_cols)>6 else ''}")

# 3) build vocabulary of unique symptoms
all_syms = set()
for c in symptom_text_cols:
    vals = df[c].astype(str).map(norm_symptom)
    all_syms.update([v for v in vals.unique() if v])

feature_names = sorted(all_syms)
if not feature_names:
    raise ValueError("No valid symptoms found after normalization.")

print(f"[info] unique symptoms (features): {len(feature_names)}")

# 4) multi-hot encode rows
sym_index = {s: i for i, s in enumerate(feature_names)}
X = np.zeros((len(df), len(feature_names)), dtype=np.float32)

for row_idx, row in df[symptom_text_cols].iterrows():
    for s in row.values:
        ns = norm_symptom(s)
        if ns and ns in sym_index:
            X[row_idx, sym_index[ns]] = 1.0

# 5) encode labels
y_text = df[target_col].astype(str).map(str.strip)
le = LabelEncoder()
y = le.fit_transform(y_text)

# 6) split, train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=2000, n_jobs=None)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"[info] holdout accuracy: {acc:.3f}")
print(f"[info] classes: {list(le.classes_)[:5]} ... total={len(le.classes_)}")

# 7) save artifacts for backend
BACKEND_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, BACKEND_DIR / "symptom_model.pkl")
joblib.dump(le, BACKEND_DIR / "label_encoder.pkl")
pd.Series(feature_names, name="symptom").to_csv(BACKEND_DIR / "feature_names.csv", index=False)
print(f"[info] saved model, encoder, feature_names.csv into {BACKEND_DIR}")
