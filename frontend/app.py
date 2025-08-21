# frontend/app.py
# Streamlit UI that auto-loads symptoms from the FastAPI backend (/meta)
import streamlit as st
import requests

st.set_page_config(page_title="Symptom → Disease Predictor", page_icon="🩺")
st.title("🩺 Symptom → Disease Predictor")
st.write("Select symptoms and get a predicted disease from the model.")

# 1) Set your backend URL (use the PUBLIC Codespaces URL for port 8000)
backend_url = st.text_input(
    "Backend URL",
    value="https://<your-codespaces-host>-8000.app.github.dev",  # ← replace with your actual public URL
    help="FastAPI server URL (must be publicly reachable from your browser).",
)

# 2) Fetch feature list from /meta so it always matches the model
@st.cache_data(ttl=300)
def fetch_features(url: str):
    r = requests.get(f"{url.rstrip('/')}/meta", timeout=15)
    r.raise_for_status()
    m = r.json()
    if not m.get("n_features") or m.get("n_features", 0) == 0:
        raise RuntimeError("Backend reports zero features. Check /meta on the backend.")
    # the backend returns only a preview; fetch full list by requesting again if needed
    # in our backend, FEATURE_NAMES is used internally, so expose all via a helper:
    # If you don't have that, we can fall back by fetching a separate endpoint
    # For now, try to load from meta if available:
    # (If your /meta doesn't include full list, add a /features endpoint in backend.)
    # We'll make a best effort by calling /meta and if needed, /features:
    names = m.get("features_preview", [])
    # If preview < n_features, try /features (optional endpoint)
    if len(names) < m.get("n_features", 0):
        try:
            r2 = requests.get(f"{url.rstrip('/')}/features", timeout=15)
            if r2.ok:
                names = r2.json().get("features", names)
        except Exception:
            pass
    if not names:
        raise RuntimeError("Could not load symptom features from backend.")
    # Normalize to strings
    return [str(x).strip() for x in names]

# 3) Try to load features
features = []
load_error = None
if backend_url:
    try:
        features = fetch_features(backend_url)
    except Exception as e:
        load_error = str(e)

if load_error:
    st.error(f"Could not load symptoms from backend: {load_error}")
else:
    selected = st.multiselect("Select your symptoms:", options=features)

    if st.button("Predict"):
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            try:
                payload = {"symptoms": selected}
                resp = requests.post(f"{backend_url.rstrip('/')}/predict", json=payload, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"**Predicted disease:** {data.get('predicted_disease', 'N/A')}")
                probas = data.get("probas") or {}
                if probas:
                    st.subheader("Top probabilities")
                    # sort by prob desc and show top 10
                    for k, v in sorted(probas.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                        st.write(f"- {k}: {v:.3f}")
            except Exception as e:
                st.error(f"Request failed: {e}")
