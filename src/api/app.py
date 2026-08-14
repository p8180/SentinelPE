"""SentinelPE demo-ready Flask application."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best_model.pkl"))
DEMO_SAMPLES_PATH = BASE_DIR / "data" / "demo_samples.json"

app = Flask(__name__)
_model = None
_demo_cache = {}


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Production model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def expected_columns():
    estimator = load_model()
    if hasattr(estimator, "feature_names_in_"):
        return list(estimator.feature_names_in_)
    pre = getattr(estimator, "named_steps", {}).get("preprocessor")
    if pre is not None and hasattr(pre, "feature_names_in_"):
        return list(pre.feature_names_in_)
    return []


def complete_row(df: pd.DataFrame, columns):
    if any(c not in df.columns for c in columns):
        return None
    candidates = df.dropna(subset=columns)
    if candidates.empty:
        return None
    return candidates.iloc[0][columns].to_dict()


def get_demo_row(kind: str):
    if kind not in {"goodware", "malware"}:
        raise ValueError("Unknown demo sample.")

    if kind in _demo_cache:
        return _demo_cache[kind]

    if not DEMO_SAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"Demo samples not found: {DEMO_SAMPLES_PATH}"
        )

    import json

    data = json.loads(
        DEMO_SAMPLES_PATH.read_text(encoding="utf-8")
    )

    if kind not in data:
        raise ValueError(
            f"Demo sample '{kind}' not found."
        )

    sample = data[kind]

    if "features" not in sample:
        raise ValueError(
            f"Demo sample '{kind}' has no features."
        )

    _demo_cache[kind] = sample
    return sample


def predict_features(features):
    estimator = load_model()
    frame = pd.DataFrame([features])[expected_columns()]
    prediction = int(estimator.predict(frame)[0])
    result = {
        "prediction": prediction,
        "label": "malware" if prediction == 1 else "goodware",
    }
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(frame)[0]
        result["probability_goodware"] = float(p[0])
        result["probability_malware"] = float(p[1])
    return result


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    try:
        estimator = load_model()
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "model_type": type(estimator).__name__,
            "feature_count": len(expected_columns()),
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 503


@app.post("/demo-predict")
def demo_predict():
    payload = request.get_json(silent=True) or {}
    kind = payload.get("sample")
    try:
        sample = get_demo_row(kind)
        result = predict_features(sample["features"])
        result["source"] = sample["source"]
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a JSON object."
        }), 400

    features = payload.get("features")

    if not isinstance(features, dict) or not features:
        return jsonify({
            "error": "Provide a non-empty 'features' object."
        }), 400

    try:
        return jsonify(predict_features(features))
    except Exception as exc:
        return jsonify({
            "error": "Inference failed.",
            "details": str(exc)
        }), 422

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
