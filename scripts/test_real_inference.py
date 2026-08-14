"""Send a complete real goodware dataset row through the local API."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
GOODWARE_PATH = BASE_DIR / "datasets" / "raw" / "goodware" / "goodware.csv"
API_URL = "http://127.0.0.1:5000/predict"

model = joblib.load(MODEL_PATH)

if hasattr(model, "feature_names_in_"):
    columns = list(model.feature_names_in_)
elif "preprocessor" in getattr(model, "named_steps", {}):
    pre = model.named_steps["preprocessor"]
    columns = list(getattr(pre, "feature_names_in_", []))
else:
    columns = []

if not columns:
    raise SystemExit("Could not determine model feature names.")

# Read a reasonable batch and select the first row complete for all
# features expected by the fitted pipeline.
df = pd.read_csv(GOODWARE_PATH, encoding="latin-1", nrows=5000)

missing = [c for c in columns if c not in df.columns]
if missing:
    raise SystemExit(
        "The goodware CSV is missing model features: " + ", ".join(missing)
    )

candidates = df.dropna(subset=columns)

if candidates.empty:
    raise SystemExit(
        "No complete goodware row was found in the first 5000 records."
    )

row = candidates.iloc[0][columns].to_dict()

clean = {}
for key, value in row.items():
    if hasattr(value, "item"):
        value = value.item()
    clean[key] = value

payload = json.dumps({"features": clean}).encode("utf-8")

request = Request(
    API_URL,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urlopen(request, timeout=30) as response:
        print("API response:")
        print(response.read().decode("utf-8"))
except Exception as exc:
    raise SystemExit(
        "Could not reach the local API. Make sure "
        "'python -m src.api.app' is running.\n\n"
        f"Details: {exc}"
    )
