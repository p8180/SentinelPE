"""Export one complete goodware and one complete malware row for deployment."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MODEL = BASE / "models" / "best_model.pkl"
GOOD = BASE / "datasets" / "raw" / "goodware" / "goodware.csv"
MAL = BASE / "datasets" / "raw" / "malware-by-day"
OUT = BASE / "data" / "demo_samples.json"

model = joblib.load(MODEL)
if hasattr(model, "feature_names_in_"):
    columns = list(model.feature_names_in_)
else:
    pre = getattr(model, "named_steps", {}).get("preprocessor")
    columns = list(getattr(pre, "feature_names_in_", [])) if pre is not None else []
if not columns:
    raise SystemExit("Could not determine fitted model feature names.")

def find_goodware():
    df = pd.read_csv(GOOD, encoding="latin-1", nrows=5000)
    if any(c not in df.columns for c in columns):
        return None
    x = df.dropna(subset=columns)
    return x.iloc[0][columns].to_dict() if not x.empty else None

def find_malware():
    for path in sorted(MAL.glob("*.csv")):
        try:
            df = pd.read_csv(path, encoding="latin-1", nrows=500)
        except Exception:
            continue
        if any(c not in df.columns for c in columns):
            continue
        x = df.dropna(subset=columns)
        if not x.empty:
            return x.iloc[0][columns].to_dict(), path.name
    return None, None

good = find_goodware()
mal, mal_source = find_malware()
if good is None or mal is None:
    raise SystemExit("Could not find complete demonstration rows.")

def clean(row):
    out = {}
    for k, v in row.items():
        if hasattr(v, "item"):
            v = v.item()
        out[k] = v
    return out

payload = {
    "goodware": {"features": clean(good), "source": "Goodware demonstration sample"},
    "malware": {"features": clean(mal), "source": f"Malware demonstration sample · {mal_source}"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
print(f"Wrote {OUT}")
print(f"Malware source: {mal_source}")
