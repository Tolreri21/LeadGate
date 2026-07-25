import pandas as pd
from joblib import load
import json
from pathlib import Path

num_cols = ["age", "balance", "campaign", "previous"]
cat_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
cols = cat_cols + num_cols
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def load_artifacts(models_dir=MODELS_DIR):
    with open(models_dir / "threshold.json") as f:
        threshold = json.load(f)["threshold"]

    pipe = load(models_dir / "model.joblib")
    return pipe, threshold


def predict_lead(model, threshold, payload):
    missing = sorted(set(cols) - payload.keys())
    unexpected = sorted(payload.keys() - set(cols))
    if missing or unexpected:
        raise ValueError(
            f"payload fields off: missing={missing}, unexpected={unexpected}"
        )

    df = pd.DataFrame([payload])
    df[num_cols] = df[num_cols].astype("int")
    df[cat_cols] = df[cat_cols].astype("str")
    proba = model.predict_proba(df)
    subscribe = bool(proba[:, 1][0] >= threshold)
    return {
        "subscribe": subscribe,
        "probability": float(proba[:, 1][0]),
        "threshold": threshold,
    }
