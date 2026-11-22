"""Classify network connections with the trained model.
Usage:  python predict.py connections.csv [-o results.csv]
Input CSV: NSL-KDD format, 41 feature columns (with or without a header row;
extra label/difficulty columns are ignored)."""
import argparse
import joblib, pandas as pd
from common import COLS

def load_connections(path) -> pd.DataFrame:
    """Read an NSL-KDD style CSV (path or file-like), with or without a header row."""
    df = pd.read_csv(path, header=None)
    if str(df.iloc[0, 0]) == COLS[0]:                 # header row present
        if hasattr(path, "seek"): path.seek(0)
        return pd.read_csv(path)[COLS]
    df = df.iloc[:, :41]
    df.columns = COLS
    return df

def classify(model, X: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(X[COLS])
    classes = list(model.classes_)
    out = X.copy()
    out["prediction"] = [classes[i] for i in proba.argmax(1)]
    out["confidence"] = proba.max(1).round(3)
    out["is_attack"] = out["prediction"] != "Normal"
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("-o", "--out", default="results.csv")
    a = ap.parse_args()
    res = classify(joblib.load("nids_model.joblib"), load_connections(a.csv))
    res.to_csv(a.out, index=False)
    print(res["prediction"].value_counts().to_string()); print(f"\nSaved {a.out}")
