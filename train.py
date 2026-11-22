"""Train the intrusion detection model (5-class: Normal/DoS/Probe/R2L/U2R) and save it.
Usage:  python train.py
Downloads NSL-KDD from GitHub if the files are not in this folder."""
import json, os, urllib.request, warnings
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from common import CATEGORICAL, CLASSES, COLS, NUMERIC, load_nsl
warnings.filterwarnings("ignore")

BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
for f in ("KDDTrain+.txt", "KDDTest+.txt"):
    if not os.path.exists(f):
        print("downloading", f); urllib.request.urlretrieve(BASE + f.replace("+", "%2B"), f)

train, test = load_nsl("KDDTrain+.txt"), load_nsl("KDDTest+.txt")
model = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                                ("num", "passthrough", NUMERIC)])),
    ("clf", HistGradientBoostingClassifier(random_state=42)),
])
model.fit(train[COLS], train["category"])
pred = model.predict(test[COLS])

y = test["category"]

bin_true = (test["category"] != "Normal").astype(int).values
bin_pred = (pd.Series(pred) != "Normal").astype(int).values
metrics = {
    "five_class": {"accuracy": accuracy_score(y, pred),
                   "macro_f1": f1_score(y, pred, average="macro"),
                   "weighted_f1": f1_score(y, pred, average="weighted"),
                   "report": classification_report(y, pred, output_dict=True, zero_division=0)},
    "binary_attack_vs_normal": {"accuracy": accuracy_score(bin_true, bin_pred),
                                "precision": precision_score(bin_true, bin_pred),
                                "recall": recall_score(bin_true, bin_pred),
                                "f1": f1_score(bin_true, bin_pred),
                                "false_alarm_rate": float(((bin_pred == 1) & (bin_true == 0)).sum() / (bin_true == 0).sum())},
}
json.dump(metrics, open("metrics.json", "w"), indent=2, default=float)
joblib.dump(model, "nids_model.joblib", compress=3)

cm = confusion_matrix(y, pred, labels=CLASSES)
plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("NIDS model - KDDTest+"); plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print(json.dumps({k: {m: round(v, 4) for m, v in d.items() if m != "report"} for k, d in metrics.items()}, indent=2))
print(classification_report(y, pred, zero_division=0))
