# AI-Based Network Intrusion Detection (NSL-KDD)

Gradient Boosting classifier that labels a network connection as **Normal, DoS, Probe, R2L or U2R**
(anything other than Normal = attack). Companion to the case study report.

## Setup
    pip install -r requirements.txt

## Files
| File | Purpose |
|---|---|
| `train.py` | Trains the model, saves `nids_model.joblib`, `metrics.json`, `confusion_matrix.png` (downloads NSL-KDD if missing) |
| `predict.py` | Command line predictor: `python predict.py sample_connections.csv` |
| `app.py` | Streamlit demo: `streamlit run app.py` |
| `common.py` | Feature list and attack-family mapping |
| `nids_model.joblib` | Pre-trained model |
| `KDDTest+.txt` | Test data used by the demo app |

## Results on KDDTest+ (official test set)
- 5-class: accuracy 79.8%, macro F1 0.60 (DoS F1 0.90, Probe 0.72, R2L 0.39, U2R 0.18)
- Attack vs normal: accuracy 82.2%, precision 96.8%, recall 71.0%, false alarm rate 3.1%

## Limitations
Weak on R2L/U2R and on attack types not seen in training (see case study). NSL-KDD is 1999-era traffic.
