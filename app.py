"""Streamlit demo:  streamlit run app.py"""
import joblib, pandas as pd, streamlit as st
from common import COLS, load_nsl
from predict import classify, load_connections

st.set_page_config(page_title="AI Network Intrusion Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-Based Network Intrusion Detection")
st.caption("Gradient Boosting model trained on NSL-KDD. Classifies connections as Normal, DoS, Probe, R2L or U2R.")

@st.cache_resource
def get_model(): return joblib.load("nids_model.joblib")
@st.cache_data
def get_sample(): return load_nsl("KDDTest+.txt")

model = get_model()
src = st.radio("Input", ["Random sample from KDDTest+", "Upload CSV"], horizontal=True)
if src == "Upload CSV":
    up = st.file_uploader("NSL-KDD format CSV (41 feature columns)", type="csv")
    data = load_connections(up) if up else None
    truth = None
else:
    n = st.slider("Number of connections", 10, 500, 100)
    seed = st.number_input("Random seed", 0, 9999, 42)
    s = get_sample().sample(n, random_state=int(seed)); data, truth = s[COLS], s["category"].values

if data is not None:
    res = classify(model, data)
    c1, c2, c3 = st.columns(3)
    c1.metric("Connections analysed", len(res)); c2.metric("Flagged as attack", int(res.is_attack.sum()))
    c3.metric("Alert rate", f"{res.is_attack.mean():.0%}")
    if truth is not None:
        res.insert(0, "actual", truth)
        st.metric("Accuracy on this sample", f"{(res.actual == res.prediction).mean():.1%}")
    st.bar_chart(res["prediction"].value_counts())
    show = ["actual"] if truth is not None else []
    st.dataframe(res[show + ["prediction", "confidence", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"]],
                 width="stretch")
    st.download_button("Download results", res.to_csv(index=False), "results.csv", "text/csv")
st.info("Note: the model detects known attack patterns well but misses many novel attacks (see case study).")
