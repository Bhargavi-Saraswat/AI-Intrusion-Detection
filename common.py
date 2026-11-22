"""Shared constants and helpers for the NSL-KDD intrusion detection model."""
import pandas as pd

COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
 "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
 "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
 "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
 "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
 "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLS if c not in CATEGORICAL]
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

_DOS = "back land neptune pod smurf teardrop mailbomb apache2 processtable udpstorm worm".split()
_PROBE = "ipsweep nmap portsweep satan mscan saint".split()
_U2R = "buffer_overflow loadmodule perl rootkit ps sqlattack xterm".split()

def family(label: str) -> str:
    if label == "normal": return "Normal"
    if label in _DOS: return "DoS"
    if label in _PROBE: return "Probe"
    if label in _U2R: return "U2R"
    return "R2L"   # remaining attack names in NSL-KDD are all R2L

def load_nsl(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLS + ["label", "difficulty"])
    df["category"] = df["label"].map(family)
    return df
