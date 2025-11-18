import os
import json
import re
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support

# ==========================
# 配置
# ==========================
FEATURE_COLUMNS = [
    "source_type",
    "action_status",
    "log_level",
    "log_message",
    "response_status",
    "error_level",
    "container_status",
    "container_status_code",
    "body_bytes_sent",
]

NUMERIC_COLS = ["response_status"]
CATEGORICAL_COLS = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLS]

BASE_DIR = "/opt/spark/work-dir"
TEST_DIR = f"{BASE_DIR}/data/test_labeled_parquet"

SEM_MODEL_NAME = "all-MiniLM-L12-v2"
SEM_SCALER_PATH = f"{BASE_DIR}/models/prediction_model/scaler.pkl"
SEM_MODEL_PATH = f"{BASE_DIR}/models/prediction_model/autoencoder.pth"
SEM_THRESH_PATH = f"{BASE_DIR}/models/prediction_model/threshold.pkl"

STR_MODEL_DIR = f"{BASE_DIR}/models/structured_model"
STR_PREPROC_PATH = f"{STR_MODEL_DIR}/preprocessor.pkl"
STR_MODEL_PATH = f"{STR_MODEL_DIR}/autoencoder.pth"
STR_THRESH_PATH = f"{STR_MODEL_DIR}/threshold.pkl"

METRICS_DIR = f"{BASE_DIR}/metrics"
CSV_PATH = f"{METRICS_DIR}/eval_metrics.csv"
PLOT_PATH = f"{METRICS_DIR}/model_comparison.png"
os.makedirs(METRICS_DIR, exist_ok=True)


# ==========================
# AutoEncoder
# ==========================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ==========================
# semantic 文本构造
# ==========================
def json_to_semantic(text: str) -> str:
    try:
        d = json.loads(text)
        msg = d.get("message", "")

        try:
            msg_json = json.loads(msg)
        except:
            return str(msg)

        if isinstance(msg_json, dict):
            parts = []
            for k, v in msg_json.items():
                if any(t in k.lower() for t in ["time", "timestamp", "date"]):
                    continue
                parts.append(f"{k}={v}")
            return " ".join(parts)

        return str(msg_json)
    except:
        return "[INVALID_JSON]"


# ==========================
# structured 构造
# ==========================
def safe_float(v):
    try:
        return float(v)
    except:
        return 0.0


def parse_structured(row: str):
    try:
        d = json.loads(row)
    except:
        d = {}

    st = d.get("source_type", "")
    msg = d.get("message", {})

    try:
        msg_json = json.loads(msg) if isinstance(msg, str) else msg
    except:
        msg_json = {}

    f = {c: "unknown" for c in CATEGORICAL_COLS}
    f.update({c: 0.0 for c in NUMERIC_COLS})
    f["source_type"] = st

    if st == "github_commits":
        f["commit_author"] = msg_json.get("author", "unknown")
        f["commit_message"] = msg_json.get("message", "unknown")

    elif st == "github_actions":
        f["action_status"] = msg_json.get("status", "unknown")
        f["action_conclusion"] = msg_json.get("conclusion", "unknown")

    elif st == "public_cloud":
        f["event_name"] = msg_json.get("event_name", "unknown")
        f["username"] = msg_json.get("username", "unknown")

    elif st == "app_container_log":
        f["log_level"] = msg_json.get("level", "unknown")
        f["log_message"] = msg_json.get("message", "unknown")

    elif st == "fastapi_status":
        f["container_name"] = msg_json.get("name", "unknown")

    elif st == "nginx_access":
        f["response_status"] = safe_float(msg_json.get("status", 0))

    elif st == "nginx_error":
        text = d.get("message", "")
        m = re.search(r"\[(\w+)\]", text)
        f["error_level"] = m.group(1) if m else "unknown"

    return {k: f.get(k, "unknown") for k in FEATURE_COLUMNS}


# ==========================
# 1. 加载测试集
# ==========================
print(f"📂 Loading labeled test set from: {TEST_DIR}")
files = glob(f"{TEST_DIR}/**/*.parquet", recursive=True)
if not files:
    raise FileNotFoundError("No parquet files found")

df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
print(f"✅ Loaded {len(df)} rows")

y_true = df["label"].astype(int).values
df["semantic_text"] = df["json_str"].apply(json_to_semantic)

# ==========================
# 2. semantic 模型评估
# ==========================
print("🚀 Loading semantic model ...")
sem_encoder = SentenceTransformer(SEM_MODEL_NAME)
sem_scaler = joblib.load(SEM_SCALER_PATH)
sem_threshold = float(joblib.load(SEM_THRESH_PATH))

input_dim = len(sem_scaler.mean_)
sem_model = AutoEncoder(input_dim, 64)
sem_model.load_state_dict(torch.load(SEM_MODEL_PATH, map_location="cpu"))
sem_model.eval()

print("🔍 Evaluating Semantic Model ...")
emb = sem_encoder.encode(
    df["semantic_text"].tolist(), batch_size=64, show_progress_bar=True
)
emb = sem_scaler.transform(emb).astype(np.float32)

t = torch.tensor(emb)
with torch.no_grad():
    recon = sem_model(t)
    mse = ((t - recon) ** 2).mean(dim=1).numpy()

y_pred_sem = (mse > sem_threshold).astype(int)
sem_p, sem_r, sem_f1, _ = precision_recall_fscore_support(
    y_true, y_pred_sem, average="binary"
)
sem_mse_mean = float(np.mean(mse))

print(
    f"✅ Semantic Model: P={sem_p:.4f}, R={sem_r:.4f}, F1={sem_f1:.4f}, Mean MSE={sem_mse_mean:.6f}"
)

# ==========================
# 3. structured
# ==========================
print("🔍 Evaluating Structured Model ...")
struct_rows = [parse_structured(x) for x in df["json_str"]]
df_struct = pd.DataFrame(struct_rows)

df_struct[NUMERIC_COLS] = (
    df_struct[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
)
df_struct[CATEGORICAL_COLS] = df_struct[CATEGORICAL_COLS].fillna("unknown").astype(str)

preproc = joblib.load(STR_PREPROC_PATH)
X = preproc.transform(df_struct[FEATURE_COLUMNS])
X = X.toarray().astype(np.float32) if hasattr(X, "toarray") else X.astype(np.float32)

str_model = AutoEncoder(X.shape[1], 64)
str_model.load_state_dict(torch.load(STR_MODEL_PATH, map_location="cpu"))
str_model.eval()
str_threshold = float(joblib.load(STR_THRESH_PATH))

with torch.no_grad():
    t = torch.tensor(X)
    recon = str_model(t)
    mse_s = ((t - recon) ** 2).mean(dim=1).numpy()

y_pred_str = (mse_s > str_threshold).astype(int)
str_p, str_r, str_f1, _ = precision_recall_fscore_support(
    y_true, y_pred_str, average="binary"
)
str_mse_mean = float(np.mean(mse_s))

print(
    f"✅ Structured Model: P={str_p:.4f}, R={str_r:.4f}, F1={str_f1:.4f}, Mean MSE={str_mse_mean:.6f}"
)

# ==========================
# 保存 CSV + 图表
# ==========================
df_metrics = pd.DataFrame(
    {
        "model": ["semantic", "structured"],
        "Precision": [sem_p, str_p],
        "Recall": [sem_r, str_r],
        "F1": [sem_f1, str_f1],
        "Mean MSE": [sem_mse_mean, str_mse_mean],
    }
)

df_metrics.to_csv(CSV_PATH, index=False)
print(f"📄 Metrics saved to: {CSV_PATH}")

plt.figure(figsize=(10, 6))
x = np.arange(4)
labels = ["Precision", "Recall", "F1", "Mean MSE"]

plt.bar(x - 0.35 / 2, df_metrics.loc[0, labels], 0.35, label="Semantic")
plt.bar(x + 0.35 / 2, df_metrics.loc[1, labels], 0.35, label="Structured")

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Semantic vs Structured Model Performance")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)

print(f"📊 Plot saved to: {PLOT_PATH}")
print("🎉 Evaluation finished.")
