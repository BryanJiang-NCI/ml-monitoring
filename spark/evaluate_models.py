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
# 固定字段（必须与训练脚本一致）
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

# ==========================
# 路径
# ==========================
BASE_DIR = "/opt/spark/work-dir"
TEST_DIR = os.path.join(BASE_DIR, "data/test_labeled_parquet")

SEM_MODEL_NAME = "all-MiniLM-L12-v2"
SEM_SCALER_PATH = os.path.join(BASE_DIR, "models/prediction_model/scaler.pkl")
SEM_MODEL_PATH = os.path.join(BASE_DIR, "models/prediction_model/autoencoder.pth")
SEM_THRESH_PATH = os.path.join(BASE_DIR, "models/prediction_model/threshold.pkl")

STR_MODEL_DIR = os.path.join(BASE_DIR, "models/structured_model")
STR_PREPROC_PATH = os.path.join(STR_MODEL_DIR, "preprocessor.pkl")
STR_MODEL_PATH = os.path.join(STR_MODEL_DIR, "autoencoder.pth")
STR_THRESH_PATH = os.path.join(STR_MODEL_DIR, "threshold.pkl")

METRICS_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
CSV_PATH = os.path.join(METRICS_DIR, "eval_metrics.csv")
PLOT_PATH = os.path.join(METRICS_DIR, "model_comparison.png")


# ==========================
# AutoEncoder 定义
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
# semantic 构造
# ==========================
def json_to_semantic(text: str) -> str:
    try:
        d = json.loads(text)
        msg = d.get("message", "")

        try:
            msg_json = json.loads(msg)
        except:
            return str(msg)

        parts = []
        if isinstance(msg_json, dict):
            for k, v in msg_json.items():
                if any(t in k.lower() for t in ["time", "timestamp", "date"]):
                    continue
                parts.append(f"{k}={v}")
        else:
            parts.append(str(msg_json))

        return " ".join(parts)
    except:
        return "[INVALID_JSON]"


# ==========================
# structured 构造（完全与训练保持一致）
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

    if isinstance(msg, str):
        try:
            msg_json = json.loads(msg)
        except:
            msg_json = {}
    else:
        msg_json = msg

    f = {}
    for col in CATEGORICAL_COLS:
        f[col] = "unknown"
    for col in NUMERIC_COLS:
        f[col] = 0.0

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
        m1 = re.search(r"\[(\w+)\]", text)
        f["error_level"] = m1.group(1) if m1 else "unknown"

    return {k: f.get(k, "unknown") for k in FEATURE_COLUMNS}


# ==========================
# 1. 加载测试集
# ==========================
print(f"📂 Loading labeled test set from: {TEST_DIR}")
files = glob(os.path.join(TEST_DIR, "**/*.parquet"), recursive=True)
if not files:
    raise FileNotFoundError(f"No parquet files found in {TEST_DIR}")

df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)

print(f"✅ Loaded {len(df)} rows")

y_true = df["label"].astype(int).values
# y_true = 1 - y_true

df["semantic_text"] = df["json_str"].apply(json_to_semantic)


# ==========================
# 5. 加载 语义模型
# ==========================
print("🚀 Loading semantic model ...")
sem_encoder = SentenceTransformer(SEM_MODEL_NAME)
sem_scaler = joblib.load(SEM_SCALER_PATH)
sem_threshold = float(joblib.load(SEM_THRESH_PATH))

sem_input_dim = len(sem_scaler.mean_) if hasattr(sem_scaler, "mean_") else 384
sem_model = AutoEncoder(input_dim=sem_input_dim, hidden_dim=64)
sem_model.load_state_dict(torch.load(SEM_MODEL_PATH, map_location="cpu"))
sem_model.eval()
print(
    f"✅ Semantic model loaded. input_dim={sem_input_dim}, hidden_dim=64, threshold={sem_threshold:.6f}"
)
# ==========================
# 7. 评估：语义模型
# ==========================
print("🔍 Evaluating Semantic Model ...")
texts = df["semantic_text"].tolist()
embeddings = np.array(sem_encoder.encode(texts, batch_size=64, show_progress_bar=True))
emb_scaled = sem_scaler.transform(embeddings).astype(np.float32)

sem_tensor = torch.tensor(emb_scaled)
with torch.no_grad():
    sem_recon = sem_model(sem_tensor)
    sem_mse = ((sem_tensor - sem_recon) ** 2).mean(dim=1).cpu().numpy()

# 二分类：大于阈值 => 异常(1)，否则正常(0)
y_pred_sem = (sem_mse > sem_threshold).astype(int)

sem_precision, sem_recall, sem_f1, _ = precision_recall_fscore_support(
    y_true, y_pred_sem, average="binary", zero_division=0
)
sem_mse_mean = float(np.mean(sem_mse))

print(
    f"✅ Semantic Model: "
    f"Precision={sem_precision:.4f}, Recall={sem_recall:.4f}, "
    f"F1={sem_f1:.4f}, Mean MSE={sem_mse_mean:.6f}"
)


# ==========================
# 3. Structured Model
# ==========================
struct_rows = [parse_structured(x) for x in df["json_str"]]
df_struct = pd.DataFrame(struct_rows)

for c in NUMERIC_COLS:
    df_struct[c] = pd.to_numeric(df_struct[c], errors="coerce").fillna(0.0)
for c in CATEGORICAL_COLS:
    df_struct[c] = df_struct[c].fillna("unknown").astype(str)

print("🔍 Evaluating Structured Model ...")

preprocessor = joblib.load(STR_PREPROC_PATH)
X_struct = preprocessor.transform(df_struct[FEATURE_COLUMNS])
if hasattr(X_struct, "toarray"):
    X_struct = X_struct.toarray().astype(np.float32)
else:
    X_struct = X_struct.astype(np.float32)

str_model = AutoEncoder(X_struct.shape[1], 64)
str_model.load_state_dict(torch.load(STR_MODEL_PATH, map_location="cpu"))
str_model.eval()

str_threshold = float(joblib.load(STR_THRESH_PATH))

with torch.no_grad():
    inp = torch.tensor(X_struct)
    out = str_model(inp)
    mse_struct = ((inp - out) ** 2).mean(dim=1).numpy()

y_pred_str = (mse_struct > str_threshold).astype(int)

str_precision, str_recall, str_f1, _ = precision_recall_fscore_support(
    y_true, y_pred_str, average="binary", zero_division=0
)
str_mse_mean = float(np.mean(mse_struct))

print(
    f"✅ Structured Model: "
    f"Precision={str_precision:.4f}, Recall={str_recall:.4f}, "
    f"F1={str_f1:.4f}, Mean MSE={str_mse_mean:.6f}"
)


# ==========================
# 9. 保存指标 + 画图
# ==========================
metrics = {
    "model": ["semantic", "structured"],
    "precision": [sem_precision, str_precision],
    "recall": [sem_recall, str_recall],
    "f1": [sem_f1, str_f1],
    "mean_mse": [sem_mse_mean, str_mse_mean],
}
df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv(CSV_PATH, index=False)
print(f"📄 Metrics saved to: {CSV_PATH}")

# --- 画对比图 ---
x = np.arange(4)  # 4 个指标
width = 0.35

semantic_vals = [sem_precision, sem_recall, sem_f1, sem_mse_mean]
structured_vals = [str_precision, str_recall, str_f1, str_mse_mean]

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, semantic_vals, width, label="Semantic")
plt.bar(x + width / 2, structured_vals, width, label="Structured")

plt.xticks(x, ["Precision", "Recall", "F1", "Mean MSE"])
plt.ylabel("Score")
plt.title("Semantic vs Structured Model Performance")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(PLOT_PATH)

print(f"📊 Plot saved to: {PLOT_PATH}")
print("🎉 Evaluation finished.")
