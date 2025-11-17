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
# 0. 路径配置
# ==========================
BASE_DIR = "/opt/spark/work-dir"

# 测试集目录（你之前 label 后的输出目录）
TEST_DIR = os.path.join(BASE_DIR, "data/test_labeled_parquet")

# 语义模型
SEM_MODEL_NAME = "all-MiniLM-L12-v2"
SEM_SCALER_PATH = os.path.join(BASE_DIR, "models/prediction_model/scaler.pkl")
SEM_MODEL_PATH = os.path.join(BASE_DIR, "models/prediction_model/autoencoder.pth")
SEM_THRESH_PATH = os.path.join(BASE_DIR, "models/prediction_model/threshold.pkl")

# 结构化模型
STR_MODEL_DIR = os.path.join(BASE_DIR, "models/structured_model")
STR_PREPROC_PATH = os.path.join(STR_MODEL_DIR, "preprocessor.pkl")
STR_MODEL_PATH = os.path.join(STR_MODEL_DIR, "autoencoder.pth")
STR_THRESH_PATH = os.path.join(STR_MODEL_DIR, "threshold.pkl")

# 指标输出
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
CSV_PATH = os.path.join(METRICS_DIR, "eval_metrics.csv")
PLOT_PATH = os.path.join(METRICS_DIR, "model_comparison.png")


# ==========================
# 1. 公用：AutoEncoder 定义
# ==========================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ==========================
# 2. 语义预处理：json_str → semantic_text
#    保持与你当前训练脚本一致（key=value + 忽略 time 字段）
# ==========================
def json_to_semantic(text: str) -> str:
    try:
        data = json.loads(text)
        # 你的消息结构：{"source_type": "...", "timestamp": "...", "message": "..."}
        if isinstance(data, dict) and "message" in data:
            msg = data["message"]
            # message 大多是 JSON 字符串
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except Exception:
                    # 不是 JSON 的话，直接拿原文
                    return str(msg)
        else:
            msg = data

        parts = []
        if isinstance(msg, dict):
            for k, v in msg.items():
                # 和训练时一样：过滤时间相关字段
                if any(
                    t in k.lower() for t in ["time", "timestamp", "date", "created_at"]
                ):
                    continue
                parts.append(f"{k}={v}")
        else:
            parts.append(str(msg))

        return " ".join(parts)
    except Exception:
        return "[INVALID_JSON]"


# ==========================
# 3. 结构化特征：json_str → FEATURE_COLUMNS
#    完全按你贴的 Spark 结构化脚本逻辑来解析
# ==========================
preprocessor = joblib.load(STR_PREPROC_PATH)
FEATURE_COLUMNS = list(preprocessor.feature_names_in_)


def build_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for raw in df["json_str"]:
        try:
            d = json.loads(raw)
        except Exception:
            d = {}

        st = d.get("source_type", "")
        if st != "app_container_metrics":
            msg = d.get("message", {})

        if isinstance(msg, str):
            try:
                msg_json = json.loads(msg)
            except Exception:
                # nginx_error 这种本身是纯字符串日志
                msg_json = {}
        elif isinstance(msg, dict):
            msg_json = msg
        else:
            msg_json = {}

        # 初始化一行，所有 FEATURE_COLUMNS 先置空字符串
        f = {col: "unknown" for col in FEATURE_COLUMNS}
        f["source_type"] = st

        # --- GitHub Commits ---
        if st == "github_commits":
            f["commit_email"] = msg_json.get("email", "")
            f["commit_author"] = msg_json.get("author", "")
            f["commit_repository"] = msg_json.get("repository", "")

        # --- GitHub Actions ---
        elif st == "github_actions":
            f["action_event"] = msg_json.get("event", "")
            f["action_name"] = msg_json.get("name", "")
            f["action_pipeline_file"] = msg_json.get("pipeline_file", "")
            f["action_build_branch"] = msg_json.get("build_branch", "")
            f["action_status"] = msg_json.get("status", "")
            f["action_conclusion"] = msg_json.get("conclusion", "")
            f["action_repository"] = msg_json.get("repository", "")

        # --- Public Cloud (CloudTrail / CloudWatch) ---
        elif st == "public_cloud":
            f["event_name"] = msg_json.get("event_name", "")
            f["username"] = msg_json.get("username", "")

        # --- App Container Metrics ---
        elif st == "app_container_metrics":
            f["device"] = msg_json.get("device", "")
            f["kind"] = msg_json.get("kind", "")
            f["name"] = msg_json.get("name", "")
            f["value"] = msg_json.get("value", "")

        # --- App Container Logs ---
        elif st == "app_container_log":
            f["service_name"] = msg_json.get("service_name", "")
            f["log_level"] = msg_json.get("level", "")
            f["logger_name"] = msg_json.get("logger", "")
            f["log_message"] = msg_json.get("message", "")

        # --- FastAPI Health / Heartbeat ---
        elif st == "fastapi_status":
            f["container_name"] = msg_json.get("name", "")
            f["container_status"] = msg_json.get("status", "")
            f["container_status_code"] = msg_json.get("status_code", "")
            f["container_url"] = msg_json.get("url", "")
            f["container_value"] = msg_json.get("value", "")
            f["container_message"] = msg_json.get("message", "")

        # --- Nginx Access ---
        elif st == "nginx_access":
            print(msg_json)
            f["client_ip"] = msg_json.get("remote_addr", "")
            f["request_method"] = msg_json.get("request_method", "")
            f["request_uri"] = msg_json.get("request_uri", "")
            f["response_status"] = msg_json.get("status", "")
            f["body_bytes_sent"] = msg_json.get("body_bytes_sent", "")
            f["request_time"] = msg_json.get("request_time", "")
            f["user_agent"] = msg_json.get("http_user_agent", "")

        # --- Nginx Error ---
        elif st == "nginx_error":
            msg_text = d.get("message", "")
            m_level = re.search(r"\[(\w+)\]", msg_text)
            m_detail = re.search(r": (.*)", msg_text)
            f["error_level"] = m_level.group(1) if m_level else ""
            f["error_detail"] = m_detail.group(1) if m_detail else ""

        rows.append(f)

    return pd.DataFrame(rows)


# ==========================
# 4. 加载测试集
# ==========================
print(f"📂 Loading labeled test set from: {TEST_DIR}")
parquet_files = glob(os.path.join(TEST_DIR, "*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"No parquet files found in {TEST_DIR}")

df_list = [pd.read_parquet(p) for p in parquet_files]
df = pd.concat(df_list, ignore_index=True)
print(f"✅ Test samples loaded: {len(df)}")

if "label" not in df.columns:
    raise ValueError(
        "❌ 'label' column not found in test dataset. 请确认你已经打好标签。"
    )

y_true = df["label"].astype(int).values

# 生成 semantic_text（如果还没有）
if "semantic_text" not in df.columns:
    print("🔧 semantic_text not found. Building from json_str ...")
    df["semantic_text"] = df["json_str"].apply(json_to_semantic)
    print("✅ semantic_text generated.")


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
# 6. 加载 结构化模型
# ==========================
print("🚀 Loading structured model ...")

print(preprocessor.feature_names_in_)

str_threshold = float(joblib.load(STR_THRESH_PATH))

# 用一个 dummy 行推断 input_dim
dummy_df = pd.DataFrame([{c: "" for c in FEATURE_COLUMNS}])
dummy_X = preprocessor.transform(dummy_df)
if hasattr(dummy_X, "toarray"):
    dummy_X = dummy_X.toarray()
str_input_dim = dummy_X.shape[1]

str_model = AutoEncoder(input_dim=str_input_dim, hidden_dim=64)
str_model.load_state_dict(torch.load(STR_MODEL_PATH, map_location="cpu"))
str_model.eval()
print(
    f"✅ Structured model loaded. input_dim={str_input_dim}, hidden_dim=64, threshold={str_threshold:.6f}"
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
# 8. 评估：结构化模型
# ==========================
print("🔍 Building structured features for Structured Model ...")
df_struct = build_structured_features(df)

# 保证列齐全
for c in FEATURE_COLUMNS:
    if c not in df_struct.columns:
        df_struct[c] = ""

print("🔍 Evaluating Structured Model ...")
X_struct = preprocessor.transform(df_struct[FEATURE_COLUMNS].fillna("").astype(str))
if hasattr(X_struct, "toarray"):
    X_struct = X_struct.toarray().astype(np.float32)
else:
    X_struct = np.asarray(X_struct, dtype=np.float32)

X_tensor = torch.tensor(X_struct)
with torch.no_grad():
    recon = str_model(X_tensor)
    str_mse = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()

y_pred_str = (str_mse > str_threshold).astype(int)

str_precision, str_recall, str_f1, _ = precision_recall_fscore_support(
    y_true, y_pred_str, average="binary", zero_division=0
)
str_mse_mean = float(np.mean(str_mse))

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
