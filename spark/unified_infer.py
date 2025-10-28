"""
Unified Streaming Inference (Semantic + Structured Pipelines)
=============================================================
✅ 同时运行语义化 + 结构化推理
✅ 不再使用 stream-stream join（完全兼容 Structured Streaming）
✅ 自动计算 Precision / Recall / F1 / Latency
✅ 输出 CSV 与 PNG 指标折线图
=============================================================
"""

import os
import json
import time
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, get_json_object, regexp_extract, lit, udf
from pyspark.sql.types import StringType
from sentence_transformers import SentenceTransformer

# ==========================================================
# 🧩 0. 配置
# ==========================================================
BASE_DIR = "/opt/spark/work-dir"
KAFKA_BOOTSTRAP = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"

# Semantic model
SEM_MODEL_NAME = "all-MiniLM-L6-v2"
SEM_SCALER = os.path.join(BASE_DIR, "models/prediction_model/scaler.pkl")
SEM_MODEL = os.path.join(BASE_DIR, "models/prediction_model/autoencoder.pth")
SEM_THRESH = os.path.join(BASE_DIR, "models/prediction_model/threshold.pkl")

# Structured model
STR_MODEL_DIR = os.path.join(BASE_DIR, "models/structured_model")
STR_PREPROC = os.path.join(STR_MODEL_DIR, "preprocessor.pkl")
STR_MODEL = os.path.join(STR_MODEL_DIR, "autoencoder.pth")
STR_THRESH = os.path.join(STR_MODEL_DIR, "threshold.pkl")

# Metrics output
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
CSV_PATH = os.path.join(METRICS_DIR, "inference_metrics.csv")
PNG_PATH = os.path.join(METRICS_DIR, "inference_metrics.png")

# ==========================================================
# 🧩 1. 加载模型
# ==========================================================
print("🚀 Loading models ...")

sem_encoder = SentenceTransformer(SEM_MODEL_NAME)
sem_scaler = joblib.load(SEM_SCALER)
sem_threshold = float(joblib.load(SEM_THRESH))


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Semantic
sem_input_dim = len(sem_scaler.mean_) if hasattr(sem_scaler, "mean_") else 384
sem_model = AutoEncoder(sem_input_dim, 128)
sem_model.load_state_dict(torch.load(SEM_MODEL, map_location="cpu"))
sem_model.eval()

# Structured
preprocessor = joblib.load(STR_PREPROC)
threshold = float(joblib.load(STR_THRESH))
FEATURE_COLUMNS = [
    "commit_sha",
    "commit_author",
    "commit_date",
    "commit_message",
    "commit_url",
    "action_id",
    "action_name",
    "action_status",
    "action_conclusion",
    "action_created_at",
    "action_url",
    "event_id",
    "event_name",
    "username",
    "event_time",
    "container_name",
    "cpu_perc",
    "mem_perc",
    "mem_usage",
    "net_io",
    "block_io",
    "pids",
    "log_time",
    "log_level",
    "logger_name",
    "log_message",
    "client_ip",
    "request_method",
    "request_uri",
    "response_status",
    "body_bytes_sent",
    "request_time",
    "user_agent",
    "error_time",
    "error_level",
    "error_detail",
    "source_type",
]
_probe = preprocessor.transform(pd.DataFrame([{c: None for c in FEATURE_COLUMNS}]))
input_dim = _probe.shape[1]
str_model = AutoEncoder(input_dim, 128)
str_model.load_state_dict(torch.load(STR_MODEL, map_location="cpu"))
str_model.eval()

print(
    f"✅ Models loaded. Semantic threshold={sem_threshold:.6f}, Structured threshold={threshold:.6f}"
)

# ==========================================================
# 🧩 2. 初始化 Spark
# ==========================================================
spark = (
    SparkSession.builder.appName("Unified_Streaming_Inference_Final")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ==========================================================
# 🧩 3. Kafka Source
# ==========================================================
df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)
df = df_kafka.selectExpr("CAST(value AS STRING) AS json_str")


# ==========================================================
# 🧩 4. JSON → semantic_text
# ==========================================================
def json_to_semantic(text):
    try:
        d = json.loads(text)
        parts = []
        for k, v in d.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    parts.append(f"{k}.{kk}: {vv}")
            else:
                parts.append(f"{k}: {v}")
        return ". ".join(parts)
    except Exception:
        return "[INVALID_JSON]"


json_to_semantic_udf = udf(json_to_semantic, StringType())
df = df.withColumn("semantic_text", json_to_semantic_udf(col("json_str")))

# ==========================================================
# 🧩 5. 结构化字段解析（语义列随行）
# ==========================================================
df_base = df.select(
    get_json_object(col("json_str"), "$.source_type").alias("source_type"),
    get_json_object(col("json_str"), "$.timestamp").alias("timestamp"),
    get_json_object(col("json_str"), "$.message").alias("message"),
    col("semantic_text"),
)

# GitHub Commits
df_github_commits = df_base.filter(col("source_type") == "github_commits").select(
    "*",
    get_json_object(col("message"), "$.sha").alias("commit_sha"),
    get_json_object(col("message"), "$.author").alias("commit_author"),
    get_json_object(col("message"), "$.date").alias("commit_date"),
    get_json_object(col("message"), "$.message").alias("commit_message"),
    get_json_object(col("message"), "$.url").alias("commit_url"),
)

# GitHub Actions
df_github_actions = df_base.filter(col("source_type") == "github_actions").select(
    "*",
    get_json_object(col("message"), "$.id").alias("action_id"),
    get_json_object(col("message"), "$.name").alias("action_name"),
    get_json_object(col("message"), "$.status").alias("action_status"),
    get_json_object(col("message"), "$.conclusion").alias("action_conclusion"),
    get_json_object(col("message"), "$.created_at").alias("action_created_at"),
    get_json_object(col("message"), "$.url").alias("action_url"),
)

# Public cloud
df_public_cloud = df_base.filter(col("source_type") == "public_cloud").select(
    "*",
    get_json_object(col("message"), "$.event_id").alias("event_id"),
    get_json_object(col("message"), "$.event_name").alias("event_name"),
    get_json_object(col("message"), "$.username").alias("username"),
    get_json_object(col("message"), "$.event_time").alias("event_time"),
)

# App metrics
df_app_metrics = df_base.filter(col("source_type") == "app_container_metrics").select(
    "*",
    get_json_object(col("message"), "$.Container").alias("container_name"),
    get_json_object(col("message"), "$.CPUPerc").alias("cpu_perc"),
    get_json_object(col("message"), "$.MemPerc").alias("mem_perc"),
    get_json_object(col("message"), "$.MemUsage").alias("mem_usage"),
    get_json_object(col("message"), "$.NetIO").alias("net_io"),
    get_json_object(col("message"), "$.BlockIO").alias("block_io"),
    get_json_object(col("message"), "$.PIDs").alias("pids"),
)

# App logs
df_app_logs = df_base.filter(col("source_type") == "app_container_log").select(
    "*",
    get_json_object(col("message"), "$.time").alias("log_time"),
    get_json_object(col("message"), "$.level").alias("log_level"),
    get_json_object(col("message"), "$.logger").alias("logger_name"),
    get_json_object(col("message"), "$.message").alias("log_message"),
)

# Nginx access
df_nginx = df_base.filter(col("source_type") == "nginx").select(
    "*",
    get_json_object(col("message"), "$.remote_addr").alias("client_ip"),
    get_json_object(col("message"), "$.request_method").alias("request_method"),
    get_json_object(col("message"), "$.request_uri").alias("request_uri"),
    get_json_object(col("message"), "$.status").alias("response_status"),
    get_json_object(col("message"), "$.body_bytes_sent").alias("body_bytes_sent"),
    get_json_object(col("message"), "$.request_time").alias("request_time"),
    get_json_object(col("message"), "$.http_user_agent").alias("user_agent"),
)

# Nginx error
df_nginx_error = df_base.filter(col("source_type") == "nginx_error").select(
    "*",
    regexp_extract("message", r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})", 1).alias(
        "error_time"
    ),
    regexp_extract("message", r"\[(\w+)\]", 1).alias("error_level"),
    regexp_extract("message", r": (.*)", 1).alias("error_detail"),
)

df_structured = (
    df_github_commits.unionByName(df_github_actions, allowMissingColumns=True)
    .unionByName(df_public_cloud, allowMissingColumns=True)
    .unionByName(df_app_metrics, allowMissingColumns=True)
    .unionByName(df_app_logs, allowMissingColumns=True)
    .unionByName(df_nginx, allowMissingColumns=True)
    .unionByName(df_nginx_error, allowMissingColumns=True)
)

for c in FEATURE_COLUMNS:
    if c not in df_structured.columns:
        df_structured = df_structured.withColumn(c, lit(None).cast(StringType()))

df_final = df_structured  # ✅ 无需 join，语义列已随行


# ==========================================================
# 🧩 6. foreachBatch 推理
# ==========================================================
def infer_batch(batch_df, epoch_id: int):
    pdf = batch_df.toPandas()
    if pdf.empty:
        return

    start = time.time()

    # --- Semantic inference ---
    sem_embeddings = np.array([sem_encoder.encode(t) for t in pdf["semantic_text"]])
    sem_scaled = sem_scaler.transform(sem_embeddings).astype(np.float32)
    sem_tensor = torch.tensor(sem_scaled)
    with torch.no_grad():
        sem_recon = sem_model(sem_tensor)
        sem_mse = torch.mean((sem_tensor - sem_recon) ** 2, dim=1).numpy()
    sem_pred = np.where(
        sem_mse > sem_threshold * 1.5, 2, np.where(sem_mse > sem_threshold, 1, 0)
    )

    # --- Structured inference ---
    X = preprocessor.transform(pdf[FEATURE_COLUMNS].fillna("").astype(str))
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32)
    X_tensor = torch.tensor(X_dense, dtype=torch.float32)
    with torch.no_grad():
        recon = str_model(X_tensor)
        mse = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
    str_pred = np.where(mse > threshold * 1.5, 2, np.where(mse > threshold, 1, 0))

    latency = (time.time() - start) / len(pdf)
    y_true = np.zeros(len(pdf))

    def metrics(y_true, y_pred):
        return (
            precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_score(y_true, y_pred, average="macro", zero_division=0),
            f1_score(y_true, y_pred, average="macro", zero_division=0),
        )

    sem_p, sem_r, sem_f1 = metrics(y_true, sem_pred)
    str_p, str_r, str_f1 = metrics(y_true, str_pred)

    new_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "batch_id": epoch_id,
        "semantic_precision": sem_p,
        "semantic_recall": sem_r,
        "semantic_f1": sem_f1,
        "structured_precision": str_p,
        "structured_recall": str_r,
        "structured_f1": str_f1,
        "avg_latency": latency,
    }

    if os.path.exists(CSV_PATH):
        dfm = pd.read_csv(CSV_PATH)
        dfm = pd.concat([dfm, pd.DataFrame([new_row])], ignore_index=True)
    else:
        dfm = pd.DataFrame([new_row])
    dfm.to_csv(CSV_PATH, index=False)

    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(dfm["batch_id"], dfm["semantic_f1"], label="Semantic F1")
    plt.plot(dfm["batch_id"], dfm["structured_f1"], label="Structured F1")
    plt.plot(dfm["batch_id"], dfm["avg_latency"], label="Latency (s)")
    plt.xlabel("Batch ID")
    plt.ylabel("Score / Time")
    plt.title("Inference Performance Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PNG_PATH)

    print(
        f"✅ [Batch {epoch_id}] Semantic F1={sem_f1:.3f} | Structured F1={str_f1:.3f} | Latency={latency:.3f}s"
    )


# ==========================================================
# 🧩 7. 启动流
# ==========================================================
query = (
    df_final.writeStream.outputMode("append")
    .foreachBatch(infer_batch)
    .option(
        "checkpointLocation",
        os.path.join(BASE_DIR, "data/_checkpoints_unified_infer_final"),
    )
    .trigger(processingTime="30 seconds")
    .start()
)

print(f"📡 Unified streaming inference started. Topic={KAFKA_TOPIC}")
query.awaitTermination()
