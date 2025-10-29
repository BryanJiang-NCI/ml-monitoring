# structured_infer.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col,
    get_json_object,
    regexp_extract,
    regexp_replace,
    lit,
)
from pyspark.sql.types import StringType

# ==========================================================
# 🧩 0. 路径 & Kafka 配置
# ==========================================================
MODEL_DIR = "/opt/spark/work-dir/models/structured_model"
PREPROC_FILE = os.path.join(MODEL_DIR, "preprocessor.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "autoencoder.pth")
THRESH_FILE = os.path.join(MODEL_DIR, "threshold.pkl")

KAFKA_BOOTSTRAP = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"

OUT_PARQUET = "/opt/spark/work-dir/data/structured_infer"
CKPT_DIR = "/opt/spark/work-dir/data/_checkpoints_structured_infer"

# 和训练保持完全一致的特征列集合（按你训练脚本中那份为准）
FEATURE_COLUMNS = [
    # —— 下面这组示例列名请与你训练时使用的列保持一致 ——
    # GitHub
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
    # Public cloud
    "event_id",
    "event_name",
    "username",
    "event_time",
    # App metrics
    "container_name",
    "cpu_perc",
    "mem_perc",
    "mem_usage",
    "net_io",
    "block_io",
    "pids",
    # App logs
    "log_time",
    "log_level",
    "logger_name",
    "log_message",
    # Nginx access
    "client_ip",
    "request_method",
    "request_uri",
    "response_status",
    "body_bytes_sent",
    "request_time",
    "user_agent",
    # Nginx error
    "error_time",
    "error_level",
    "error_detail",
    # 通用
    "source_type",  # 训练里若包含它，一定要保留；不需要可移除
]

# ==========================================================
# 🧩 1. 加载模型组件
# ==========================================================
print(f"🚀 Loading components from {MODEL_DIR} ...")
preprocessor = joblib.load(PREPROC_FILE)  # sklearn ColumnTransformer(Pipeline)
threshold = joblib.load(THRESH_FILE) if os.path.exists(THRESH_FILE) else 0.01


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.0)
        )
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# 用一个假的 1 行输入探测 input_dim
_probe = preprocessor.transform(pd.DataFrame([{c: None for c in FEATURE_COLUMNS}]))

input_dim = _probe.shape[1]

model = AutoEncoder(input_dim=input_dim, hidden_dim=64)
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()
print(f"✅ Model loaded. input_dim={input_dim}, threshold={threshold:.6f}")

# ==========================================================
# 🧩 2. Spark 初始化 & Kafka 源
# ==========================================================
spark = (
    SparkSession.builder.appName("Structured_Streaming_Inference")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)
df = df_kafka.selectExpr("CAST(value AS STRING) AS json_str")

# ==========================================================
# 🧩 3. 通用字段
# ==========================================================
df_base = df.select(
    get_json_object(col("json_str"), "$.source_type").alias("source_type"),
    get_json_object(col("json_str"), "$.timestamp").alias("timestamp"),
    get_json_object(col("json_str"), "$.message").alias("message"),
)

# ==========================================================
# 🧩 4. 各源解析（与 structure_preprocessing.py 一致）
# ==========================================================

# GitHub Commits
df_github_commits = df_base.filter(col("source_type") == "github_commits").select(
    col("source_type"),
    get_json_object(col("message"), "$.sha").alias("commit_sha"),
    get_json_object(col("message"), "$.author").alias("commit_author"),
    get_json_object(col("message"), "$.date").alias("commit_date"),
    get_json_object(col("message"), "$.message").alias("commit_message"),
    get_json_object(col("message"), "$.url").alias("commit_url"),
)

# GitHub Actions
df_github_actions = df_base.filter(col("source_type") == "github_actions").select(
    col("source_type"),
    get_json_object(col("message"), "$.id").alias("action_id"),
    get_json_object(col("message"), "$.name").alias("action_name"),
    get_json_object(col("message"), "$.status").alias("action_status"),
    get_json_object(col("message"), "$.conclusion").alias("action_conclusion"),
    get_json_object(col("message"), "$.created_at").alias("action_created_at"),
    get_json_object(col("message"), "$.url").alias("action_url"),
)

# Public cloud (CloudTrail/CloudWatch)
df_public_cloud = df_base.filter(col("source_type") == "public_cloud").select(
    col("source_type"),
    get_json_object(col("message"), "$.event_id").alias("event_id"),
    get_json_object(col("message"), "$.event_name").alias("event_name"),
    get_json_object(col("message"), "$.username").alias("username"),
    get_json_object(col("message"), "$.event_time").alias("event_time"),
)

# App container metrics
df_app_metrics = df_base.filter(col("source_type") == "app_container_metrics").select(
    col("source_type"),
    get_json_object(col("message"), "$.Container").alias("container_name"),
    get_json_object(col("message"), "$.CPUPerc").alias("cpu_perc"),
    get_json_object(col("message"), "$.MemPerc").alias("mem_perc"),
    get_json_object(col("message"), "$.MemUsage").alias("mem_usage"),
    get_json_object(col("message"), "$.NetIO").alias("net_io"),
    get_json_object(col("message"), "$.BlockIO").alias("block_io"),
    get_json_object(col("message"), "$.PIDs").alias("pids"),
)

# App container logs (JSON 化)
df_app_logs = df_base.filter(col("source_type") == "app_container_log").select(
    col("source_type"),
    get_json_object(col("message"), "$.time").alias("log_time"),
    get_json_object(col("message"), "$.level").alias("log_level"),
    get_json_object(col("message"), "$.logger").alias("logger_name"),
    get_json_object(col("message"), "$.message").alias("log_message"),
)

# Nginx access (JSON 格式)
df_nginx = df_base.filter(col("source_type") == "nginx").select(
    col("source_type"),
    get_json_object(col("message"), "$.remote_addr").alias("client_ip"),
    get_json_object(col("message"), "$.request_method").alias("request_method"),
    get_json_object(col("message"), "$.request_uri").alias("request_uri"),
    get_json_object(col("message"), "$.status").alias("response_status"),
    get_json_object(col("message"), "$.body_bytes_sent").alias("body_bytes_sent"),
    get_json_object(col("message"), "$.request_time").alias("request_time"),
    get_json_object(col("message"), "$.http_user_agent").alias("user_agent"),
)

# Nginx error（多样式简化）
df_nginx_error = df_base.filter(col("source_type") == "nginx_error").select(
    col("source_type"),
    regexp_extract("message", r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})", 1).alias(
        "error_time"
    ),
    regexp_extract("message", r"\[(\w+)\]", 1).alias("error_level"),
    regexp_extract("message", r": (.*)", 1).alias("error_detail"),
)

# 合并统一表
df_final = (
    df_github_commits.unionByName(df_github_actions, allowMissingColumns=True)
    .unionByName(df_public_cloud, allowMissingColumns=True)
    .unionByName(df_app_metrics, allowMissingColumns=True)
    .unionByName(df_app_logs, allowMissingColumns=True)
    .unionByName(df_nginx, allowMissingColumns=True)
    .unionByName(df_nginx_error, allowMissingColumns=True)
)

# 为缺失列补空，确保列对齐
for c in FEATURE_COLUMNS:
    if c not in df_final.columns:
        df_final = df_final.withColumn(c, lit(None).cast(StringType()))

# 只保留训练需要的列 + 便于溯源的字段
df_features = df_final.select(
    *FEATURE_COLUMNS, col("source_type").alias("src_type_runtime")
)


# ==========================================================
# 🧩 5. foreachBatch 推理
# ==========================================================
def infer_batch(batch_df, epoch_id: int):
    pd_df = batch_df.select(*FEATURE_COLUMNS, "src_type_runtime").toPandas()
    if pd_df.empty:
        return

    # 1) 预处理（OneHot + 标准化）
    X = preprocessor.transform(pd_df[FEATURE_COLUMNS])  # ndarray / 或者稀疏
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32)

    # 2) 推理（AutoEncoder 重构误差）
    X_tensor = torch.tensor(X_dense, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X_tensor)
        mse = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()

    # 3) 打标签
    labels = np.where(
        mse > threshold * 1.5,
        "high_anomaly",
        np.where(mse > threshold, "low_anomaly", "normal"),
    )

    # 4) 输出拼接
    out_pdf = pd_df.copy()
    out_pdf["reconstruction_error"] = mse.round(6)
    out_pdf["threshold"] = float(threshold)
    out_pdf["prediction"] = labels

    # 打印部分结果（console 观察）
    print(f"\n=== Inference Epoch {epoch_id} | rows={len(out_pdf)} ===")
    print(
        out_pdf[["src_type_runtime", "prediction", "reconstruction_error"]]
        .head(10)
        .to_string(index=False)
    )
    out_pdf = out_pdf.astype(str)

    # 写 Parquet（Append）
    spark.createDataFrame(out_pdf).write.mode("append").parquet(OUT_PARQUET)


# ==========================================================
# 🧩 6. 启动流
# ==========================================================
query = (
    df_features.writeStream.outputMode("append")
    .foreachBatch(infer_batch)
    .option("checkpointLocation", CKPT_DIR)
    .start()
)

query.awaitTermination()
