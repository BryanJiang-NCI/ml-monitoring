"""
Spark Streaming Inference (AutoEncoder + TF-IDF + PyTorch)
==========================================================
✅ 加载 TF-IDF 向量器 + AutoEncoder 模型（PyTorch）
✅ 对每条日志构建语义句 → 向量化 → 计算重建误差
✅ 动态加载训练阈值 threshold.pkl
✅ 输出 JSON：prediction / reconstruction_error / threshold / debug_text
✅ 无锁、安全、兼容 Spark Structured Streaming
==========================================================
"""

import os
import json
import joblib
import torch
import torch.nn as nn
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# ==========================================================
# 🧩 Step 0. 路径配置
# ==========================================================
MODEL_DIR = "/opt/spark-apps/models/prediction_model"
MODEL_FILE = os.path.join(MODEL_DIR, "autoencoder.pth")
VEC_FILE = os.path.join(MODEL_DIR, "tfidf.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
THRESH_FILE = os.path.join(MODEL_DIR, "threshold.pkl")

KAFKA_BOOTSTRAP = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"

# ==========================================================
# 🧩 Step 1. 加载模型组件
# ==========================================================
print(f"🚀 Loading model and components from {MODEL_DIR} ...")

vectorizer = joblib.load(VEC_FILE)
scaler = joblib.load(SCALER_FILE)
threshold = joblib.load(THRESH_FILE) if os.path.exists(THRESH_FILE) else 0.01
input_dim = len(vectorizer.get_feature_names_out())


# 定义与训练一致的 AutoEncoder 结构（⚠️ 无 Sigmoid）
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))  # Linear 输出

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = AutoEncoder(input_dim=input_dim, hidden_dim=128)
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()

print(f"✅ AutoEncoder loaded (input_dim={input_dim}, threshold={threshold:.6f})")

# ==========================================================
# 🧩 Step 2. 字段定义与文本构建函数
# ==========================================================
cols_keep = [
    "source_type",
    "ingest_time",
    "access_client",
    "access_method",
    "access_uri",
    "access_status",
    "response_size",
    "request_time",
    "http_referer",
    "user_agent",
    "error_message",
    "error_client",
    "error_server",
    "error_request",
    "error_host",
    "syslog_message",
    "syslog_identifier",
    "syslog_priority",
    "syslog_facility",
    "host",
    "gh_status",
    "gh_conclusion",
    "gh_head_branch",
    "commit_message",
    "cmdb_service",
    "cmdb_domain",
    "cmdb_owner",
    "cmdb_language",
    "cmdb_deployment_env",
    "metric_name",
    "metric_value",
]


def safe_text(x):
    if x is None or str(x).strip() == "":
        return "[NO_DATA]"
    return str(x)


def numeric_to_text(name, val):
    try:
        val = float(val)
    except Exception:
        return f"{name} unknown"
    if val > 0.9:
        return f"{name} high"
    elif val > 0.5:
        return f"{name} medium"
    else:
        return f"{name} normal"


def build_semantic_sentence(row: dict):
    parts = []
    for c in cols_keep:
        val = row.get(c)
        if c in ["metric_value", "access_status", "response_size", "request_time"]:
            val = numeric_to_text(c, val)
        else:
            val = safe_text(val)
        parts.append(f"{c}: {val}")
    return ". ".join(parts)


# ==========================================================
# 🧩 Step 3. 实时推理函数（基于 PyTorch）
# ==========================================================
def predict_json(row_json: str):
    try:
        data = json.loads(row_json)
        text = build_semantic_sentence(data)

        # 向量化 & 标准化
        # X_vec = vectorizer.transform([text]).toarray()
        # X_scaled = scaler.transform(X_vec)
        # X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        X_vec = vectorizer.transform([text])  # 不转 array
        X_scaled = scaler.transform(X_vec)  # 保留稀疏结构
        X_tensor = torch.tensor(X_scaled.toarray(), dtype=torch.float32)

        # 重构误差
        with torch.no_grad():
            reconstructed = model(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).item()

        # 根据阈值判断
        if mse > threshold * 1.5:
            label = "high_anomaly"
        elif mse > threshold:
            label = "low_anomaly"
        else:
            label = "normal"

        return json.dumps(
            {
                "prediction": label,
                "reconstruction_error": round(mse, 6),
                "threshold": round(threshold, 6),
                "source_type": data.get("source_type", "unknown"),
                "debug_text": text[:160],
            }
        )
    except Exception as e:
        return json.dumps({"prediction": "error", "error": str(e)})


predict_udf = udf(predict_json, StringType())

# ==========================================================
# 🧩 Step 4. 初始化 Spark
# ==========================================================
spark = (
    SparkSession.builder.appName("AutoEncoder_TFIDF_Streaming_Inference")
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("✅ Spark initialized.")

# ==========================================================
# 🧩 Step 5. Kafka 数据源
# ==========================================================
df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

df_base = df_kafka.selectExpr("CAST(value AS STRING) as message")

# ==========================================================
# 🧩 Step 6. 推理与输出
# ==========================================================
df_pred = df_base.withColumn("result_json", predict_udf(col("message")))
df_out = df_pred.select(col("message"), col("result_json"))

query = (
    df_out.writeStream.outputMode("append")
    .format("console")
    .option("truncate", False)
    .option("numRows", 5)
    .start()
)

query.awaitTermination()
