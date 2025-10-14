"""
Spark Streaming Inference (Semantic Pipeline + AutoEncoder)
============================================================
✅ 与语义向量落地逻辑完全一致（同样的 json → semantic_text → encode）
✅ 从 Kafka 实时读取 → 向量化 → 使用训练好的 AutoEncoder 推理
✅ 打印预测结果（可选同时落地 parquet）
============================================================
"""

import os
import json
import torch
import torch.nn as nn
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, get_json_object
from pyspark.sql.types import (
    StringType,
    ArrayType,
    FloatType,
    StructType,
    StructField,
    DoubleType,
)

# ==========================================================
# 🧩 Step 0. 路径配置
# ==========================================================
KAFKA_SERVERS = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"

MODEL_DIR = "/opt/spark/models/autoencoder_tfidf_torch"
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "autoencoder.pth")
THRESH_FILE = os.path.join(MODEL_DIR, "threshold.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"🚀 Initializing SentenceTransformer: {MODEL_NAME}")
encoder = SentenceTransformer(MODEL_NAME)

# ==========================================================
# 🧩 Step 1. 加载 AutoEncoder 模型与标准化器
# ==========================================================
scaler = joblib.load(SCALER_FILE)
threshold = float(joblib.load(THRESH_FILE))
input_dim = int(getattr(scaler, "mean_", np.zeros(384)).shape[0]) or 384


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = AutoEncoder(input_dim=input_dim, hidden_dim=128)
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()
print(f"✅ Model loaded (input_dim={input_dim}, threshold={threshold:.6f})")

# ==========================================================
# 🧩 Step 2. Spark 初始化
# ==========================================================
spark = (
    SparkSession.builder.appName("SemanticStreamingInference")
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)
    .config("spark.sql.execution.arrow.pyspark.enabled", True)
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("✅ Spark initialized successfully.")

# ==========================================================
# 🧩 Step 3. 从 Kafka 读取
# ==========================================================
df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVERS)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)
df_raw = df_kafka.selectExpr("CAST(value AS STRING) as message")


# ==========================================================
# 🧩 Step 4. JSON → Semantic Sentence（保持一致）
# ==========================================================
def json_to_semantic(text):
    try:
        data = json.loads(text)
        parts = []
        for k, v in data.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    parts.append(f"{k}.{kk}: {vv}")
            else:
                parts.append(f"{k}: {v}")
        return ". ".join(parts)
    except Exception:
        return "[INVALID_JSON]"


semantic_udf = udf(json_to_semantic, StringType())
df_semantic = df_raw.withColumn("semantic_text", semantic_udf(col("message")))
df_semantic = df_semantic.withColumn(
    "source_type", get_json_object(col("message"), "$.source_type")
)
df_semantic = df_semantic.withColumn(
    "ingest_time", get_json_object(col("message"), "$.timestamp")
)


# ==========================================================
# 🧩 Step 5. 向量化 + AutoEncoder 推理
# ==========================================================
def infer_semantic(text):
    try:
        emb = encoder.encode(text)
        emb_scaled = scaler.transform([emb]).astype(np.float32)
        xt = torch.tensor(emb_scaled)
        with torch.no_grad():
            recon = model(xt)
            mse = torch.mean((xt - recon) ** 2, dim=1).item()
        if mse > threshold * 1.5:
            label = "high_anomaly"
        elif mse > threshold:
            label = "low_anomaly"
        else:
            label = "normal"
        return json.dumps(
            {
                "prediction": label,
                "mse": round(mse, 6),
                "threshold": round(threshold, 6),
            }
        )
    except Exception as e:
        return json.dumps({"prediction": "error", "error": str(e)})


infer_udf = udf(infer_semantic, StringType())

df_pred = df_semantic.withColumn("result", infer_udf(col("semantic_text")))

# ==========================================================
# 🧩 Step 6. 输出结果
# ==========================================================
query_console = (
    df_pred.select("source_type", "ingest_time", "semantic_text", "result")
    .writeStream.outputMode("append")
    .format("console")
    .option("truncate", False)
    .option("numRows", 5)
    .start()
)

# --- 等待任务 ---
print(f"📡 Streaming inference started from Kafka topic: {KAFKA_TOPIC}")
query_console.awaitTermination()
