"""
Spark Streaming Inference (Semantic Pipeline + Isolation Forest)
================================================================
✅ 与语义向量落地逻辑完全一致（同样的 json → semantic_text → encode）
✅ 从 Kafka 实时读取 → 向量化 → 使用训练好的 Isolation Forest 推理
✅ 打印预测结果（可选同时落地 parquet）
✅ 异常检测结果写入文件 (/opt/spark/work-dir/data/anomaly_iforest.jsonl)
================================================================
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, get_json_object
from pyspark.sql.types import StringType

# ==========================================================
# 🧩 Step 0. 路径配置
# ==========================================================
KAFKA_SERVERS = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"

BASE_DIR = "/opt/spark/work-dir"
MODEL_DIR = os.path.join(BASE_DIR, "models", "iforest_model")

SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "iforest.pkl")
THRESH_FILE = os.path.join(MODEL_DIR, "threshold.pkl")
ANOMALY_LOG_FILE = os.path.join(BASE_DIR, "data/anomaly.jsonl")
MODEL_NAME = "all-MiniLM-L12-v2"

print(f"🚀 Initializing SentenceTransformer: {MODEL_NAME}")
encoder = SentenceTransformer(MODEL_NAME)

# ==========================================================
# 🧩 Step 1. 加载 Isolation Forest 模型与标准化器
# ==========================================================
scaler = joblib.load(SCALER_FILE)
model = joblib.load(MODEL_FILE)
threshold = float(joblib.load(THRESH_FILE))
print(f"✅ IsolationForest loaded (threshold={threshold:.6f})")

# ==========================================================
# 🧩 Step 2. Spark 初始化
# ==========================================================
spark = (
    SparkSession.builder.appName("SemanticStreamingInference_IForest")
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)
    .config("spark.sql.execution.arrow.pyspark.enabled", True)
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
        if isinstance(data, dict) and "message" in data:
            try:
                msg = json.loads(data["message"])
            except Exception:
                msg = data["message"]
        else:
            msg = data

        parts = []
        if isinstance(msg, dict):
            for k, v in msg.items():
                if any(
                    t in k.lower() for t in ["time", "timestamp", "date", "created_at"]
                ):
                    continue
                parts.append(f"{k} {v}")
        else:
            parts.append(str(msg))
        return " ".join(parts)
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
# 🧩 Step 5. 向量化 + Isolation Forest 推理 + 异常记录
# ==========================================================
def infer_iforest(text):
    try:
        # --- 语义向量化 ---
        emb = encoder.encode(text)
        emb_scaled = scaler.transform([emb])

        # --- Isolation Forest 预测 ---
        score = -model.score_samples(emb_scaled)[0]  # 越高越异常
        label = "anomaly" if score > threshold else "normal"
        ratio = score / threshold

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "semantic_text": text,
            "prediction": label,
            "score": round(score, 6),
            "threshold": round(threshold, 6),
        }

        # --- 异常写入文件 ---
        if label != "normal":
            with open(ANOMALY_LOG_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")
            print(f"⚠️ Anomaly detected → logged to {ANOMALY_LOG_FILE}: {result}")

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"prediction": "error", "error": str(e)})


infer_udf = udf(infer_iforest, StringType())
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

print(f"📡 Streaming inference started (Isolation Forest) from topic: {KAFKA_TOPIC}")
query_console.awaitTermination()
