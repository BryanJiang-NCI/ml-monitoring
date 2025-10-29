"""
Semantic Vector Streaming Pipeline (Simplified Version)
========================================================
✅ 仅提取 Kafka 消息中的 message 字段
✅ 去除所有 Vector 元信息（file、host、timestamp 等）
✅ 过滤时间类字段 (time/timestamp/date)
✅ SentenceTransformer 向量化 → Parquet + 控制台输出
========================================================
"""

import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, get_json_object
from pyspark.sql.types import StringType, ArrayType, FloatType
from sentence_transformers import SentenceTransformer

# ==========================================================
# 🧩 Step 0. 配置
# ==========================================================
KAFKA_SERVERS = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"
OUTPUT_PATH = "/opt/spark/work-dir/data/semantic_vectors"
CHECKPOINT_PATH = "/opt/spark/work-dir/data/_checkpoints_semantic_vectors"
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"🚀 Initializing SentenceTransformer: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# ==========================================================
# 🧩 Step 1. 初始化 Spark
# ==========================================================
spark = (
    SparkSession.builder.appName("SemanticVectorStreaming")
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)
    .config("spark.sql.execution.arrow.pyspark.enabled", True)
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("✅ Spark initialized successfully.")

# ==========================================================
# 🧩 Step 2. Kafka Source
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
# 🧩 Step 3. JSON → Semantic Text
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
                parts.append(f"{k}: {v}")
        else:
            parts.append(str(msg))

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
# 🧩 Step 4. SentenceTransformer 向量化
# ==========================================================
def encode_text(text):
    try:
        embedding = model.encode(text)
        return [float(x) for x in embedding]
    except Exception:
        return []


encode_udf = udf(encode_text, ArrayType(FloatType()))
df_vec = df_semantic.withColumn("embedding", encode_udf(col("semantic_text")))

# ==========================================================
# 🧩 Step 5. 输出结果
# ==========================================================
# --- 控制台输出（调试用） ---
query_console = (
    df_vec.select("source_type", "semantic_text")
    .writeStream.outputMode("append")
    .format("console")
    .option("truncate", False)
    .option("numRows", 5)
    .start()
)

# --- 写入 Parquet ---
query_parquet = (
    df_vec.select("source_type", "ingest_time", "semantic_text", "embedding")
    .writeStream.outputMode("append")
    .format("parquet")
    .option("path", OUTPUT_PATH)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(processingTime="60 seconds")
    .start()
)

spark.streams.awaitAnyTermination()
