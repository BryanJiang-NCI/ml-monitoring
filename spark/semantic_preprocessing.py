"""
semantic_preprocessing.py
Semantic Vector Streaming Pipeline with Spark Streaming
"""

import os
import json
import time
import threading
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, get_json_object
from pyspark.sql.types import StringType, ArrayType, FloatType
from sentence_transformers import SentenceTransformer

# base settings
KAFKA_SERVERS = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"
OUTPUT_PATH = "/opt/spark/work-dir/data/semantic_data"
CHECKPOINT_PATH = "/opt/spark/work-dir/data/_checkpoints_semantic_data"
MODEL_NAME = "all-MiniLM-L12-v2"
TARGET_ROWS = 5000

print(f"🚀 Initializing SentenceTransformer: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# spark session
spark = (
    SparkSession.builder.appName("SemanticVectorStreaming")
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)
    .config("spark.sql.execution.arrow.pyspark.enabled", True)
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("✅ Spark initialized successfully.")

# kafaka read stream
df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVERS)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

df_raw = df_kafka.selectExpr("CAST(value AS STRING) as message")


# json to semantic text
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
                parts.append(f"{k}={v}")
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


# text to embedding
def encode_text(text):
    try:
        embedding = model.encode(text)
        return [float(x) for x in embedding]
    except Exception:
        return []


encode_udf = udf(encode_text, ArrayType(FloatType()))
df_vec = df_semantic.withColumn("embedding", encode_udf(col("semantic_text")))


global_counter = {"total_rows": 0}


def write_and_count(batch_df, batch_id):
    """Writes batch DataFrame to Parquet and updates global row count"""
    global global_counter

    batch_df.select(
        "source_type", "ingest_time", "semantic_text", "embedding"
    ).write.mode("append").format("parquet").save(OUTPUT_PATH)

    count = batch_df.count()
    global_counter["total_rows"] += count

    print(
        f"| Batch {batch_id}: Processed {count} rows. Total: {global_counter['total_rows']}/{TARGET_ROWS} |"
    )


query_parquet = (
    df_vec.writeStream.outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .foreachBatch(write_and_count)
    .start()
)


print(f"\n⏰ Streaming query started. Monitoring total rows. Target: {TARGET_ROWS}...")

try:
    while global_counter["total_rows"] < TARGET_ROWS and query_parquet.isActive:
        time.sleep(10)

    if query_parquet.isActive:
        print(f"\n🛑 Target row count ({TARGET_ROWS}) reached. Stopping query...")
        query_parquet.stop()

except Exception as e:
    print(f"\n⚠️ Error occurred or interruption: {e}. Stopping query...")
    if query_parquet.isActive:
        query_parquet.stop()

spark.stop()
print("✅ Spark session terminated.")
