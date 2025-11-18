import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, get_json_object, regexp_extract, lower, when, lit

# ============================================
# ⚙️ 参数配置
# ============================================
KAFKA_BOOTSTRAP = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"

OUTPUT_PATH = "/opt/spark/work-dir/data/test_labeled_parquet"
CHECKPOINT_PATH = "/opt/spark/work-dir/data/_checkpoints_test_labeled_parquet"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================
# 🚀 初始化 Spark
# ============================================
spark = (
    SparkSession.builder.appName("KafkaToLabeledParquet")
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("✅ Spark initialized: Kafka → Labeled Parquet")

# ============================================
# 📥 Kafka 流
# ============================================
df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

df_raw = df_kafka.selectExpr("CAST(value AS STRING) AS json_str")

# ============================================
# 📌 Step 1: 解析 JSON 基础字段
# ============================================
df = df_raw.select(
    col("json_str"),
    get_json_object(col("json_str"), "$.source_type").alias("source_type"),
    get_json_object(col("json_str"), "$.message").alias("message"),
)

# ============================================
# 🔧 Step 2: 提取用于标签判断的字段（仅在对应 source_type 下解析）
# ============================================

# msg_lower（所有日志都可以安全解析）
df = df.withColumn("msg_lower", lower(col("message")))

# -------- Nginx status only when source_type == nginx_access --------
df = df.withColumn(
    "nginx_status_raw",
    when(
        col("source_type") == "nginx_access",
        regexp_extract(col("message"), r"status[=: ](\d+)", 1),
    ).otherwise(""),
)

df = df.withColumn(
    "nginx_status",
    when(col("nginx_status_raw") != "", col("nginx_status_raw").cast("int")).otherwise(
        None
    ),
)


# ============================================
# 📌 Step 3: 自动规则打标签 label
# ============================================
df = df.withColumn(
    "label",
    when(
        # 1. app log error
        (col("source_type") == "app_container_log")
        & (
            col("msg_lower").contains("error")
            | col("msg_lower").contains("unreachable")
            | col("msg_lower").contains("Error")
        ),
        1,
    )
    .when(
        # 2. nginx 5xx
        (col("source_type") == "nginx_access") & (col("nginx_status") >= 500),
        1,
    )
    .otherwise(0),
)

# ============================================
# 💾 Step 4: 写入 Parquet
# ============================================
query = (
    df.writeStream.outputMode("append")
    .format("parquet")
    .option("path", OUTPUT_PATH)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .partitionBy("source_type")
    .trigger(processingTime="20 seconds")
    .start()
)

print(f"📡 Kafka stream started → {OUTPUT_PATH}")
query.awaitTermination()
