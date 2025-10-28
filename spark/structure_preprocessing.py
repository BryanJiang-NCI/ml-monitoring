from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    get_json_object,
    regexp_extract,
    from_json,
    explode_outer,
    regexp_replace,
)
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType

# ========== Spark Init ==========
spark = SparkSession.builder.appName("LogPreprocessingMultiSource").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ========== Step 0: Kafka Source ==========
df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "kafka-kraft:9092")
    .option("subscribe", "monitoring-data")
    .option("startingOffsets", "latest")
    .load()
)
df = df.selectExpr("CAST(value AS STRING) as json_str")

# ========== Step 1: 通用字段提取 ==========
df_base = df.select(
    get_json_object(col("json_str"), "$.source_type").alias("source_type"),
    get_json_object(col("json_str"), "$.timestamp").alias("timestamp"),
    get_json_object(col("json_str"), "$.message").alias("message"),
)

# ========== Step 2: 各类日志分流解析 ==========

## --- GitHub Commits ---
df_github_commits = df_base.filter(col("source_type") == "github_commits").select(
    col("source_type"),
    get_json_object(col("message"), "$.sha").alias("commit_sha"),
    get_json_object(col("message"), "$.author").alias("commit_author"),
    get_json_object(col("message"), "$.date").alias("commit_date"),
    get_json_object(col("message"), "$.message").alias("commit_message"),
    get_json_object(col("message"), "$.url").alias("commit_url"),
)

## --- GitHub Actions ---
df_github_actions = df_base.filter(col("source_type") == "github_actions").select(
    col("source_type"),
    get_json_object(col("message"), "$.id").alias("action_id"),
    get_json_object(col("message"), "$.name").alias("action_name"),
    get_json_object(col("message"), "$.status").alias("action_status"),
    get_json_object(col("message"), "$.conclusion").alias("action_conclusion"),
    get_json_object(col("message"), "$.created_at").alias("action_created_at"),
    get_json_object(col("message"), "$.url").alias("action_url"),
)

## --- Public Cloud (CloudTrail / CloudWatch) ---
df_public_cloud = df_base.filter(col("source_type") == "public_cloud").select(
    col("source_type"),
    get_json_object(col("message"), "$.event_id").alias("event_id"),
    get_json_object(col("message"), "$.event_name").alias("event_name"),
    get_json_object(col("message"), "$.username").alias("username"),
    get_json_object(col("message"), "$.event_time").alias("event_time"),
)

## --- 4. App Container Metrics ---
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

## --- 5. App Container Log ---
df_app_logs = df_base.filter(col("source_type") == "app_container_log").select(
    col("source_type"),
    get_json_object(col("message"), "$.time").alias("log_time"),
    get_json_object(col("message"), "$.level").alias("log_level"),
    get_json_object(col("message"), "$.logger").alias("logger_name"),
    get_json_object(col("message"), "$.message").alias("log_message"),
)

# --- Nginx ---
df_nginx = df_base.filter(col("source_type") == "nginx_access").select(
    col("source_type"),
    get_json_object(col("message"), "$.remote_addr").alias("client_ip"),
    get_json_object(col("message"), "$.request_method").alias("request_method"),
    get_json_object(col("message"), "$.request_uri").alias("request_uri"),
    get_json_object(col("message"), "$.status").alias("response_status"),
    get_json_object(col("message"), "$.body_bytes_sent").alias("body_bytes_sent"),
    get_json_object(col("message"), "$.request_time").alias("request_time"),
    get_json_object(col("message"), "$.http_user_agent").alias("user_agent"),
)

## --- Nginx Error Logs (Simplified) ---
df_nginx_error = df_base.filter(col("source_type") == "nginx_error").select(
    col("source_type"),
    regexp_extract("message", r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})", 1).alias(
        "error_time"
    ),
    regexp_extract("message", r"\[(\w+)\]", 1).alias("error_level"),
    regexp_extract("message", r": (.*)", 1).alias("error_detail"),
)


# ========== Step 3: 合并为统一结构 ==========
df_final = (
    df_github_commits.unionByName(df_github_actions, allowMissingColumns=True)
    .unionByName(df_public_cloud, allowMissingColumns=True)
    .unionByName(df_app_metrics, allowMissingColumns=True)
    .unionByName(df_app_logs, allowMissingColumns=True)
    .unionByName(df_nginx, allowMissingColumns=True)
    .unionByName(df_nginx_error, allowMissingColumns=True)
)

# ========== Step 4: 输出到控制台 + 文件 ==========
query_console = (
    df_final.writeStream.outputMode("append")
    .format("console")
    .option("truncate", "false")
    .start()
)

query_file = (
    df_final.writeStream.outputMode("append")
    .format("parquet")
    .option(
        "checkpointLocation", "/opt/spark/work-dir/data/_checkpoints_structured_data"
    )
    .option("path", "/opt/spark/work-dir/data/structured_data")
    .partitionBy("source_type")
    .start()
)

spark.streams.awaitAnyTermination()
