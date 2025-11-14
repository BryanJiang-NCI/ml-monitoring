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
    get_json_object(col("message"), "$.email").alias("commit_email"),
    get_json_object(col("message"), "$.author").alias("commit_author"),
    get_json_object(col("message"), "$.repository").alias("commit_repository"),
)

## --- GitHub Actions ---
df_github_actions = df_base.filter(col("source_type") == "github_actions").select(
    col("source_type"),
    get_json_object(col("message"), "$.event").alias("action_event"),
    get_json_object(col("message"), "$.name").alias("action_name"),
    get_json_object(col("message"), "$.pipeline_file").alias("action_pipeline_file"),
    get_json_object(col("message"), "$.build_branch").alias("action_build_branch"),
    get_json_object(col("message"), "$.status").alias("action_status"),
    get_json_object(col("message"), "$.conclusion").alias("action_conclusion"),
    get_json_object(col("message"), "$.repository").alias("action_repository"),
)

## --- Public Cloud (CloudTrail / CloudWatch) ---
df_public_cloud = df_base.filter(col("source_type") == "public_cloud").select(
    col("source_type"),
    get_json_object(col("message"), "$.event_name").alias("event_name"),
    get_json_object(col("message"), "$.username").alias("username"),
)

## --- 4. App Container Metrics ---
df_app_metrics = df_base.filter(col("source_type") == "app_container_metrics").select(
    col("source_type"),
    get_json_object(col("message"), "$.device").alias("device"),
    get_json_object(col("message"), "$.kind").alias("kind"),
    get_json_object(col("message"), "$.name").alias("name"),
    get_json_object(col("message"), "$.value").alias("value"),
)

## --- 5. App Container Log ---
df_app_logs = df_base.filter(col("source_type") == "app_container_log").select(
    col("source_type"),
    get_json_object(col("message"), "$.level").alias("log_level"),
    get_json_object(col("message"), "$.logger").alias("logger_name"),
    get_json_object(col("message"), "$.message").alias("log_message"),
    get_json_object(col("message"), "$.service_name").alias("service_name"),
)

df_app_heartbeat = df_base.filter(col("source_type") == "fastapi_status").select(
    col("source_type"),
    get_json_object(col("message"), "$.name").alias("container_name"),
    get_json_object(col("message"), "$.status").alias("container_status"),
    get_json_object(col("message"), "$.status_code").alias("container_status_code"),
    get_json_object(col("message"), "$.url").alias("container_url"),
    get_json_object(col("message"), "$.value").alias("container_value"),
    get_json_object(col("message"), "$.message").alias("container_message"),
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
    .unionByName(df_app_heartbeat, allowMissingColumns=True)
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
