import time
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    get_json_object,
    regexp_extract,
    from_json,
    regexp_replace,
)
from pyspark.sql.types import StringType

# ==========================================================
# 🧩 Step 0. 配置与初始化
# ==========================================================
TARGET_ROWS = 5000  # <--- 目标收集行数
KAFKA_SERVERS = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"
PARQUET_PATH = "/opt/spark/work-dir/data/structured_data"
CHECKPOINT_PATH = "/opt/spark/work-dir/data/_checkpoints_structured_data"
# 使用全局变量在 Driver 端进行计数
global_counter = {"total_rows": 0}

spark = SparkSession.builder.appName("LogPreprocessingMultiSource").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print(f"✅ Spark initialized. Target rows: {TARGET_ROWS}")


# ==========================================================
# 🧩 Step 1-3. 原始数据处理与合并 (逻辑不变，仅移除末尾的 await)
# ==========================================================
df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVERS)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)
df = df.selectExpr("CAST(value AS STRING) as json_str")

# 通用字段提取
df_base = df.select(
    get_json_object(col("json_str"), "$.source_type").alias("source_type"),
    get_json_object(col("json_str"), "$.timestamp").alias("timestamp"),
    get_json_object(col("json_str"), "$.message").alias("message"),
)

# 各类日志分流解析
df_github_commits = df_base.filter(col("source_type") == "github_commits").select(
    col("source_type"),
    get_json_object(col("message"), "$.email").alias("commit_email"),
    get_json_object(col("message"), "$.author").alias("commit_author"),
    get_json_object(col("message"), "$.repository").alias("commit_repository"),
)

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

df_public_cloud = df_base.filter(col("source_type") == "public_cloud").select(
    col("source_type"),
    get_json_object(col("message"), "$.event_name").alias("event_name"),
    get_json_object(col("message"), "$.username").alias("username"),
)

df_app_metrics = df_base.filter(col("source_type") == "app_container_metrics").select(
    col("source_type"),
    get_json_object(col("message"), "$.device").alias("device"),
    get_json_object(col("message"), "$.kind").alias("kind"),
    get_json_object(col("message"), "$.name").alias("name"),
    get_json_object(col("message"), "$.value").alias("value"),
)

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

df_nginx_error = df_base.filter(col("source_type") == "nginx_error").select(
    col("source_type"),
    regexp_extract("message", r"\[(\w+)\]", 1).alias("error_level"),
    regexp_extract("message", r": (.*)", 1).alias("error_detail"),
)

df_final = (
    df_github_commits.unionByName(df_github_actions, allowMissingColumns=True)
    .unionByName(df_public_cloud, allowMissingColumns=True)
    .unionByName(df_app_metrics, allowMissingColumns=True)
    .unionByName(df_app_logs, allowMissingColumns=True)
    .unionByName(df_nginx, allowMissingColumns=True)
    .unionByName(df_nginx_error, allowMissingColumns=True)
    .unionByName(df_app_heartbeat, allowMissingColumns=True)
)


# ==========================================================
# 🧩 Step 4. ForeachBatch 计数与输出 (新逻辑)
# ==========================================================


def write_and_count(batch_df, batch_id):
    """
    处理每个微批次：写入 Parquet，并更新全局计数器。
    """
    global global_counter

    # 1. 写入 Parquet 文件 (文件输出)
    batch_df.write.mode("append").format("parquet").partitionBy("source_type").save(
        PARQUET_PATH
    )

    # 2. 控制台输出 (模拟原始 console 流程)
    # print(f"\n--- Batch {batch_id} ---")
    # batch_df.show(5, truncate=False)  # 只显示前 5 行

    # 3. 更新全局计数器
    count = batch_df.count()
    global_counter["total_rows"] += count

    # 4. 打印进度
    print(
        f"| Processed {count} rows. Total: {global_counter['total_rows']}/{TARGET_ROWS} |"
    )

    # 5. 检查是否达到目标
    if global_counter["total_rows"] >= TARGET_ROWS:
        # 抛出异常，让主循环捕获并停止
        raise Exception("Target row count reached, initiating shutdown.")


# --- 启动流式查询 ---
query_stream = (
    df_final.writeStream.outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .foreachBatch(write_and_count)
    .start()
)

# --- 主线程监控与终止 ---
print(f"\n⏰ Streaming query started. Monitoring total rows. Target: {TARGET_ROWS}...")

try:
    # 使用 awaitAnyTermination(timeout) 避免无限阻塞，并在内部通过 Exception 触发终止
    # 因为 foreachBatch 中的 Exception 会传递到 Driver 线程
    # 如果超过 10 小时还没跑完，也自动退出
    query_stream.awaitTermination(timeout=36000)

except Exception as e:
    # 捕获我们在 foreachBatch 中抛出的 "Target row count reached" 异常
    if "Target row count reached" in str(e):
        print(f"\n🛑 Target row count ({TARGET_ROWS}) reached. Stopping query...")
        query_stream.stop()
    else:
        print(f"\n⚠️ Unexpected error during streaming: {e}. Stopping query...")
        query_stream.stop()

spark.stop()
print("✅ Spark session terminated.")
