import os
import time
import requests
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, get_json_object, regexp_extract, lower, when
from pyspark.sql.types import IntegerType

# ============================================
# ⚙️ 参数配置
# ============================================
KAFKA_BOOTSTRAP = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"
TARGET_ROWS = 1000  # 目标收集行数
OUTPUT_PATH = "/opt/spark/work-dir/data/test_labeled_parquet"
CHECKPOINT_PATH = "/opt/spark/work-dir/data/_checkpoints_test_labeled_parquet"

# ============================================
# ⚙️ 异常注入配置 (精简)
# ============================================
MAX_INJECTION_COUNT = 25
API_ENDPOINT = "http://nginx/error"  # 你的实际接口地址
DELAY_BETWEEN_CALLS = 0.1  # 每次调用之间暂停的秒数

# 使用全局变量在 Driver 端进行计数
global_counter = {"total_rows": 0}
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ============================================
# 外部接口调用函数 (精简)
# ============================================
def call_anomaly_injection_api(api_url: str) -> bool:
    """
    调用外部接口，触发一次异常生成（假设接口自行控制生成的日志数量）。
    """
    try:
        # 使用 GET 请求，不携带 payload
        response = requests.get(api_url, timeout=10)
        print(response.text)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        # 忽略详细错误，返回失败状态
        return False


# ============================================
# 🚀 批量异常注入逻辑 (精简)
# ============================================
def run_injection_before_stream():
    """
    一次性执行 50 次接口调用，将异常数据推送到 Kafka。
    """
    print(f"==================================================")
    print(f"🚀 启动批量异常注入 (目标: {MAX_INJECTION_COUNT} 次调用)...")
    print(f"   API: {API_ENDPOINT}")

    start_time = time.time()

    for i in range(MAX_INJECTION_COUNT):
        call_anomaly_injection_api(API_ENDPOINT)

        # 每次调用之间暂停
        time.sleep(DELAY_BETWEEN_CALLS)

    print(f"🎉 批量注入完成，耗时: {time.time() - start_time:.2f}s")
    print(f"==================================================")


# ============================================
# 🚀 初始化 Spark
# ============================================
spark = (
    SparkSession.builder.appName("KafkaToLabeledParquetAutoStop")
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"✅ Spark initialized. Target rows: {TARGET_ROWS}")


# ============================================
# 📌 Step 1-3: 数据处理与自动打标签
# ============================================
df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

df_raw = df_kafka.selectExpr("CAST(value AS STRING) AS json_str")

df = df_raw.select(
    col("json_str"),
    get_json_object(col("json_str"), "$.source_type").alias("source_type"),
    get_json_object(col("json_str"), "$.message").alias("message"),
)

df = df.withColumn("msg_lower", lower(col("message")))

df_labeled = df.withColumn(
    "label",
    when(
        # 1. app log error
        (col("source_type") == "app_container_log")
        & (
            col("msg_lower").contains("error")
            | col("msg_lower").contains("unreachable")
            | col("msg_lower").contains("ERROR")
        ),
        1,
    )
    .when(
        # 2. nginx 5xx
        (col("source_type") == "nginx_access")
        & (col("msg_lower").contains('"status":500')),
        1,
    )
    .otherwise(0),  # 正常 -> 0
).drop("msg_lower")


# ============================================
# 💾 Step 4: ForeachBatch 计数与输出
# ============================================
def write_and_count(batch_df, batch_id):
    """
    处理每个微批次：写入 Parquet，并更新全局计数器。
    """
    global global_counter

    # 写入 Parquet 文件
    batch_df.select(
        col("source_type"),
        col("message"),
        col("json_str"),
        col("label").cast(IntegerType()),
    ).write.mode("append").format("parquet").partitionBy("source_type").save(
        OUTPUT_PATH
    )

    # 更新全局计数器
    count = batch_df.count()
    global_counter["total_rows"] += count

    # 打印进度
    print(
        f"| Batch {batch_id}: Processed {count} rows. Total: {global_counter['total_rows']}/{TARGET_ROWS} |"
    )

    # 检查是否达到目标
    if global_counter["total_rows"] >= TARGET_ROWS:
        # 抛出异常，触发主线程终止
        raise Exception("Target row count reached, initiating shutdown.")


# ============================================
# ⚙️ 启动主程序
# ============================================
if __name__ == "__main__":

    # 1. 启动流式查询（先启动！）
    query = (
        df_labeled.writeStream.outputMode("append")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .foreachBatch(write_and_count)
        .start()
    )

    print(f"📡 Kafka stream started → {OUTPUT_PATH}. Waiting for {TARGET_ROWS} rows...")

    # ⭐ 给 Spark 几秒钟时间建立 Kafka 连接
    time.sleep(3)

    # 2. 启动批量异常注入（此时 Kafka → Spark 流已经在工作）
    run_injection_before_stream()

    try:
        # 阻塞主线程，直到达到目标条数（在 foreachBatch 中触发）
        query.awaitTermination(timeout=36000)

    except Exception as e:
        if "Target row count reached" in str(e):
            print(f"\n🛑 Target row count ({TARGET_ROWS}) reached. Stopping query...")
            query.stop()
        else:
            print(f"\n⚠️ Unexpected error during streaming: {e}. Stopping query...")
            query.stop()

    spark.stop()
    print("✅ Spark session terminated.")
