"""
generate_testset.py
Generate a test dataset with labeled anomalies from Kafka stream
"""

import os
import time
import requests
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, get_json_object, regexp_extract, lower, when
from pyspark.sql.types import IntegerType


KAFKA_BOOTSTRAP = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"
TARGET_ROWS = 1000
OUTPUT_PATH = "/opt/spark/work-dir/data/test_labeled_parquet"
CHECKPOINT_PATH = "/opt/spark/work-dir/data/_checkpoints_test_labeled_parquet"

# injection settings
MAX_INJECTION_COUNT = 25
API_ENDPOINT = "http://nginx/error"
DELAY_BETWEEN_CALLS = 0.1

global_counter = {"total_rows": 0}
os.makedirs(OUTPUT_PATH, exist_ok=True)


def call_anomaly_injection_api(api_url: str) -> bool:
    """
    request simulate api to inject anomaly data into kafka
    """
    try:
        response = requests.get(api_url, timeout=10)
        print(response.text)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def run_injection_before_stream():
    """
    Run batch anomaly injection before starting the main stream processing.
    """
    print(f"==================================================")
    print(f"🚀 Started micro batch fault injection: {MAX_INJECTION_COUNT} calls")
    print(f"   API: {API_ENDPOINT}")

    start_time = time.time()

    for i in range(MAX_INJECTION_COUNT):
        call_anomaly_injection_api(API_ENDPOINT)

        time.sleep(DELAY_BETWEEN_CALLS)

    print(f"🎉 batch injection finished: {time.time() - start_time:.2f}s")
    print(f"==================================================")


spark = (
    SparkSession.builder.appName("KafkaToLabeledParquetAutoStop")
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"✅ Spark initialized. Target rows: {TARGET_ROWS}")


# data handling and labeling
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
        (col("source_type") == "app_container_log")
        & (
            col("msg_lower").contains("error")
            | col("msg_lower").contains("unreachable")
            | col("msg_lower").contains("ERROR")
        ),
        1,
    )
    .when(
        (col("source_type") == "nginx_access")
        & (col("msg_lower").contains('"status":500')),
        1,
    )
    .otherwise(0),
).drop("msg_lower")


def write_and_count(batch_df, batch_id):
    """
    Write each micro-batch to Parquet and count total rows.
    """
    global global_counter

    # save to parquet with partitioning
    batch_df.select(
        col("source_type"),
        col("message"),
        col("json_str"),
        col("label").cast(IntegerType()),
    ).write.mode("append").format("parquet").partitionBy("source_type").save(
        OUTPUT_PATH
    )

    count = batch_df.count()
    global_counter["total_rows"] += count

    print(
        f"| Batch {batch_id}: Processed {count} rows. Total: {global_counter['total_rows']}/{TARGET_ROWS} |"
    )

    if global_counter["total_rows"] >= TARGET_ROWS:
        raise Exception("Target row count reached, initiating shutdown.")


if __name__ == "__main__":
    # start streaming query with foreachBatch
    query = (
        df_labeled.writeStream.outputMode("append")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .foreachBatch(write_and_count)
        .start()
    )

    print(f"📡 Kafka stream started → {OUTPUT_PATH}. Waiting for {TARGET_ROWS} rows...")

    time.sleep(3)

    run_injection_before_stream()

    try:
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
