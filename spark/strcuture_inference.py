"""
Spark Structured Streaming Realtime Inference (Full Safe Version - Fixed)
==================================================
✅ 对齐训练集字段定义（含 access_method / access_uri）
✅ Worker 懒加载 IsolationForest 模型（joblib）
✅ 无自定义类，兼容 Spark / Python / Kafka
✅ 彻底修复 SparkContext 序列化错误（无 DataFrame 引用）
==================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    get_json_object,
    from_json,
    pandas_udf,
    coalesce,
    regexp_replace,
    explode_outer,
)
from pyspark.sql.types import *
import pandas as pd

# ==========================================================
# 🧩 配置部分
# ==========================================================
MODEL_PATH = "/opt/spark/work-dir/models/isolation_forest.pkl"
CHECKPOINT_PATH = "/opt/spark/work-dir/data/_checkpoints_infer"
TRIGGER_INTERVAL = "30 seconds"
KAFKA_BROKER = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"

# ==========================================================
# 🧩 初始化 Spark
# ==========================================================
spark = (
    SparkSession.builder.appName("RealtimeInferenceFullSafe")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("✅ Spark initialized...")

# ==========================================================
# 🧩 从 Kafka 读取数据流
# ==========================================================
df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BROKER)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

df_raw = df_kafka.selectExpr("CAST(value AS STRING) as json_str")

base_df = df_raw.select(
    get_json_object(col("json_str"), "$.source_type").alias("source_type"),
    get_json_object(col("json_str"), "$.timestamp").alias("ingest_time"),
    get_json_object(col("json_str"), "$.message").alias("message"),
)

# ==========================================================
# 🧩 定义 schema
# ==========================================================
schema_nginx = StructType(
    [
        StructField("remote_addr", StringType()),
        StructField("request_method", StringType()),
        StructField("request_uri", StringType()),
        StructField("status", StringType()),
        StructField("body_bytes_sent", StringType()),
        StructField("request_time", StringType()),
        StructField("http_referer", StringType()),
        StructField("http_user_agent", StringType()),
    ]
)

schema_syslog = StructType(
    [
        StructField("SYSLOG_IDENTIFIER", StringType()),
        StructField("PRIORITY", StringType()),
        StructField("SYSLOG_FACILITY", StringType()),
        StructField("host", StringType()),
        StructField("message", StringType()),
    ]
)

schema_host_metrics = StructType(
    [
        StructField("name", StringType()),
        StructField("host", StringType()),
        StructField("source_type", StringType()),
        StructField("counter", StructType([StructField("value", DoubleType())]), True),
        StructField("gauge", StructType([StructField("value", DoubleType())]), True),
    ]
)

schema_github_commit = ArrayType(
    StructType(
        [
            StructField("sha", StringType()),
            StructField(
                "commit",
                StructType(
                    [
                        StructField(
                            "author",
                            StructType(
                                [
                                    StructField("name", StringType()),
                                    StructField("email", StringType()),
                                    StructField("date", StringType()),
                                ]
                            ),
                        ),
                        StructField("message", StringType()),
                    ]
                ),
            ),
            StructField("author", StructType([StructField("login", StringType())])),
            StructField("html_url", StringType()),
        ]
    )
)

schema_cmdb = StructType(
    [
        StructField("service_name", StringType()),
        StructField("domain", StringType()),
        StructField("owner", StringType()),
        StructField("language", StringType()),
        StructField("deployment_env", StringType()),
    ]
)

# ==========================================================
# 🧩 各类日志解析
# ==========================================================
df_nginx = (
    base_df.filter(col("source_type") == "nginx_access")
    .withColumn("json_data", from_json(col("message"), schema_nginx))
    .select(
        col("source_type"),
        col("ingest_time"),
        col("json_data.remote_addr").alias("access_client"),
        col("json_data.request_method").alias("access_method"),
        col("json_data.request_uri").alias("access_uri"),
        col("json_data.status").cast("int").alias("access_status"),
        col("json_data.body_bytes_sent").cast("int").alias("response_size"),
        col("json_data.request_time").cast("double").alias("request_time"),
        col("json_data.http_referer").alias("http_referer"),
        col("json_data.http_user_agent").alias("user_agent"),
    )
)

df_syslog = (
    base_df.filter(col("source_type") == "syslog")
    .withColumn("json_data", from_json(col("message"), schema_syslog))
    .select(
        col("source_type"),
        col("ingest_time"),
        col("json_data.host"),
        col("json_data.message").alias("syslog_message"),
        col("json_data.SYSLOG_IDENTIFIER").alias("syslog_identifier"),
        col("json_data.PRIORITY").cast("int").alias("syslog_priority"),
        col("json_data.SYSLOG_FACILITY").cast("int").alias("syslog_facility"),
    )
)

df_host = (
    base_df.filter(col("source_type") == "host_metrics")
    .withColumn("json_data", from_json(col("message"), schema_host_metrics))
    .select(
        col("source_type"),
        col("ingest_time"),
        col("json_data.name").alias("metric_name"),
        col("json_data.host").alias("host"),
        coalesce(col("json_data.counter.value"), col("json_data.gauge.value")).alias(
            "metric_value"
        ),
    )
)

df_commit = (
    base_df.filter(col("source_type") == "github_commits")
    .withColumn("msg_clean", regexp_replace(col("message"), r"\\\\", ""))
    .withColumn("data", from_json(col("msg_clean"), schema_github_commit))
    .withColumn("commit", explode_outer("data"))
    .select(
        col("source_type"),
        col("ingest_time"),
        col("commit.commit.message").alias("commit_message"),
        col("commit.author.login").alias("github_user"),
        col("commit.html_url").alias("commit_url"),
    )
)

df_cmdb = (
    base_df.filter(col("source_type") == "cmdb")
    .withColumn("json_data", from_json(col("message"), schema_cmdb))
    .select(
        col("source_type"),
        col("ingest_time"),
        col("json_data.service_name").alias("cmdb_service"),
        col("json_data.domain").alias("cmdb_domain"),
        col("json_data.owner").alias("cmdb_owner"),
        col("json_data.language").alias("cmdb_language"),
        col("json_data.deployment_env").alias("cmdb_deployment_env"),
    )
)

# ==========================================================
# 🧩 合并所有流
# ==========================================================
union_df = (
    df_nginx.unionByName(df_syslog, True)
    .unionByName(df_host, True)
    .unionByName(df_commit, True)
    .unionByName(df_cmdb, True)
)

# ==========================================================
# 🧩 预测输入列（必须与训练完全对齐）
# ==========================================================
predict_columns = [
    "source_type",
    "ingest_time",
    "access_client",
    "access_method",
    "access_uri",
    "access_status",
    "response_size",
    "request_time",
    "http_referer",
    "user_agent",
    "error_message",
    "error_client",
    "error_server",
    "error_request",
    "error_host",
    "syslog_message",
    "syslog_identifier",
    "syslog_priority",
    "syslog_facility",
    "host",
    "commit_message",
    "github_user",
    "commit_url",
    "cmdb_service",
    "cmdb_domain",
    "cmdb_owner",
    "cmdb_language",
    "cmdb_deployment_env",
    "metric_name",
    "metric_value",
]

# ✅ 提前计算可用列（防止在 UDF 内部引用 Spark 对象）
active_columns = [c for c in predict_columns if c in union_df.columns]


# ==========================================================
# 🧩 Pandas UDF 推理函数（Worker 懒加载模型）
# ==========================================================
# === 用连续分数 + 0阈值；并把分数打出来便于观测 ===
@pandas_udf("struct<prediction:string, score:double, debug_reason:string>")
def predict_udf(*cols: pd.Series) -> pd.DataFrame:
    import joblib
    import numpy as np

    global _model, _model_cols

    if "_model" not in globals():
        print(f"[Worker] 🔄 Loading model from {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        _model_cols = list(_model.named_steps["preprocessor"].feature_names_in_)
        print(f"[Worker] ✅ Model loaded with {_model_cols} columns")

    pdf = pd.concat(cols, axis=1)
    pdf.columns = active_columns
    pdf = pdf.fillna("")

    # 数值列强转
    numeric_cols = [
        "access_status",
        "response_size",
        "request_time",
        "syslog_priority",
        "syslog_facility",
        "metric_value",
    ]
    for c in numeric_cols:
        if c in pdf.columns:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0)

    # 对齐模型输入列
    for c in _model_cols:
        if c not in pdf.columns:
            pdf[c] = ""
    pdf = pdf[_model_cols]

    try:
        X = _model.named_steps["preprocessor"].transform(pdf)
        scores = _model.named_steps["isoforest"].score_samples(X)  # 连续分数
        preds = np.where(scores < -0.6, "anomaly", "normal")  # 0 为经验阈值
        return pd.DataFrame(
            {"prediction": preds, "score": scores, "debug_reason": [""] * len(preds)}
        )
    except Exception as e:
        reason = str(e)[:200]
        return pd.DataFrame(
            {
                "prediction": ["error"] * len(pdf),
                "score": [float("nan")] * len(pdf),
                "debug_reason": [reason] * len(pdf),
            }
        )


# ==========================================================
# 🧩 输出流定义
# ==========================================================
predicted_df = union_df.withColumn(
    "pred",
    predict_udf(*[col(c) for c in active_columns]),
).select(
    col("ingest_time"),
    col("source_type"),
    col("pred.prediction").alias("prediction"),
    col("pred.score").alias("anomaly_score"),
    col("pred.debug_reason").alias("debug_reason"),
)

# ==========================================================
# 🧩 启动流式任务
# ==========================================================
query = (
    predicted_df.writeStream.outputMode("append")
    .format("console")
    .option("truncate", False)
    .option("numRows", 20)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(processingTime=TRIGGER_INTERVAL)
    .start()
)

print("🚀 Realtime inference pipeline running...")
query.awaitTermination()
