# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, from_json, pandas_udf
# from pyspark.sql.types import (
#     StructType,
#     StructField,
#     StringType,
#     DoubleType,
#     TimestampType,
# )
# import pandas as pd
# import joblib
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin

# # ========== 路径 & 配置 ==========
# KAFKA_BROKER = "kafka-kraft:9092"
# KAFKA_TOPIC_IN = "monitoring-data"
# MODEL_PATH = "/opt/spark-apps/models/pyod_autoencoder.pkl"
# PREPROCESSOR_PATH = "/opt/spark-apps/models/pyod_preprocessor.pkl"
# CHECKPOINT_PATH = "/opt/spark-data/_checkpoints_infer_auto_single"
# TRIGGER_INTERVAL = "30 seconds"

# # ========== 初始化 Spark ==========
# spark = (
#     SparkSession.builder.appName("RealtimeAutoEncoderInferenceSingle")
#     .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
#     .getOrCreate()
# )
# spark.sparkContext.setLogLevel("ERROR")


# # ========== 修复 TextSelector 反序列化问题 ==========
# class TextSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, key):
#         self.key = key

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X[self.key].astype(str).fillna("")


# # ========== 加载模型 ==========
# print("📦 Loading trained AutoEncoder model and preprocessor ...")
# model = joblib.load(MODEL_PATH)
# preprocessor = joblib.load(PREPROCESSOR_PATH)
# print("✅ Model & Preprocessor loaded successfully!")

# # ========== Step 1. 定义 Kafka 输入结构 ==========
# schema = StructType(
#     [
#         StructField("source_type", StringType()),
#         StructField("timestamp", StringType()),
#         StructField("client_ip", StringType()),
#         StructField("user", StringType()),
#         StructField("http_method", StringType()),
#         StructField("url", StringType()),
#         StructField("status_code", DoubleType()),
#         StructField("response_size", DoubleType()),
#         StructField("syslog_text", StringType()),
#         StructField("gh_status", StringType()),
#         StructField("commit_message", StringType()),
#         StructField("cmdb_service_name", StringType()),
#         StructField("cmdb_language", StringType()),
#     ]
# )

# df_kafka = (
#     spark.readStream.format("kafka")
#     .option("kafka.bootstrap.servers", KAFKA_BROKER)
#     .option("subscribe", KAFKA_TOPIC_IN)
#     .option("startingOffsets", "latest")
#     .load()
# )

# df_parsed = (
#     df_kafka.selectExpr("CAST(value AS STRING) as json_str")
#     .withColumn("data", from_json(col("json_str"), schema))
#     .select("data.*")
#     .withColumn("timestamp", col("timestamp").cast(TimestampType()))
# )


# # ========== Step 2. 定义 Pandas UDF 进行实时推理 ==========
# @pandas_udf("string")  # 输出每条记录的预测结果 ("normal" / "anomaly")
# def predict_single_batch(pdf: pd.DataFrame) -> pd.Series:
#     if pdf.empty:
#         return pd.Series([])

#     # 缺失值填充
#     pdf = pdf.fillna(
#         {
#             "client_ip": "0.0.0.0",
#             "user": "unknown",
#             "http_method": "GET",
#             "url": "",
#             "syslog_text": "",
#             "commit_message": "",
#             "gh_status": "",
#             "cmdb_service_name": "",
#             "cmdb_language": "",
#             "status_code": 0,
#             "response_size": 0,
#         }
#     )

#     # 确保字段顺序与训练一致
#     df_infer = pdf[
#         [
#             "source_type",
#             "client_ip",
#             "user",
#             "http_method",
#             "url",
#             "status_code",
#             "response_size",
#             "syslog_text",
#             "gh_status",
#             "commit_message",
#             "cmdb_service_name",
#             "cmdb_language",
#         ]
#     ]

#     # === 预处理 ===
#     X = preprocessor.transform(df_infer)
#     if hasattr(X, "toarray"):
#         X = X.toarray()

#     # === 计算重构误差 ===
#     scores = model.decision_function(X)
#     threshold = model.threshold_ * 1.5  # 适度放宽阈值
#     preds = (scores > threshold).astype(int)
#     pred_labels = ["anomaly" if s > threshold else "normal" for s in scores]

#     # 可选：调试信息（仅Driver端）
#     print(f"[DEBUG] mean_score={np.mean(scores):.3f}, threshold={threshold:.3f}")

#     return pd.Series(pred_labels)


# # ========== Step 3. 增加推理列 ==========
# # df_pred = df_parsed.withColumn(
# #     "prediction", predict_single_batch(*[col(c) for c in df_parsed.columns])
# # )
# from pyspark.sql.functions import struct

# df_pred = df_parsed.withColumn(
#     "prediction", predict_single_batch(struct(*[col(c) for c in df_parsed.columns]))
# )

# # ========== Step 4. 输出结果 ==========
# query = (
#     df_pred.writeStream.outputMode("append")
#     .format("console")
#     .option("truncate", False)
#     .option("checkpointLocation", CHECKPOINT_PATH)
#     .trigger(processingTime=TRIGGER_INTERVAL)
#     .start()
# )

# print("🚀 Real-time AutoEncoder per-log inference started...")
# query.awaitTermination()


"""
spark_stream_inference_autoencoder.py
---------------------------------------------------------
实时逐条日志推理版本（PyOD AutoEncoder）
 - 从 Kafka 实时读取 df_final 格式日志
 - 加载已训练的 AutoEncoder 模型与预处理器
 - 每条日志 → 预处理 → 异常预测
---------------------------------------------------------
"""

# import pandas as pd
# import joblib
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, from_json, pandas_udf, struct
# from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
# from sklearn.base import BaseEstimator, TransformerMixin

# # ==========================================================
# # 🧩 配置部分
# # ==========================================================
# KAFKA_BROKER = "kafka-kraft:9092"
# KAFKA_TOPIC_IN = "monitoring-data"
# CHECKPOINT_PATH = "/opt/spark-data/_checkpoints_infer_autoencoder"
# MODEL_PATH = "/opt/spark-apps/models/pyod_autoencoder.pkl"
# PREPROCESSOR_PATH = "/opt/spark-apps/models/pyod_preprocessor.pkl"
# TRIGGER_INTERVAL = "30 seconds"

# # ==========================================================
# # 🧩 初始化 Spark
# # ==========================================================
# spark = (
#     SparkSession.builder.appName("RealtimeInferenceAutoEncoder")
#     .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50")
#     .config("spark.sql.shuffle.partitions", "2")
#     .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
#     .getOrCreate()
# )
# spark.sparkContext.setLogLevel("ERROR")


# # ========== 修复 TextSelector 反序列化问题 ==========
# class TextSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, key):
#         self.key = key

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X[self.key].astype(str).fillna("")


# print("📦 Loading PyOD AutoEncoder model and preprocessor...")
# preproc = joblib.load(PREPROCESSOR_PATH)
# model = joblib.load(MODEL_PATH)
# print("✅ Model and preprocessor loaded successfully!")

# # ==========================================================
# # 🧩 Kafka 数据结构（与 df_final 对齐）
# # ==========================================================
# schema = StructType(
#     [
#         StructField("source_type", StringType()),
#         StructField("ingest_time", StringType()),
#         StructField("access_client", StringType()),
#         StructField("access_request", StringType()),
#         StructField("access_status", DoubleType()),
#         StructField("response_size", DoubleType()),
#         StructField("request_time", DoubleType()),
#         StructField("user_agent", StringType()),
#         StructField("error_message", StringType()),
#         StructField("syslog_message", StringType()),
#         StructField("syslog_identifier", StringType()),
#         StructField("syslog_priority", DoubleType()),
#         StructField("syslog_facility", DoubleType()),
#         StructField("gh_status", StringType()),
#         StructField("gh_conclusion", StringType()),
#         StructField("commit_author", StringType()),
#         StructField("commit_message", StringType()),
#         StructField("cmdb_service", StringType()),
#         StructField("cmdb_domain", StringType()),
#         StructField("cmdb_language", StringType()),
#         StructField("cmdb_deployment_env", StringType()),
#     ]
# )

# # ==========================================================
# # 🧩 Step 1. 从 Kafka 读取实时日志流
# # ==========================================================
# df_kafka = (
#     spark.readStream.format("kafka")
#     .option("kafka.bootstrap.servers", KAFKA_BROKER)
#     .option("subscribe", KAFKA_TOPIC_IN)
#     .option("startingOffsets", "latest")
#     .load()
# )

# df_raw = df_kafka.selectExpr("CAST(value AS STRING) as json_str")
# # df_raw.writeStream.format("console").option("truncate", False).start()

# df_parsed = df_raw.withColumn("data", from_json(col("json_str"), schema)).select(
#     "data.*"
# )


# # ==========================================================
# # 🧩 Step 2. 定义 Pandas UDF 进行 AutoEncoder 推理
# # ==========================================================
# @pandas_udf("string")
# def predict_autoencoder(struct_series: pd.Series) -> pd.Series:
#     """结构化数据 → DataFrame → 预处理 → AutoEncoder 推理"""
#     if struct_series.empty:
#         return pd.Series([])

#     # 结构化转换
#     records = []
#     for r in struct_series:
#         if isinstance(r, dict):
#             records.append(r)
#         else:
#             try:
#                 records.append(r.asDict(recursive=True))
#             except Exception:
#                 records.append({})
#     pdf = pd.DataFrame(records)
#     input_len = len(struct_series)

#     # 缺失值补齐
#     pdf = pdf.fillna(0)

#     try:
#         # 预处理 + 模型推理
#         X = preproc.transform(pdf)
#         preds = model.predict(X)  # 0=normal, 1=anomaly
#         result = pd.Series(["anomaly" if p == 1 else "normal" for p in preds])
#     except Exception as e:
#         print(f"[ERROR] Prediction failed: {e}")
#         result = pd.Series(["error"] * input_len)

#     # 保证长度匹配
#     if len(result) != input_len:
#         print(
#             f"[WARN] Result length mismatch: expected {input_len}, got {len(result)}; padding..."
#         )
#         while len(result) < input_len:
#             result = pd.concat([result, pd.Series(["error"])], ignore_index=True)
#         result = result[:input_len]

#     return result


# # ==========================================================
# # 🧩 Step 3. 执行实时推理
# # ==========================================================
# df_pred = df_parsed.withColumn(
#     "prediction", predict_autoencoder(struct(*[col(c) for c in df_parsed.columns]))
# )

# # ==========================================================
# # 🧩 Step 4. 输出结果
# # ==========================================================
# query = (
#     df_pred.select(
#         "source_type", "ingest_time", "access_client", "access_status", "prediction"
#     )
#     .writeStream.outputMode("append")
#     .format("console")
#     .option("truncate", True)
#     .option("checkpointLocation", CHECKPOINT_PATH)
#     .trigger(processingTime=TRIGGER_INTERVAL)
#     .start()
# )

# print("🚀 Real-time AutoEncoder inference started...")
# query.awaitTermination()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spark Structured Streaming Realtime Inference (Fixed Version)
-------------------------------------------------------------
全量支持：nginx_access / nginx_error / syslog / github_actions / github_commits / cmdb
从 Kafka 实时读取 → 多源解析 → 统一 df_final → PyOD AutoEncoder 实时推理
✅ 已修复: ContextOnlyValidOnDriver / PicklingError
"""

import os
import joblib
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    LongType,
    ArrayType,
)
from pyspark.sql.functions import (
    col,
    get_json_object,
    from_json,
    regexp_extract,
    regexp_replace,
    explode_outer,
    pandas_udf,
)
from pyspark.sql import functions as F


# ==================== Config ====================
KAFKA_BROKER = "kafka-kraft:9092"
KAFKA_TOPIC_IN = "monitoring-data"
MODEL_DIR = "/opt/spark-apps/models"
MODEL_PATH = os.path.join(MODEL_DIR, "pyod_autoencoder.pkl")
PREPROC_PATH = os.path.join(MODEL_DIR, "pyod_preprocessor.pkl")

# ==================== Spark Init ====================
spark = SparkSession.builder.appName("AIOps-Stream-Inference-Full-Fixed").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


from sklearn.base import BaseEstimator, TransformerMixin


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key].astype(str).fillna("")


print(f"📦 Using model paths:\n - MODEL: {MODEL_PATH}\n - PREPROC: {PREPROC_PATH}")
print("✅ Spark Session initialized.")


# ==================== Step 0: Kafka Source ====================
df_raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BROKER)
    .option("subscribe", KAFKA_TOPIC_IN)
    .option("startingOffsets", "latest")
    .load()
)
df_raw = df_raw.selectExpr("CAST(value AS STRING) as json_str")

# ==================== Step 1: 通用字段提取 ====================
df_base = df_raw.select(
    get_json_object(col("json_str"), "$.source_type").alias("source_type"),
    get_json_object(col("json_str"), "$.timestamp").alias("ingest_time"),
    get_json_object(col("json_str"), "$.message").alias("message"),
)

# ==================== Step 2: 各类日志解析 ====================

# --- Nginx Access ---
nginx_access_schema = StructType(
    [
        StructField("time_local", StringType()),
        StructField("remote_addr", StringType()),
        StructField("request", StringType()),
        StructField("status", StringType()),
        StructField("body_bytes_sent", StringType()),
        StructField("request_time", StringType()),
        StructField("http_referer", StringType()),
        StructField("http_user_agent", StringType()),
    ]
)

df_nginx = (
    df_base.filter(col("source_type") == "nginx_access")
    .withColumn("json_data", from_json(col("message"), nginx_access_schema))
    .select(
        col("source_type"),
        col("ingest_time"),
        col("json_data.remote_addr").alias("access_client"),
        col("json_data.request").alias("access_request"),
        col("json_data.status").cast("int").alias("access_status"),
        col("json_data.body_bytes_sent").cast("int").alias("response_size"),
        col("json_data.request_time").cast("double").alias("request_time"),
        col("json_data.http_referer").alias("http_referer"),
        col("json_data.http_user_agent").alias("user_agent"),
    )
)

# --- Nginx Error ---
df_nginx_error = df_base.filter(col("source_type") == "nginx_error").select(
    col("source_type"),
    col("ingest_time"),
    regexp_extract("message", r":\s+(.*?)(?:, client:|$)", 1).alias("error_message"),
    regexp_extract("message", r"client:\s*([\d\.]+)", 1).alias("error_client"),
    regexp_extract("message", r"server:\s*([\w\.\-]+)", 1).alias("error_server"),
    regexp_extract("message", r"request:\s*\"(.*?)\"", 1).alias("error_request"),
    regexp_extract("message", r"host:\s*\"(.*?)\"", 1).alias("error_host"),
)

# --- Syslog ---
syslog_schema = StructType(
    [
        StructField("MESSAGE_ID", StringType()),
        StructField("PRIORITY", StringType()),
        StructField("SYSLOG_FACILITY", StringType()),
        StructField("SYSLOG_IDENTIFIER", StringType()),
        StructField("_PID", StringType()),
        StructField("host", StringType()),
        StructField("message", StringType()),
    ]
)

df_syslog = (
    df_base.filter(col("source_type") == "syslog")
    .withColumn("json_data", from_json(col("message"), syslog_schema))
    .select(
        col("source_type"),
        col("ingest_time"),
        col("json_data.host").alias("host"),
        col("json_data.message").alias("syslog_message"),
        col("json_data.SYSLOG_IDENTIFIER").alias("syslog_identifier"),
        col("json_data.PRIORITY").cast("int").alias("syslog_priority"),
        col("json_data.SYSLOG_FACILITY").cast("int").alias("syslog_facility"),
    )
)

# --- GitHub Actions ---
github_schema = StructType(
    [
        StructField("total_count", LongType()),
        StructField(
            "workflow_runs",
            ArrayType(
                StructType(
                    [
                        StructField("id", LongType()),
                        StructField("name", StringType()),
                        StructField("status", StringType()),
                        StructField("conclusion", StringType()),
                        StructField("run_started_at", StringType()),
                        StructField("updated_at", StringType()),
                        StructField("head_branch", StringType()),
                    ]
                )
            ),
        ),
    ]
)

df_github_actions = (
    df_base.filter(col("source_type") == "github_actions")
    .withColumn("gh_parsed", from_json(col("message"), github_schema))
    .withColumn("run", explode_outer(col("gh_parsed.workflow_runs")))
    .select(
        col("source_type"),
        col("ingest_time"),
        col("run.id").alias("gh_run_id"),
        col("run.name").alias("gh_name"),
        col("run.status").alias("gh_status"),
        col("run.conclusion").alias("gh_conclusion"),
        col("run.run_started_at").alias("gh_run_started_at"),
        col("run.updated_at").alias("gh_updated_at"),
        col("run.head_branch").alias("gh_head_branch"),
    )
)

# --- GitHub Commits ---
github_commits_schema = ArrayType(
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

df_github_commits = (
    df_base.filter(col("source_type") == "github_commits")
    .withColumn("message_clean", regexp_replace(col("message"), r"\\\\", r""))
    .withColumn("commit_array", from_json(col("message_clean"), github_commits_schema))
    .withColumn("commit", explode_outer(col("commit_array")))
    .select(
        col("source_type"),
        col("ingest_time"),
        col("commit.sha").alias("commit_sha"),
        col("commit.commit.author.name").alias("commit_author"),
        col("commit.commit.author.email").alias("commit_email"),
        col("commit.commit.author.date").alias("commit_date"),
        col("commit.commit.message").alias("commit_message"),
        col("commit.author.login").alias("github_user"),
        col("commit.html_url").alias("commit_url"),
    )
)

# --- CMDB ---
df_cmdb = df_base.filter(col("source_type") == "cmdb").select(
    col("source_type"),
    col("ingest_time"),
    get_json_object(col("message"), "$.service_name").alias("cmdb_service"),
    get_json_object(col("message"), "$.domain").alias("cmdb_domain"),
    get_json_object(col("message"), "$.owner").alias("cmdb_owner"),
    get_json_object(col("message"), "$.language").alias("cmdb_language"),
    get_json_object(col("message"), "$.deployment_env").alias("cmdb_deployment_env"),
)

# ==================== Step 3: Union All Sources ====================
df_final = (
    df_nginx.unionByName(df_nginx_error, allowMissingColumns=True)
    .unionByName(df_syslog, allowMissingColumns=True)
    .unionByName(df_github_actions, allowMissingColumns=True)
    .unionByName(df_github_commits, allowMissingColumns=True)
    .unionByName(df_cmdb, allowMissingColumns=True)
)


# ==================== Step 4: 惰性加载推理函数 ====================
@pandas_udf("string")
def predict_autoencoder_udf(*cols):
    import joblib
    import pandas as pd
    import os

    # --- 每个 Executor 加载一次 ---
    global _model, _preproc
    if "_model" not in globals():
        print("🧠 Loading model inside executor...")
        _model = joblib.load(MODEL_PATH)
        _preproc = joblib.load(PREPROC_PATH)

    pdf = pd.concat(cols, axis=1)
    pdf.columns = df_final.columns

    try:
        cols_keep = [
            "source_type",
            "access_client",
            "access_request",
            "access_status",
            "response_size",
            "request_time",
            "user_agent",
            "error_message",
            "syslog_message",
            "syslog_identifier",
            "syslog_priority",
            "syslog_facility",
            "gh_status",
            "gh_conclusion",
            "commit_author",
            "commit_message",
            "cmdb_service",
            "cmdb_domain",
            "cmdb_language",
            "cmdb_deployment_env",
        ]
        df_model = pdf[[c for c in cols_keep if c in pdf.columns]].fillna("")
        X = _preproc.transform(df_model)
        preds = _model.predict(X)
        return pd.Series(["anomaly" if p == 1 else "normal" for p in preds])
    except Exception as e:
        return pd.Series(["error"] * len(pdf))


# ==================== Step 5: 应用推理 ====================
df_pred = df_final.withColumn(
    "prediction", predict_autoencoder_udf(*[col(c) for c in df_final.columns])
)

# ==================== Step 6: 输出 ====================
query_console = (
    df_pred.select(
        "source_type", "ingest_time", "access_client", "access_status", "prediction"
    )
    .writeStream.outputMode("append")
    .format("console")
    .option("truncate", "false")
    .option("numRows", 20)
    .start()
)

output_path = "/opt/spark-data"
query_file = (
    df_pred.writeStream.outputMode("append")
    .format("parquet")
    .option("path", output_path + "/inference_full")
    .option("checkpointLocation", output_path + "/_checkpoints_infer_full")
    .trigger(processingTime="600 seconds")
    .start()
)

query_console.awaitTermination()
query_file.awaitTermination()
