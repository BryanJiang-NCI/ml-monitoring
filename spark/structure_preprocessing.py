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
