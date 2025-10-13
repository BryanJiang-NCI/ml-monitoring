from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    get_json_object,
    regexp_extract,
    from_json,
    explode_outer,
    regexp_replace,
    coalesce,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    DoubleType,
    ArrayType,
    MapType,
)
from pyspark.sql import functions as F
from pyspark.sql.functions import window, count, collect_list, first


# ========== Spark Init ==========
spark = (
    SparkSession.builder.appName("LogPreprocessingMultiSource")
    .config("spark.driver.memory", "2g")
    .config("spark.executor.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# ========== Step 0: Kafka Source ==========
df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "kafka-kraft:9092")
    .option("subscribe", "monitoring-data")
    .option("startingOffsets", "earliest")
    .load()
)
df = df.selectExpr("CAST(value AS STRING) as json_str")

# ========== Step 1: 通用字段提取 ==========
df_base = df.select(
    get_json_object(col("json_str"), "$.source_type").alias("source_type"),
    get_json_object(col("json_str"), "$.timestamp").alias("ingest_time"),
    get_json_object(col("json_str"), "$.message").alias("message"),
)

# ========== Step 2: 各类日志分流解析 ==========

# --- Nginx ---
nginx_access_schema = StructType(
    [
        StructField("time_local", StringType()),
        StructField("remote_addr", StringType()),
        StructField("request_method", StringType()),
        StructField("request_uri", StringType()),
        StructField("status", StringType()),
        StructField("body_bytes_sent", StringType()),
        StructField("request_time", StringType()),
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
        col("json_data.request_method").alias("access_method"),
        col("json_data.request_uri").alias("access_uri"),
        col("json_data.status").cast("int").alias("access_status"),
        col("json_data.body_bytes_sent").cast("int").alias("response_size"),
        col("json_data.request_time").cast("double").alias("request_time"),
        col("json_data.http_user_agent").alias("user_agent"),
    )
)

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
        StructField("PRIORITY", StringType()),
        StructField("SYSLOG_FACILITY", StringType()),
        StructField("SYSLOG_IDENTIFIER", StringType()),
        StructField("host", StringType()),
        StructField("message", StringType()),
        StructField("source_type", StringType()),
        StructField("ingest_time", StringType()),
    ]
)

df_syslog = (
    df_base.filter(col("source_type") == "syslog")
    .withColumn("json_data", from_json(col("message"), syslog_schema))
    .select(
        col("json_data.source_type"),
        col("json_data.ingest_time"),
        col("json_data.host"),
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
            StructField(
                "author",
                StructType([StructField("login", StringType())]),
            ),
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

# host_metrics_schema = StructType(
#     [
#         StructField("name", StringType()),
#         StructField("host", StringType()),
#         StructField("source_type", StringType()),
#         StructField("ingest_time", StringType()),
#         StructField("counter", StructType([StructField("value", DoubleType())]), True),
#         StructField("gauge", StructType([StructField("value", DoubleType())]), True),
#     ]
# )
# df_host_metrics = (
#     df_base.filter(col("source_type") == "host_metrics")
#     .withColumn("json_data", from_json(col("message"), host_metrics_schema))
#     .select(
#         col("json_data.source_type"),
#         col("json_data.ingest_time"),
#         col("json_data.name").alias("metric_name"),
#         col("json_data.host").alias("host"),
#         coalesce(col("json_data.counter.value"), col("json_data.gauge.value")).alias(
#             "metric_value"
#         ),
#     )
# )
#     )
df_host_metrics = df_base.filter(col("source_type") == "host_metrics").select(
    col("source_type"),
    get_json_object(col("message"), "$.timestamp").alias("ingest_time"),
    get_json_object(col("message"), "$.name").alias("metric_name"),
    get_json_object(col("message"), "$.host").alias("host"),
    coalesce(
        get_json_object(col("message"), "$.counter.value").cast("double"),
        get_json_object(col("message"), "$.gauge.value").cast("double"),
    ).alias("metric_value"),
    # get_json_object(col("message"), "$.tags.collector").alias("metric_collector"),
    # get_json_object(col("message"), "$.tags.cpu").alias("metric_cpu"),
    # get_json_object(col("message"), "$.tags.mode").alias("metric_mode"),
    # get_json_object(col("message"), "$.namespace").alias("metric_namespace"),
)

# ========== Step 3: 合并为宽表结构 ==========
df_final = (
    df_nginx.unionByName(df_syslog, allowMissingColumns=True)
    .unionByName(df_github_actions, allowMissingColumns=True)
    .unionByName(df_github_commits, allowMissingColumns=True)
    .unionByName(df_cmdb, allowMissingColumns=True)
    .unionByName(df_nginx_error, allowMissingColumns=True)
    .unionByName(df_host_metrics, allowMissingColumns=True)
)

# ========== Step 4: 输出到控制台 ==========
query_console = (
    df_final.writeStream.outputMode("append")
    .format("console")
    .option("truncate", "false")
    .option("numRows", 5)
    .start()
)

output_path = "/opt/spark-data"
query = (
    df_final.writeStream.outputMode("append")
    .format("parquet")  # or "json"
    .option("path", output_path + "/parquet")
    .option("checkpointLocation", output_path + "/_checkpoints")
    .trigger(processingTime="60 seconds")  # 每分钟写入一次
    .start()
)

query_console.awaitTermination()
query.awaitTermination()
