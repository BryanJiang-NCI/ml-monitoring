from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    get_json_object,
    regexp_extract,
    from_json,
    explode_outer,
    regexp_replace,
    when,
    collect_list,
    map_from_entries,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    DoubleType,
    ArrayType,
)
from pyspark.sql import functions as F

# ========== Spark Init ==========
spark = SparkSession.builder.appName("LogPreprocessingMultiSource").getOrCreate()
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
    get_json_object(col("json_str"), "$.timestamp").alias("timestamp"),
    get_json_object(col("json_str"), "$.message").alias("message"),
)

# ========== Step 2: 各类日志分流解析 ==========

# --- Nginx ---
df_nginx = df_base.filter(col("source_type") == "nginx").select(
    col("source_type"),
    regexp_extract("message", r"^(\S+)", 1).alias("client_ip"),
    regexp_extract("message", r" - (\S+) ", 1).alias("user"),
    regexp_extract("message", r"\[(.*?)\]", 1).alias("timestamp_local"),
    regexp_extract("message", r"\"(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)", 1).alias(
        "http_method"
    ),
    regexp_extract(
        "message", r"\"(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(\S+)", 1
    ).alias("url"),
    regexp_extract("message", r"(HTTP\/\d\.\d)", 1).alias("http_version"),
    regexp_extract("message", r"HTTP\/\d\.\d\"\s+(\d+)", 1).alias("status_code"),
    regexp_extract("message", r"HTTP\/\d\.\d\"\s+\d+\s+(\d+)", 1).alias(
        "response_size"
    ),
)

# --- Syslog ---
df_syslog = df_base.filter(col("source_type") == "syslog").select(
    col("source_type"),
    regexp_extract("message", r"^<(\d+)>", 1).alias("syslog_priority"),
    regexp_extract("message", r"^<\d+>(\d+)", 1).alias("syslog_version"),
    regexp_extract("message", r"<\d+>\d+\s+(\S+)", 1).alias("syslog_timestamp"),
    regexp_extract("message", r"kernel: (.*)", 1).alias("syslog_text"),
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
    get_json_object(col("message"), "$.service_name").alias("cmdb_service_name"),
    get_json_object(col("message"), "$.domain").alias("cmdb_domain"),
    get_json_object(col("message"), "$.owner").alias("cmdb_owner"),
    get_json_object(col("message"), "$.language").alias("cmdb_language"),
    get_json_object(col("message"), "$.deployment_env").alias("cmdb_deployment_env"),
)

# ========== Step 3: 合并为宽表结构 ==========
df_final = (
    df_nginx.unionByName(df_syslog, allowMissingColumns=True)
    .unionByName(df_github_actions, allowMissingColumns=True)
    .unionByName(df_github_commits, allowMissingColumns=True)
    .unionByName(df_cmdb, allowMissingColumns=True)
    # .unionByName(df_host_metrics, allowMissingColumns=True)
)

# ========== Step 4: 输出到控制台 ==========
query = (
    df_final.writeStream.outputMode("append")
    .format("console")
    .option("truncate", "false")
    .start()
)

query.awaitTermination()
