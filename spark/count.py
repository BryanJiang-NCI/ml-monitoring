from pyspark.sql import SparkSession

# 数据目录路径
DATA_DIR = "/opt/spark/work-dir/data/semantic_vectors"

spark = SparkSession.builder.appName("CountTrainingDatasetRecords").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(DATA_DIR)
count = df.count()

print(f"✅ Total records: {count:,}")
spark.stop()
