#!/bin/bash
# ==============================================================
# 🚀 Generic Spark Submit Script
# Author: Bryan
# Description:
#   Run any Spark job by specifying the Python script name only.
#   Example: bash run_spark.sh semantic_train.py
# ==============================================================

# === 参数检查 ===
if [ -z "$1" ]; then
  echo "❌ Usage: bash run_spark.sh <script_name.py>"
  echo "Example: bash run_spark.sh semantic_train.py"
  exit 1
fi

SCRIPT_NAME=$1
APP_PATH="/opt/spark/models/${SCRIPT_NAME}"
SPARK_MASTER_URL="spark://spark-master:7077"
SPARK_BIN="/opt/bitnami/spark/bin/spark-submit"

echo "🚀 Submitting Spark job for ${SCRIPT_NAME} ..."
echo "----------------------------------------------"

# === submit spark job ===
${SPARK_BIN} \
  --master ${SPARK_MASTER_URL} \
  --deploy-mode client \
  --driver-memory 4G \
  --executor-memory 2G \
  --executor-cores 2 \
  --conf spark.jars.ivy=/tmp/.ivy2 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1 \
  ${APP_PATH}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "✅ Spark job ${SCRIPT_NAME} completed successfully."
else
  echo "❌ Spark job ${SCRIPT_NAME} failed (exit code: ${EXIT_CODE})."
fi
