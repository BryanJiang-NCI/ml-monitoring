#!/bin/bash
# ==============================================================
# 🚀 Generic Spark Submit Script
# Author: Bryan
# Description:
#   Run any Spark job by specifying the Python script name only.
#   It will automatically install Python dependencies from
#   /opt/spark/work-dir/requirements.txt if available.
#   Example: bash run_spark.sh semantic_train.py
# ==============================================================

# === 参数检查 ===
if [ -z "$1" ]; then
  echo "❌ Usage: bash run_spark.sh <script_name.py>"
  echo "Example: bash run_spark.sh semantic_train.py"
  exit 1
fi

SCRIPT_NAME=$1
APP_PATH="/opt/spark/work-dir/${SCRIPT_NAME}"
SPARK_MASTER_URL="spark://spark-master:7077"
SPARK_BIN="/opt/spark/bin/spark-submit"
REQ_FILE="/opt/spark/work-dir/requirements.txt"

echo ""
echo "🚀 Submitting Spark job for ${SCRIPT_NAME} ..."
echo "----------------------------------------------"

# # === Step 1. 自动安装依赖 ===
# if [ -f "${REQ_FILE}" ]; then
#   echo "📦 Installing Python dependencies from ${REQ_FILE} ..."
#   pip install -r ${REQ_FILE} --no-cache-dir
#   if [ $? -ne 0 ]; then
#     echo "⚠️ Dependency installation failed, continuing anyway..."
#   else
#     echo "✅ Dependencies installed successfully."
#   fi
# else
#   echo "ℹ️ No requirements.txt found — skipping dependency installation."
# fi

# === Step 2. 检查 Spark 脚本是否存在 ===
if [ ! -f "${APP_PATH}" ]; then
  echo "❌ Spark app not found at: ${APP_PATH}"
  exit 1
fi

# === Step 3. 提交 Spark 任务 ===
${SPARK_BIN} \
  --master ${SPARK_MASTER_URL} \
  --driver-memory 512M \
  --executor-memory 512M \
  --executor-cores 1 \
  --conf spark.cores.max=5 \
  --conf spark.jars.ivy=/tmp/.ivy2 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1 \
  ${APP_PATH}

EXIT_CODE=$?

# === Step 4. 结果输出 ===
if [ ${EXIT_CODE} -eq 0 ]; then
  echo "✅ Spark job ${SCRIPT_NAME} completed successfully."
else
  echo "❌ Spark job ${SCRIPT_NAME} failed (exit code: ${EXIT_CODE})."
fi
