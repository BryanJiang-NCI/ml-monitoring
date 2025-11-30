#!/bin/bash
set -e

echo "=== Preparing Datasets ==="

SPARK_CONTAINER="spark-master"


# ----------------------------
# 2~4: The rest must run in Spark container
# ----------------------------
# Check Spark container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${SPARK_CONTAINER}$"; then
    echo "ERROR: Spark master container '$SPARK_CONTAINER' is not running."
    echo "Please start your stack first:  docker compose up -d"
    exit 1
fi

echo "[2/4] Running semantic preprocessing..."
docker exec -it $SPARK_CONTAINER \
    run_spark semantic_preprocessing.py
echo "✓ Semantic dataset ready."

echo "[3/4] Running structured preprocessing..."
docker exec -it $SPARK_CONTAINER \
    run_spark structure_preprocessing.py
echo "✓ Structured dataset ready."

echo "[4/4] Generating labeled parquet test set..."
docker exec -it $SPARK_CONTAINER \
    run_spark generate_testset.py
echo "✓ Test set parquet generated."

echo "=== Dataset Preparation Completed Successfully ==="


echo "[1/4] Generating CMDB (host-side Python)..."
source venv/bin/activate
python spark/generate_cmdb.py
echo "✓ CMDB generated at spark/data/cmdb.jsonl"