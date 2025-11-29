#!/bin/bash
set -e

docker exec -it $SPARK_CONTAINER \
    run_spark structure_train.py

docker exec -it $SPARK_CONTAINER \
    run_spark semantic_train.py

docker exec -it $SPARK_CONTAINER \
    run_spark root_cause_train.py