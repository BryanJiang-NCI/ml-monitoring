FROM apache/spark:latest

USER root
WORKDIR /opt/spark/work-dir

ENV HOME=/opt/spark/work-dir

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-dev build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_DISABLE_XET=1
ENV HF_HOME=/opt/spark/work-dir/models/huggingface