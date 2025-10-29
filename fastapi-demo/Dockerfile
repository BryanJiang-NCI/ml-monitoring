FROM python:3.10-slim-bookworm

WORKDIR /app

COPY main.py .

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    wget https://github.com/prometheus/node_exporter/releases/download/v1.8.1/node_exporter-1.8.1.linux-amd64.tar.gz && \
    tar -xzf node_exporter-1.8.1.linux-amd64.tar.gz && \
    mv node_exporter-1.8.1.linux-amd64/node_exporter /usr/local/bin/node_exporter && \
    rm -rf node_exporter-* && \
    pip install --no-cache-dir fastapi uvicorn prometheus-client prometheus-fastapi-instrumentator && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

CMD sh -c "\
    nohup node_exporter --web.listen-address=:9100 >/dev/null 2>&1 & \
    uvicorn main:app --host 0.0.0.0 --port 8000 \
"