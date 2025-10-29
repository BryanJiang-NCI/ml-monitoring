FROM python:3.10-slim-bookworm

WORKDIR /app

COPY main.py .

RUN apt-get update && \
    apt-get install -y --no-install-recommends stress-ng && \
    pip install --no-cache-dir fastapi uvicorn prometheus-client prometheus-fastapi-instrumentator && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
