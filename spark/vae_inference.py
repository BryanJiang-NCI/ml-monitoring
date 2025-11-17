"""
VAE Streaming Inference Script
==============================
✅ 从 Kafka 实时消费日志 → SentenceTransformer 编码 → 标准化
✅ 使用训练好的 VAE 模型推理重构误差
✅ 输出异常分数与标签
"""

import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, get_json_object
from pyspark.sql.types import StringType


# ==========================================================
# 🧩 Variational AutoEncoder
# ==========================================================
class VAE(nn.Module):
    def __init__(self, input_dim=384, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ==========================================================
# 🧩 Paths
# ==========================================================
BASE_DIR = "/opt/spark/work-dir"
MODEL_DIR = os.path.join(BASE_DIR, "models", "vae_model")
ANOMALY_LOG_FILE = os.path.join(BASE_DIR, "data/anomaly.jsonl")

KAFKA_SERVERS = "kafka-kraft:9092"
KAFKA_TOPIC = "monitoring-data"
MODEL_NAME = "all-MiniLM-L12-v2"

# ==========================================================
# 🧠 Load Model + Scaler
# ==========================================================
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
threshold = float(joblib.load(os.path.join(MODEL_DIR, "threshold.pkl")))

encoder = SentenceTransformer(MODEL_NAME)
vae = VAE(input_dim=384, latent_dim=32)
vae.load_state_dict(torch.load(os.path.join(MODEL_DIR, "vae.pth"), map_location="cpu"))
vae.eval()

print(f"✅ VAE model loaded. Threshold={threshold:.6f}")

# ==========================================================
# 🔍 Spark init
# ==========================================================
spark = (
    SparkSession.builder.appName("VAEStreamingInference")
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVERS)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)
df_raw = df_kafka.selectExpr("CAST(value AS STRING) as message")


# ==========================================================
# 🧩 Step 4. JSON → Semantic Sentence（保持一致）
# ==========================================================
def json_to_semantic(text):
    """
    无模板的 JSON → 自然语言生成器
    自动 verbalize JSON 的 key 和 value。
    支持任意结构化数据源，无需模型或模板。
    """

    try:
        obj = json.loads(text)

        # 取内部 message 字段
        if isinstance(obj, dict) and "message" in obj:
            try:
                msg = json.loads(obj["message"])
            except:
                msg = obj["message"]
        else:
            msg = obj

        # 统一为 dict
        if not isinstance(msg, dict):
            return f"The event reports: {str(msg)}."

        fragments = []

        for k, v in msg.items():
            # 去掉时间类字段
            if any(t in k.lower() for t in ["time", "timestamp", "date"]):
                continue

            # key 自然词化（下划线 → 空格）
            k2 = k.replace("_", " ").replace("-", " ").strip()

            # value 正常处理
            if isinstance(v, dict):
                v2 = f"a structured value containing {len(v)} fields"
            else:
                v2 = str(v)

            fragments.append(f"{k2} is {v2}")

        if not fragments:
            return "This event contains non-time information but no describable fields."

        sentence = ", ".join(fragments)
        return f"This event indicates that {sentence}."

    except Exception:
        return "[INVALID_JSON]"


semantic_udf = udf(json_to_semantic, StringType())
df_semantic = df_raw.withColumn("semantic_text", semantic_udf(col("message")))


# ==========================================================
# ⚙️ Inference UDF
# ==========================================================
def infer_semantic(text):
    try:
        emb = encoder.encode(text)
        emb_scaled = scaler.transform([emb]).astype(np.float32)
        x = torch.tensor(emb_scaled)

        with torch.no_grad():
            recon, mu, logvar = vae(x)
            mse = torch.mean((x - recon) ** 2, dim=1).item()

        label = "abnormal" if mse > threshold else "normal"
        result = {
            "semantic_text": text,
            "distance": round(mse, 6),
            "threshold": round(threshold, 6),
            "prediction": label,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if label != "normal":
            with open(ANOMALY_LOG_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"prediction": "error", "error": str(e)})


infer_udf = udf(infer_semantic, StringType())
df_pred = df_semantic.withColumn("result", infer_udf(col("semantic_text")))

# ==========================================================
# 📊 Output
# ==========================================================
query_console = (
    df_pred.select("semantic_text", "result")
    .writeStream.outputMode("append")
    .format("console")
    .option("truncate", False)
    .option("numRows", 5)
    .start()
)

print(f"📡 Streaming inference started from Kafka topic: {KAFKA_TOPIC}")
query_console.awaitTermination()
