"""
root_cause_listener.py
=========================
🏗️ Event-driven Root Cause Inference on Spark
------------------------------------------------
持续监听 feedback_samples.jsonl 文件，
当发现新的人工确认异常（feedback_label=true）时，
自动执行语义匹配根因分析并打印结果。

"""

import os
import json
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from pyspark.sql import SparkSession

# ===============================
# 🔧 基础路径配置
# ===============================
BASE_DIR = "/opt/spark/work-dir"
MODEL_PATH = f"{BASE_DIR}/models/root_cause_model"
CMDB_FILE = f"{BASE_DIR}/data/cmdb.jsonl"
FEEDBACK_FILE = f"{BASE_DIR}/data/feedback_samples.jsonl"

CHECK_INTERVAL = 5  # 每5秒检查一次新样本
PROCESSED_RECORDS = set()  # 记录已处理的样本行

# ===============================
# ⚙️ 初始化 Spark 与模型
# ===============================
spark = SparkSession.builder.appName("RootCauseListener").getOrCreate()

print("🚀 Starting Root Cause Listener (event-driven mode)...")
print(f"📡 Monitoring feedback file: {FEEDBACK_FILE}")

encoder = SentenceTransformer(MODEL_PATH)
cmdb = pd.read_json(CMDB_FILE, lines=True)
entities = cmdb["service_name"].tolist()
entity_vecs = encoder.encode(entities, convert_to_tensor=True)


# ===============================
# 🧠 根因分析函数
# ===============================
def infer_root_cause(anomaly_text: str, top_k: int = 3):
    anomaly_vec = encoder.encode(anomaly_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(anomaly_vec, entity_vecs)[0]
    top_results = np.argsort(-cos_scores)[:top_k]
    results = [
        {"entity": entities[i], "confidence": float(cos_scores[i])} for i in top_results
    ]
    return results


# ===============================
# 🔁 文件监听循环
# ===============================
while True:
    try:
        if not os.path.exists(FEEDBACK_FILE):
            time.sleep(CHECK_INTERVAL)
            continue

        with open(FEEDBACK_FILE, "r") as f:
            for line in f:
                if not line.strip() or line in PROCESSED_RECORDS:
                    continue

                PROCESSED_RECORDS.add(line)
                data = json.loads(line)

                # 仅处理人工确认的异常
                if str(data.get("feedback_label", "")).lower() != "true":
                    continue

                text = data.get("semantic_text", "")
                if not text:
                    continue

                print("\n🧩 New confirmed anomaly detected!")
                print(f"⏱️ Timestamp: {data.get('timestamp')}")
                print(f"🧾 Text: {text[:180]}...")

                # 执行根因分析
                results = infer_root_cause(text)
                print("🔍 Root Cause Candidates:")
                for r in results:
                    print(f" - {r['entity']} (confidence={r['confidence']:.3f})")

                print("✅ Root cause inference completed.\n")

        time.sleep(CHECK_INTERVAL)

    except Exception as e:
        print(f"⚠️ Listener error: {e}")
        time.sleep(CHECK_INTERVAL)
