"""
root_cause_train_stable_lean.py
======================================
稳定弱监督 RCA 训练脚本 (精简版)

关键特性：
1. 保持 CMDB/服务侧 Embedding 稳定（冻结）
2. 仅训练 Log 侧的 Semantic Encoder
"""

import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

BASE_DIR = "/opt/spark/work-dir"
MODEL_SAVE_DIR = f"{BASE_DIR}/models/root_cause_model"
CMDB_FILE = f"{BASE_DIR}/data/cmdb.jsonl"
FEEDBACK_FILE = f"{BASE_DIR}/data/feedback_samples.jsonl"
MODEL_NAME = "all-MiniLM-L12-v2"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 1. CMDB 加载与实体构建 ---
cmdb = pd.read_json(CMDB_FILE, lines=True)

# 实体文本构建 (使用 lambda 简化)
cmdb["entity_text"] = cmdb.apply(
    lambda r: f"{r.get('service_name','')} {r.get('domain','')} {r.get('system','')} {' '.join(r.get('dependencies',[]))}",
    axis=1,
)
svc_map = dict(zip(cmdb["service_name"], cmdb["entity_text"]))
print(f"📦 Loaded {len(cmdb)} services.")


# --- 2. 弱监督标签提取函数 ---
def extract_service(text, svc_names):
    # 查找 URL 模式
    if m := re.search(r"http://([^:/]+)", text):
        return m.group(1)
    # 查找服务名模式
    matched = [s for s in svc_names if s in text]
    if matched:
        return matched[0]
    # 查找 name= 模式
    if m := re.search(r"name=([a-zA-Z0-9_\-]+)", text):
        return m.group(1).split("_")[0]
    return None


# --- 3. 数据加载与样本构建 (最优化) ---
feedback = pd.read_json(FEEDBACK_FILE, lines=True)
feedback = feedback[feedback["feedback_label"] == "true"]
print(f"📦 Loaded {len(feedback)} confirmed anomalies.")

# 列表推导式构建训练样本 (一行完成筛选和映射)
train_samples = [
    InputExample(texts=[row["semantic_text"], svc_map[pos_svc]])
    for _, row in feedback.iterrows()
    if (pos_svc := extract_service(row["semantic_text"], svc_map.keys()))  # 提取服务
    and pos_svc in svc_map  # 确保服务在 CMDB 中
]

print(f"🎯 Training pairs: {len(train_samples)}")

# --- 4. 稳定训练 (仅训练 Semantic Encoder) ---
semantic_encoder = SentenceTransformer(MODEL_NAME)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)
train_loss = losses.MultipleNegativesRankingLoss(semantic_encoder)

print("🚀 Training semantic-text encoder for RCA...")

# 训练配置
semantic_encoder.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=50,
    show_progress_bar=True,
)

semantic_encoder.save(MODEL_SAVE_DIR)

print(f"🏁 Stable RCA model saved to {MODEL_SAVE_DIR}")
