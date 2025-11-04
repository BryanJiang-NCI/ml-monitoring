"""
root_cause_train.py
====================
Train Root Cause Correlation Model
----------------------------------
输入:
 - feedback_samples.jsonl (历史确认的异常样本)
 - cmdb_entities.jsonl (系统中服务/组件/主机的CMDB数据)
输出:
 - 保存到 models/root_cause_model
"""

import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

BASE_DIR = "/opt/spark/work-dir"
MODEL_SAVE_DIR = f"{BASE_DIR}/models/root_cause_model"
FEEDBACK_FILE = f"{BASE_DIR}/data/feedback_samples.jsonl"
CMDB_FILE = f"{BASE_DIR}/data/cmdb.jsonl"
MODEL_NAME = "all-MiniLM-L12-v2"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Step 1️⃣ 读取数据
feedback = pd.read_json(FEEDBACK_FILE, lines=True)
cmdb = pd.read_json(CMDB_FILE, lines=True)

# 仅保留带 label 的样本
feedback = feedback.dropna(subset=["semantic_text", "feedback_label"])
print(f"✅ Loaded {len(feedback)} labeled feedback samples.")

# Step 2️⃣ 组合训练样本
train_samples = []
for _, row in feedback.iterrows():
    train_samples.append(
        InputExample(texts=[row["semantic_text"], row["feedback_label"]])
    )

# Step 3️⃣ 模型初始化
encoder = SentenceTransformer(MODEL_NAME)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)
train_loss = losses.MultipleNegativesRankingLoss(encoder)

# Step 4️⃣ 训练
print("🚀 Training root cause model...")
encoder.fit(
    train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=50
)
encoder.save(MODEL_SAVE_DIR)
print(f"✅ Root Cause Model saved to {MODEL_SAVE_DIR}")
