"""
Incremental (Feedback) Training Script for AutoEncoder
======================================================
✅ 读取 feedback_samples.jsonl（增量数据）
✅ 加载已保存模型与标准化器
✅ 微调 AutoEncoder 模型（2–3 个 epoch）
✅ 重新计算 MSE 与动态阈值
✅ 保存更新后的模型与阈值
✅ 清空 feedback 文件
======================================================
"""

import os
import torch
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from semantic_train import AutoEncoder

# ==========================================================
# 路径配置（与主模型保持一致）
# ==========================================================
BASE_DIR = "/opt/spark/work-dir"
MODEL_DIR = f"{BASE_DIR}/models/prediction_model"
FEEDBACK_FILE = f"{BASE_DIR}/data/feedback_samples.jsonl"
SCALER_FILE = f"{MODEL_DIR}/scaler.pkl"
MODEL_FILE = f"{MODEL_DIR}/autoencoder.pth"
THRESH_FILE = f"{MODEL_DIR}/threshold.pkl"
MODEL_NAME = "all-MiniLM-L12-v2"
hidden_dim = 64

# ==========================================================
# Step 1. 加载增量数据
# ==========================================================
if not os.path.exists(FEEDBACK_FILE) or os.path.getsize(FEEDBACK_FILE) == 0:
    print("⚠️ No new feedback samples found. Skip retraining.")
    exit(0)

df = pd.read_json(FEEDBACK_FILE, lines=True)
if df.empty:
    print("⚠️ Feedback file is empty. Skip retraining.")
    exit(0)

texts = df["semantic_text"].tolist()
print(f"📦 Loaded {len(texts)} new feedback samples.")

# ==========================================================
# Step 2. 编码与标准化
# ==========================================================
encoder = SentenceTransformer(MODEL_NAME)
scaler = joblib.load(SCALER_FILE)

X = encoder.encode(texts)
X_scaled = scaler.transform(X).astype(np.float32)
X_tensor = torch.tensor(X_scaled)

# ==========================================================
# Step 3. 加载旧模型
# ==========================================================
model = AutoEncoder(input_dim=X_tensor.shape[1], hidden_dim=hidden_dim)
model.load_state_dict(torch.load(MODEL_FILE))
model.encoder[2].p = 0.0
model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_tensor = X_tensor.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.MSELoss()

# ==========================================================
# Step 4. 微批增量训练
# ==========================================================
EPOCHS = 3
print(f"🚀 Starting incremental fine-tuning for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    recon = model(X_tensor)
    loss = criterion(recon, X_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.6f}")

# ==========================================================
# Step 5. 更新阈值并保存模型
# ==========================================================
model.eval()
with torch.no_grad():
    reconstructed = model(X_tensor)
    mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

threshold = float(np.percentile(mse, 97.5))
mean_mse = float(np.mean(mse))
print(f"📊 Computed 97.5th percentile threshold: {threshold:.6f}")
print(f"📈 Mean MSE after incremental training: {mean_mse:.6f}")

# torch.save(model.state_dict(), MODEL_FILE)
# joblib.dump(threshold, THRESH_FILE)

print("💾 Incremental model and threshold updated successfully.")

# ==========================================================
# Step 6. 清空反馈文件
# ==========================================================
# open(FEEDBACK_FILE, "w").close()
# print("🧹 Feedback file cleared. Incremental retraining complete.\n")

# ==========================================================
# ✅ End
# ==========================================================
print("✅ Incremental AutoEncoder fine-tuning completed.")
