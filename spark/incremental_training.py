# """
# incremental_training.py
# ========================
# Incremental (Feedback) Training Script for AI Monitoring System

# Main workflow:
# 1️⃣ Read feedback_samples.jsonl (incremental data)
# 2️⃣ Load saved model and scaler
# 3️⃣ Fine-tune the model with new samples (2–3 epochs)
# 4️⃣ Recalculate threshold and save updated model
# 5️⃣ Optionally clear feedback file
# """

import os
import torch
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from semantic_train import AutoEncoder

# ============================
# 路径配置（可根据部署路径调整）
# ============================
BASE_DIR = "/opt/spark/work-dir"
MODEL_DIR = f"{BASE_DIR}/models/semantic_autoencoder"
FEEDBACK_FILE = f"{BASE_DIR}/data/feedback_samples.jsonl"
SCALER_FILE = f"{MODEL_DIR}/scaler.pkl"
MODEL_FILE = f"{MODEL_DIR}/autoencoder.pth"
THRESH_FILE = f"{MODEL_DIR}/threshold.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# ============================
# Step 1. 加载增量数据
# ============================
if not os.path.exists(FEEDBACK_FILE) or os.path.getsize(FEEDBACK_FILE) == 0:
    print("⚠️ No new feedback samples found. Skip retraining.")
    exit(0)

df = pd.read_json(FEEDBACK_FILE, lines=True)
if df.empty:
    print("⚠️ Feedback file is empty. Skip retraining.")
    exit(0)

texts = df["semantic_text"].tolist()
print(f"📦 Loaded {len(texts)} new feedback samples.")

# ============================
# Step 2. 编码与标准化
# ============================
encoder = SentenceTransformer(MODEL_NAME)
scaler = joblib.load(SCALER_FILE)

X = encoder.encode(texts)
X_scaled = scaler.transform(X).astype(np.float32)
X_tensor = torch.tensor(X_scaled)

# ============================
# Step 3. 加载旧模型
# ============================
model = AutoEncoder(input_dim=X_tensor.shape[1], hidden_dim=128)
model.load_state_dict(torch.load(MODEL_FILE))
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# ============================
# Step 4. 微批增量训练
# ============================
EPOCHS = 3
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    recon = model(X_tensor)
    loss = criterion(recon, X_tensor)
    loss.backward()
    optimizer.step()
    print(f"🌀 Epoch {epoch+1}/{EPOCHS} | Loss = {loss.item():.6f}")

# ============================
# Step 5. 更新模型与阈值
# ============================
torch.save(model.state_dict(), MODEL_FILE)
print("✅ Incremental model updated successfully.")

recons = model(X_tensor).detach().numpy()
mse_values = np.mean((X_scaled - recons) ** 2, axis=1)
new_threshold = np.mean(mse_values) + 3 * np.std(mse_values)
joblib.dump(new_threshold, THRESH_FILE)
print(f"✅ New threshold saved: {new_threshold:.6f}")

# ============================
# Step 6. 清空反馈文件
# ============================
open(FEEDBACK_FILE, "w").close()
print("🧹 Feedback file cleared. Incremental retraining complete.")
