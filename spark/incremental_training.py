"""
Incremental (Feedback) Training Script for AutoEncoder
======================================================
🔄 修改点（按你的需求）：
- 增量训练完成后，不覆盖 prediction_model
- 新建一个同级目录：feedback_model
- 将模型和 threshold 保存到 feedback_model
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
# 路径配置
# ==========================================================
BASE_DIR = "/opt/spark/work-dir"

MODEL_DIR = f"{BASE_DIR}/models/prediction_model"  # 原模型目录
FEEDBACK_MODEL_DIR = f"{BASE_DIR}/models/feedback_model"  # 新的反馈模型目录

FEEDBACK_FILE = f"{BASE_DIR}/data/feedback_samples.jsonl"
SCALER_FILE = f"{MODEL_DIR}/scaler.pkl"
MODEL_FILE = f"{MODEL_DIR}/autoencoder.pth"  # 原模型
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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# ==========================================================
# Step 4. 微批增量训练
# ==========================================================
EPOCHS = 10
print(f"🚀 Starting incremental fine-tuning for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    recon = model(X_tensor)
    loss = criterion(recon, X_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.6f}")

# ==========================================================
# Step 5. 更新阈值
# ==========================================================
model.eval()
with torch.no_grad():
    reconstructed = model(X_tensor)
    mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

threshold = float(np.percentile(mse, 97.5))
mean_mse = float(np.mean(mse))

print(f"📊 Computed 97.5th percentile threshold: {threshold:.6f}")
print(f"📈 Mean MSE after incremental training: {mean_mse:.6f}")

# ==========================================================
# Step 6. 保存到新的 feedback_model 目录
# ==========================================================
os.makedirs(FEEDBACK_MODEL_DIR, exist_ok=True)

FEEDBACK_MODEL_FILE = f"{FEEDBACK_MODEL_DIR}/autoencoder_feedback.pth"
FEEDBACK_THRESH_FILE = f"{FEEDBACK_MODEL_DIR}/threshold_feedback.pkl"

torch.save(model.state_dict(), FEEDBACK_MODEL_FILE)
joblib.dump(threshold, FEEDBACK_THRESH_FILE)

print(f"💾 Feedback model saved to: {FEEDBACK_MODEL_FILE}")
print(f"💾 Feedback threshold saved to: {FEEDBACK_THRESH_FILE}")

# ==========================================================
# Step 7. 不覆盖 prediction_model，也不清空 feedback 文件（按需打开）
# ==========================================================
# open(FEEDBACK_FILE, "w").close()
# print("🧹 Feedback file cleared.\n")

print("✅ Incremental AutoEncoder feedback fine-tuning completed.\n")
