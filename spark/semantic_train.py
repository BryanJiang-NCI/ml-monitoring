"""
Train AutoEncoder from Parquet Embeddings
=========================================
✅ 从语义向量 parquet 文件中读取 embedding
✅ 训练 PyTorch AutoEncoder（重构异常检测模型）
✅ 保存 scaler、模型权重、阈值
=========================================
"""

import os
import glob
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==========================================================
# 🧩 Step 0. 配置路径
# ==========================================================
PARQUET_DIR = "./spark-data/semantic_vectors"
MODEL_DIR = "./spark-apps/models/autoencoder_tfidf_torch"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# 🧩 Step 1. 加载 Parquet 数据
# ==========================================================
print(f"📂 Loading parquet files from: {PARQUET_DIR}")
files = [
    f
    for f in glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
    if os.path.getsize(f) > 0
]

if not files:
    raise RuntimeError("❌ No valid parquet files found!")

dfs = []
for f in tqdm(files[:10], desc="📥 Loading parquet (limit 10 files for now)"):
    try:
        df_part = pd.read_parquet(f)
        if "embedding" in df_part.columns:
            dfs.append(df_part[["embedding"]].dropna())
    except Exception as e:
        print(f"⚠️ Skip {f}: {e}")

if not dfs:
    raise RuntimeError("❌ No embeddings found in parquet data.")

df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(df):,} embeddings.")

# 转为 numpy
X = np.stack(df["embedding"].to_numpy())
print(f"🧠 Embedding shape: {X.shape}")

# ==========================================================
# 🧩 Step 2. 标准化
# ==========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("💾 Saved scaler.pkl")


# ==========================================================
# 🧩 Step 3. 定义 AutoEncoder
# ==========================================================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


input_dim = X_scaled.shape[1]
model = AutoEncoder(input_dim=input_dim, hidden_dim=128)
print(f"🧩 AutoEncoder initialized: input_dim={input_dim}, hidden_dim=128")

# ==========================================================
# 🧩 Step 4. 训练
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
dataset = torch.utils.data.TensorDataset(X_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
print(f"🚀 Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for (batch,) in loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(loader):.6f}")

# ==========================================================
# 🧩 Step 5. 计算阈值（95分位）
# ==========================================================
model.eval()
with torch.no_grad():
    reconstructed = model(X_tensor)
    mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

threshold = float(np.percentile(mse, 95))
joblib.dump(threshold, os.path.join(MODEL_DIR, "threshold.pkl"))
print(f"📊 Computed 95th percentile threshold: {threshold:.6f}")

# ==========================================================
# 🧩 Step 6. 保存模型
# ==========================================================
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "autoencoder.pth"))
print(f"💾 Saved model to {MODEL_DIR}/autoencoder.pth")

print("\n✅ Training complete.")
print(f"📁 Model directory: {MODEL_DIR}")
print(f"🧩 Threshold (95% MSE): {threshold:.6f}")
