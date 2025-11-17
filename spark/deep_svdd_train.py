"""
Deep SVDD Training Script
=========================
✅ 从 Parquet 向量文件加载 embedding 数据
✅ 仅使用正常样本训练（假设数据集大部分为正常）
✅ 训练一个特征提取网络并计算中心点 c
✅ 保存模型与阈值文件（threshold.pkl, model.pth, center.pkl）
=========================
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


# -----------------------
# 模型定义
# -----------------------
class DeepSVDD(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------
# 训练主逻辑
# -----------------------
def train_deep_svdd(
    parquet_dir="/opt/spark/work-dir/data/semantic_vectors",
    model_dir="/opt/spark/work-dir/models/deep_svdd_model",
    epochs=20,
    lr=1e-3,
    batch_size=128,
):
    os.makedirs(model_dir, exist_ok=True)

    # Step 1: 读取所有 parquet 向量文件
    files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    dfs = []
    for f in tqdm(files, desc="📥 Reading parquet files"):
        try:
            df = pd.read_parquet(f)
            if "embedding" in df.columns:
                dfs.append(df[["embedding"]].dropna())
        except Exception as e:
            print("⚠️ Skip:", f, e)

    if not dfs:
        raise RuntimeError("❌ No embeddings found in parquet data.")

    df = pd.concat(dfs, ignore_index=True)
    X = np.stack(df["embedding"].to_numpy())
    print(f"✅ Loaded {len(X)} samples with shape {X.shape}")

    # Step 2: 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print("💾 Saved scaler.pkl")

    # Step 3: 模型初始化
    input_dim = X_scaled.shape[1]
    model = DeepSVDD(input_dim=input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 4: 计算初始中心点 c
    model.eval()
    with torch.no_grad():
        c = torch.mean(model(X_tensor), dim=0)
    torch.save(c, os.path.join(model_dir, "center.pt"))
    print("📍 Initial center computed and saved.")

    # Step 5: 训练
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for (batch,) in loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = torch.mean((outputs - c) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.6f}")

    # Step 6: 计算阈值
    model.eval()
    with torch.no_grad():
        dist = torch.mean((model(X_tensor) - c) ** 2, dim=1).cpu().numpy()
    threshold = np.percentile(dist, 97.5)
    mean_dist = np.mean(dist)
    print(f"📊 Computed 97.5th percentile threshold: {threshold:.6f}")
    print(f"📈 Mean distance: {mean_dist:.6f}")

    joblib.dump(threshold, os.path.join(model_dir, "threshold.pkl"))
    torch.save(model.state_dict(), os.path.join(model_dir, "deep_svdd.pth"))
    print(f"💾 Model saved to {model_dir}")
    print("✅ Training complete.")


if __name__ == "__main__":
    train_deep_svdd()
