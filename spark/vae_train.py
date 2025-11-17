"""
VAE Training Script for Semantic Log Embeddings
===============================================
✅ 训练 VAE 模型来学习正常日志的潜在分布
✅ 采用句向量（SentenceTransformer） + 标准化
✅ 输出模型、scaler、阈值（基于重构误差）
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
# 🧠 Training Logic
# ==========================================================
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.01 * kl_div  # 控制KL损失比例


def train_vae(
    parquet_dir="/opt/spark/work-dir/data/semantic_vectors",
    model_dir="/opt/spark/work-dir/models/vae_model",
    epochs=15,
    batch_size=128,
    lr=1e-3,
    latent_dim=32,
):
    os.makedirs(model_dir, exist_ok=True)

    # Step 1. Load parquet files
    files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    dfs = []
    for f in tqdm(files, desc="📥 Reading parquet files"):
        try:
            df_part = pd.read_parquet(f)
            if "embedding" in df_part.columns:
                dfs.append(df_part[["embedding"]].dropna())
        except Exception:
            continue

    df = pd.concat(dfs, ignore_index=True)
    X = np.stack(df["embedding"].to_numpy())
    print(f"✅ Loaded {len(X)} samples with shape {X.shape}")

    # Step 2. Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print("💾 Saved scaler.pkl")

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 3. Train VAE
    model = VAE(input_dim=X.shape[1], latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("🚀 Training VAE...")
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / len(loader):.6f}")

    # Step 4. Compute reconstruction error threshold
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(X_tensor)
        mse = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()
    # threshold = float(np.percentile(mse, 97.5))
    mse_array = np.array(mse)
    mean = mse_array.mean()
    std = mse_array.std()

    # 3 sigma rule
    threshold = mean + 3 * std

    print(f"📊 Mean MSE: {mean:.6f}, Std: {std:.6f}, Threshold: {threshold:.6f}")
    mean_mse = float(np.mean(mse))
    print(f"📊 Threshold: {threshold:.6f} | Mean MSE: {mean_mse:.6f}")

    torch.save(model.state_dict(), os.path.join(model_dir, "vae.pth"))
    joblib.dump(threshold, os.path.join(model_dir, "threshold.pkl"))
    print(f"💾 Model saved to {model_dir}")
    print("✅ Training complete.")


if __name__ == "__main__":
    train_vae()
