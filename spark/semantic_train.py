"""
semantic_train.py
Train AutoEncoder from Parquet Embeddings
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


# autoencoder definition
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(
    parquet_dir: str = "/opt/spark/work-dir/data/semantic_data",
    model_dir: str = "/opt/spark/work-dir/models/prediction_model",
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 64,
):
    """use AutoEncoder to train embeddings from parquet files"""
    os.makedirs(model_dir, exist_ok=True)

    # load data
    print(f"📂 Loading parquet files from: {parquet_dir}")
    files = [
        f
        for f in glob.glob(os.path.join(parquet_dir, "*.parquet"))
        if os.path.getsize(f) > 0
    ]

    print(f"📁 Found {len(files)} parquet files.")
    if not files:
        raise RuntimeError("❌ No valid parquet files found!")

    dfs = []
    for f in tqdm(files, desc="📥 Reading parquet files"):
        try:
            df_part = pd.read_parquet(f)
            if "embedding" in df_part.columns:
                valid_df = df_part[["embedding"]].dropna()
                if len(valid_df) > 0:
                    dfs.append(valid_df)
        except Exception as e:
            print(f"⚠️ Skip {f}: {e}")

    if not dfs:
        raise RuntimeError("❌ No embeddings found in parquet data.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"✅ Loaded {len(df):,} embeddings from {len(dfs)} valid parquet files.")
    X = np.stack(df["embedding"].to_numpy())
    print(f"🧠 Embedding shape: {X.shape}")

    # scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print("💾 Saved scaler.pkl")

    # initialize model
    input_dim = X_scaled.shape[1]
    model = AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    print(f"🧩 AutoEncoder initialized: input_dim={input_dim}, hidden_dim={hidden_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train model
    print(f"🚀 Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.6f}")

    # calculate threshold
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

    threshold = float(np.percentile(mse, 97.5))
    print(f"📊 Computed 97.5th percentile threshold: {threshold:.6f}")
    mse = float(np.mean(mse))
    print(f"📈 Mean MSE after training: {mse:.6f}")
    joblib.dump(threshold, os.path.join(model_dir, "threshold.pkl"))

    # save model
    torch.save(model.state_dict(), os.path.join(model_dir, "autoencoder.pth"))
    print(f"💾 Saved model to {model_dir}/autoencoder.pth")

    print("\n✅ Training complete.")
    print(f"📁 Model directory: {model_dir}")


if __name__ == "__main__":
    train_autoencoder()
