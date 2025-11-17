"""
Isolation Forest Training Script
===============================
✅ 从 Parquet 文件读取嵌入向量
✅ 训练 IsolationForest 异常检测模型
✅ 计算异常分数阈值（97.5分位）
✅ 保存模型与 scaler
===============================
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

# ======================
# 路径配置
# ======================
PARQUET_DIR = "/opt/spark/work-dir/data/semantic_vectors"
MODEL_DIR = "/opt/spark/work-dir/models/iforest_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================
# Step 1. 加载所有 Parquet 文件
# ======================
print(f"📂 Loading parquet files from: {PARQUET_DIR}")
files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
dfs = []

for f in tqdm(files, desc="📥 Reading parquet files"):
    try:
        df = pd.read_parquet(f)
        if "embedding" in df.columns:
            valid_df = df[["embedding"]].dropna()
            dfs.append(valid_df)
    except Exception as e:
        print(f"⚠️ Skip {f}: {e}")

df = pd.concat(dfs, ignore_index=True)
X = np.stack(df["embedding"].to_numpy())
print(f"✅ Loaded {len(X)} samples with shape {X.shape}")

# ======================
# Step 2. 标准化
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("💾 Saved scaler.pkl")

# ======================
# Step 3. 训练 Isolation Forest
# ======================
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,  # 假设 2% 异常
    random_state=42,
    max_samples="auto",
    n_jobs=-1,
)
print("🚀 Training Isolation Forest...")
model.fit(X_scaled)

# ======================
# Step 4. 计算异常分数与阈值
# ======================
scores = -model.score_samples(X_scaled)
threshold = np.percentile(scores, 97.5)
print(f"📊 Computed 97.5th percentile threshold: {threshold:.6f}")
print(f"📈 Mean score: {np.mean(scores):.6f}")

# ======================
# Step 5. 保存模型与阈值
# ======================
joblib.dump(model, os.path.join(MODEL_DIR, "iforest.pkl"))
joblib.dump(threshold, os.path.join(MODEL_DIR, "threshold.pkl"))
print("💾 Model and threshold saved successfully.")
print(f"📁 Model directory: {MODEL_DIR}")
