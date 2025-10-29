"""
Structured Data AutoEncoder Training (Robust Dense Version)
===========================================================
- 读取 /data/structured_data
- 数值列标准化、分类列 OneHot（强制 dense，避免 numpy.object_ 问题）
- 训练 AutoEncoder，无监督阈值
- 保存 preprocessor / model / threshold
===========================================================
"""

import os
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "/opt/spark/work-dir/data/structured_data"
MODEL_DIR = "/opt/spark/work-dir/models/structured_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Spark Read
# -----------------------------
spark = SparkSession.builder.appName("Structured_AutoEncoder_Train").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print(f"📦 Loading structured parquet data from {DATA_PATH} ...")
df = spark.read.parquet(DATA_PATH)
pdf = df.toPandas()
print(f"✅ Loaded {len(pdf)} records, {len(pdf.columns)} columns.")

if len(pdf) == 0:
    raise RuntimeError("No data found in structured_data. Please ingest logs first.")

# -----------------------------
# Keep a compact feature set
# -----------------------------
cols_keep = [
    "source_type",
    "commit_author",
    "commit_message",
    "action_status",
    "action_conclusion",
    "event_name",
    "username",
    "cpu_perc",
    "mem_perc",
    "container_name",
    "log_level",
    "log_message",
    "response_status",
    "error_level",
]

# 只保留存在的列（避免某些源还没进来时报错）
cols_keep = [c for c in cols_keep if c in pdf.columns]
pdf = pdf[cols_keep].fillna("")

# -----------------------------
# Numeric / Categorical split
# -----------------------------
numeric_cols, categorical_cols = [], []
pdf_work = pdf.copy()

for c in pdf_work.columns:
    try:
        s = pdf_work[c].astype(str)
        if s.str.contains("%").any():
            pdf_work[c] = s.str.replace("%", "", regex=False)
        # 尝试转数值
        as_num = pd.to_numeric(pdf_work[c], errors="coerce")
        ratio = as_num.notna().mean()
        if ratio > 0.9:
            pdf_work[c] = as_num.fillna(0.0)
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    except Exception:
        categorical_cols.append(c)

print(f"🧩 Numeric columns: {numeric_cols}")
print(f"🧩 Categorical columns: {categorical_cols}")

# -----------------------------
# Preprocessor (force dense)
# 兼容 sklearn>=1.2 使用 sparse_output=False
# 兼容旧版使用 sparse=False
# -----------------------------
onehot_kwargs = {}
try:
    # sklearn >= 1.2
    OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    onehot_kwargs = {"sparse_output": False}
except TypeError:
    # sklearn < 1.2
    onehot_kwargs = {"sparse": False}

numeric_transformer = (
    Pipeline(steps=[("scaler", StandardScaler())]) if len(numeric_cols) > 0 else "drop"
)
categorical_transformer = (
    Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", **onehot_kwargs))]
    )
    if len(categorical_cols) > 0
    else "drop"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop",
)

X_processed = preprocessor.fit_transform(pdf_work)

# 统一为 float32 的 dense ndarray
if not isinstance(X_processed, np.ndarray):
    # Just in case（大多数情况下 OneHot 已经 dense 了）
    try:
        X_processed = X_processed.toarray()
    except AttributeError:
        X_processed = np.asarray(X_processed)

X_processed = X_processed.astype(np.float32, copy=False)
print(
    f"✅ Feature shape after preprocessing: {X_processed.shape}, dtype={X_processed.dtype}"
)

# 保存 preprocessor
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

# -----------------------------
# AutoEncoder
# -----------------------------
input_dim = X_processed.shape[1]
if input_dim == 0:
    raise RuntimeError(
        "No features after preprocessing. Check your columns and encoders."
    )


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.0)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = AutoEncoder(input_dim=input_dim, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# -----------------------------
# Train / Val split
# -----------------------------
X_train, X_val = train_test_split(X_processed, test_size=0.2, random_state=42)

# 直接转 Tensor（float32）
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_tensor)
    loss = criterion(out, X_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val_tensor)
        val_loss = criterion(val_out, X_val_tensor)

    print(
        f"Epoch [{epoch+1}/{epochs}] Train Loss={loss.item():.6f}  Val Loss={val_loss.item():.6f}"
    )

# -----------------------------
# Threshold
# -----------------------------
model.eval()
with torch.no_grad():
    recon = model(X_train_tensor)
    errors = torch.mean((X_train_tensor - recon) ** 2, dim=1).cpu().numpy()

threshold = float(np.mean(errors) + 2 * np.std(errors))
print(f"✅ Threshold calculated: {threshold:.6f}")

# -----------------------------
# Save
# -----------------------------
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "autoencoder.pth"))
joblib.dump(threshold, os.path.join(MODEL_DIR, "threshold.pkl"))
print(f"🎯 Saved preprocessor / model / threshold to {MODEL_DIR}")
