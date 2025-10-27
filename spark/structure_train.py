"""
Structured Data AutoEncoder Training
====================================
✅ 从 structured_data Parquet 读取结构化日志数据
✅ 对分类字段做 OneHot 编码
✅ 对数值字段做标准化
✅ 使用 AutoEncoder 进行无监督重构训练
✅ 输出模型文件 + 阈值
====================================
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

# ==========================================================
# 🧩 Step 0. 路径配置
# ==========================================================
DATA_PATH = "/opt/spark/work-dir/data/structured_data"
MODEL_DIR = "/opt/spark/work-dir/models/structured_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# 🧩 Step 1. Spark 初始化并读取 Parquet
# ==========================================================
spark = SparkSession.builder.appName("Structured_AutoEncoder_Train").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print(f"📦 Loading structured parquet data from {DATA_PATH} ...")
df = spark.read.parquet(DATA_PATH)
pdf = df.toPandas()
print(f"✅ Loaded {len(pdf)} records, {len(pdf.columns)} columns.")

# ==========================================================
# 🧩 Step 2. 字段选择与类型推断
# ==========================================================
# 只保留前几个关键字段（太多字段模型会太稀疏）
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

pdf = pdf[cols_keep].fillna("")

# 尝试推断哪些字段是数值列
numeric_cols = []
categorical_cols = []

for c in pdf.columns:
    try:
        # 尝试把百分比或数值列转 float
        if pdf[c].astype(str).str.contains("%").any():
            pdf[c] = pdf[c].astype(str).str.replace("%", "").astype(float)
            numeric_cols.append(c)
        elif pd.to_numeric(pdf[c], errors="coerce").notnull().mean() > 0.9:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0)
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    except Exception:
        categorical_cols.append(c)

print(f"🧩 Numeric columns: {numeric_cols}")
print(f"🧩 Categorical columns: {categorical_cols}")

# ==========================================================
# 🧩 Step 3. 特征转换 Pipeline
# ==========================================================
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# 拟合 + 转换
X_processed = preprocessor.fit_transform(pdf)
print(f"✅ Feature shape after preprocessing: {X_processed.shape}")

# 保存处理器
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

# ==========================================================
# 🧩 Step 4. 构建 AutoEncoder 模型
# ==========================================================
input_dim = X_processed.shape[1]


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = AutoEncoder(input_dim=input_dim, hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ==========================================================
# 🧩 Step 5. 划分数据集并训练
# ==========================================================
X_train, X_val = train_test_split(X_processed, test_size=0.2, random_state=42)

# ✅ 直接转换为 tensor（不需要 toarray）
X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
X_val_tensor = torch.tensor(np.array(X_val), dtype=torch.float32)

epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, X_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor)
        val_loss = criterion(val_output, X_val_tensor)

    print(
        f"Epoch [{epoch+1}/{epochs}] Train Loss={loss.item():.6f} Val Loss={val_loss.item():.6f}"
    )

# ==========================================================
# 🧩 Step 6. 阈值计算
# ==========================================================
model.eval()
with torch.no_grad():
    recon = model(X_train_tensor)
    errors = torch.mean((X_train_tensor - recon) ** 2, dim=1).numpy()
threshold = np.mean(errors) + 2 * np.std(errors)
print(f"✅ Threshold calculated: {threshold:.6f}")

# ==========================================================
# 🧩 Step 7. 模型保存
# ==========================================================
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "autoencoder.pth"))
joblib.dump(threshold, os.path.join(MODEL_DIR, "threshold.pkl"))
print(f"🎯 Model & threshold saved to {MODEL_DIR}")
