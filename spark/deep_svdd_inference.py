"""
Deep SVDD Inference Script
==========================
✅ 从训练好的模型加载参数
✅ 对单条日志文本计算 embedding → distance score
✅ 高于阈值则判定为异常
==========================
"""

import os
import json
import torch
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from deep_svdd_train import DeepSVDD

# -----------------------
# 路径定义
# -----------------------
BASE_DIR = "/opt/spark/work-dir"
MODEL_DIR = os.path.join(BASE_DIR, "models/deep_svdd_model")
MODEL_FILE = os.path.join(MODEL_DIR, "deep_svdd.pth")
CENTER_FILE = os.path.join(MODEL_DIR, "center.pt")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
THRESH_FILE = os.path.join(MODEL_DIR, "threshold.pkl")

# -----------------------
# 加载模型与参数
# -----------------------
encoder = SentenceTransformer("all-MiniLM-L12-v2")
scaler = joblib.load(SCALER_FILE)
threshold = float(joblib.load(THRESH_FILE))

model = DeepSVDD(input_dim=384)
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()
c = torch.load(CENTER_FILE)
print(f"✅ Model & Center loaded, threshold={threshold:.6f}")


# -----------------------
# 推理函数
# -----------------------
def infer_log(text: str):
    emb = encoder.encode(text)
    emb_scaled = scaler.transform([emb])
    x = torch.tensor(emb_scaled, dtype=torch.float32)
    with torch.no_grad():
        z = model(x)
        dist = torch.mean((z - c) ** 2).item()
    label = "abnormal" if dist > threshold else "normal"
    result = {
        "semantic_text": text,
        "distance": round(dist, 6),
        "threshold": round(threshold, 6),
        "prediction": label,
    }
    print(json.dumps(result, indent=2))
    return result


# -----------------------
# 示例测试
# -----------------------
if __name__ == "__main__":
    test_logs = [
        "status 200 ok response",
        "status 500 internal server error",
        "CPU usage 95% warning",
        "User authentication succeeded",
    ]
    for log in test_logs:
        infer_log(log)
