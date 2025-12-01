"""
root_cause_train.py
root cause analysis model training using weakly supervised feedback samples.
"""

import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

BASE_DIR = "/opt/spark/work-dir"
MODEL_SAVE_DIR = f"{BASE_DIR}/models/root_cause_model"
CMDB_FILE = f"{BASE_DIR}/data/cmdb.jsonl"
FEEDBACK_FILE = f"{BASE_DIR}/data/feedback_samples.jsonl"
MODEL_NAME = "all-MiniLM-L12-v2"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# cmdb loading
cmdb = pd.read_json(CMDB_FILE, lines=True)

# entity text splicing
cmdb["entity_text"] = cmdb.apply(
    lambda r: f"{r.get('service_name','')} {r.get('domain','')} {r.get('system','')} {' '.join(r.get('dependencies',[]))}",
    axis=1,
)
svc_map = dict(zip(cmdb["service_name"], cmdb["entity_text"]))
print(f"📦 Loaded {len(cmdb)} services.")


def extract_service(text, svc_names):
    """find strict service name in the text using multiple patterns"""
    if m := re.search(r"http://([^:/]+)", text):
        return m.group(1)
    matched = [s for s in svc_names if s in text]
    if matched:
        return matched[0]
    if m := re.search(r"name=([a-zA-Z0-9_\-]+)", text):
        return m.group(1).split("_")[0]
    return None


# train samples preparation
feedback = pd.read_json(FEEDBACK_FILE, lines=True)
feedback = feedback[feedback["feedback_label"] == "true"]
print(f"📦 Loaded {len(feedback)} confirmed anomalies.")

train_samples = [
    InputExample(texts=[row["semantic_text"], svc_map[pos_svc]])
    for _, row in feedback.iterrows()
    if (pos_svc := extract_service(row["semantic_text"], svc_map.keys()))
    and pos_svc in svc_map
]

print(f"🎯 Training pairs: {len(train_samples)}")

semantic_encoder = SentenceTransformer(MODEL_NAME)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)
train_loss = losses.MultipleNegativesRankingLoss(semantic_encoder)

print("🚀 Training semantic-text encoder for RCA...")

# training settings
semantic_encoder.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=50,
    show_progress_bar=True,
)

semantic_encoder.save(MODEL_SAVE_DIR)

print(f"🏁 Stable RCA model saved to {MODEL_SAVE_DIR}")
