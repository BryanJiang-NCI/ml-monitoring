"""
evaluate_root.py
Evaluate and fine-tune root cause analysis model using labeled feedback samples.
"""

import json
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

# load model and cmdb dataset
BASE = "/opt/spark/work-dir"
model = SentenceTransformer(f"{BASE}/models/root_cause_model")
cmdb = pd.read_json(f"{BASE}/data/cmdb.jsonl", lines=True)

# construct service description map
cmdb["desc"] = cmdb.fillna("").apply(
    lambda r: f"{r['service_name']} {r['domain']} {r['system']} {' '.join(r.get('dependencies',[]))}",
    axis=1,
)
svc_map = dict(zip(cmdb["service_name"], cmdb["desc"]))

# filter feedback samples if label is true and root_cause in cmdb
samples = []
with open(f"{BASE}/data/feedback_samples.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get("feedback_label") == "true" and d.get("root_cause") in svc_map:
                samples.append(
                    {
                        "log": d["semantic_text"],
                        "truth": d["root_cause"],
                        "desc": svc_map[d["root_cause"]],
                    }
                )
        except:
            pass

print(f"🎯 Loaded {len(samples)} samples.")


# evaluate function
def get_scores(m, data):
    return [util.cos_sim(m.encode(d["log"]), m.encode(d["desc"])).item() for d in data]


# Phase 1: Before
scores_pre = get_scores(model, samples)

# Phase 2: Training
train_data = [
    InputExample(texts=[s["log"], s["desc"]]) for s in samples for _ in range(10)
]
model.fit(
    train_objectives=[
        (
            DataLoader(train_data, shuffle=True, batch_size=4),
            losses.MultipleNegativesRankingLoss(model),
        )
    ],
    epochs=10,
    warmup_steps=0,
    optimizer_params={"lr": 1e-4},
    show_progress_bar=True,
)

# Phase 3: After
scores_post = get_scores(model, samples)

# Results output
print(f"\n{'ID':<4} {'Service':<15} {'Before':<8} {'After':<8} {'Gain'}")
print("-" * 55)
avg_gain = 0
for i, (pre, post) in enumerate(zip(scores_pre, scores_post)):
    gain = (post - pre) / pre
    avg_gain += gain
    print(f"S{i+1:<3} {samples[i]['truth']:<15} {pre:.3f}    {post:.3f}    +{gain:.1%}")
print("-" * 55)
print(f"🏆 Avg Improvement: +{avg_gain/len(samples):.1%}")
