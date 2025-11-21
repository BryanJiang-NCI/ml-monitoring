import json
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

# 1. 配置与加载
BASE = "/opt/spark/work-dir"
model = SentenceTransformer(f"{BASE}/models/root_cause_model")
cmdb = pd.read_json(f"{BASE}/data/cmdb.jsonl", lines=True)

# 构建服务描述映射
cmdb["desc"] = cmdb.fillna("").apply(
    lambda r: f"{r['service_name']} {r['domain']} {r['system']} {' '.join(r.get('dependencies',[]))}",
    axis=1,
)
svc_map = dict(zip(cmdb["service_name"], cmdb["desc"]))

# 2. 加载数据 (直接读取 label=true 且包含 root_cause 的数据)
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


# 3. 评估函数
def get_scores(m, data):
    return [util.cos_sim(m.encode(d["log"]), m.encode(d["desc"])).item() for d in data]


# 4. 实验流程
# Phase 1: Before
scores_pre = get_scores(model, samples)

# Phase 2: Training (10x 重采样, lr=1e-4, epochs=12)
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

# 5. 结果输出
print(f"\n{'ID':<4} {'Service':<15} {'Before':<8} {'After':<8} {'Gain'}")
print("-" * 55)
avg_gain = 0
for i, (pre, post) in enumerate(zip(scores_pre, scores_post)):
    gain = (post - pre) / pre
    avg_gain += gain
    print(f"S{i+1:<3} {samples[i]['truth']:<15} {pre:.3f}    {post:.3f}    +{gain:.1%}")
print("-" * 55)
print(f"🏆 Avg Improvement: +{avg_gain/len(samples):.1%}")
