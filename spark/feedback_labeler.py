"""
feedback_labeler.py
=====================
用于人工确认 anomaly.jsonl 中的异常事件并写入 feedback_samples.jsonl。
"""

import os
import json
import argparse

# 路径配置
BASE_DIR = "./spark/data"
ANOMALY_LOG = f"{BASE_DIR}/anomaly.jsonl"
FEEDBACK_FILE = f"{BASE_DIR}/feedback_samples.jsonl"

# 命令行参数
parser = argparse.ArgumentParser(
    description="Label feedback samples for AI monitoring system."
)
parser.add_argument(
    "--timestamp", required=True, help="Timestamp of the anomaly entry."
)
parser.add_argument(
    "--label",
    required=True,
    choices=["true", "false"],
    help="Feedback label.",
)
parser.add_argument(
    "--root_cause",
    required=False,
    help="Optional root cause service name.",
)
args = parser.parse_args()

# 检查文件
if not os.path.exists(ANOMALY_LOG):
    print("⚠️ No anomaly.jsonl found.")
    exit(1)

found = None
with open(ANOMALY_LOG) as f:
    for line in f:
        try:
            entry = json.loads(line.strip())
            if entry.get("timestamp") == args.timestamp:
                found = entry
                break
        except json.JSONDecodeError:
            continue

if not found:
    print(f"❌ No matching entry found for timestamp {args.timestamp}")
    exit(1)

# 添加标注并写入反馈文件
found["feedback_label"] = args.label
found["root_cause"] = args.root_cause if args.root_cause else None
with open(FEEDBACK_FILE, "a") as f:
    f.write(json.dumps(found) + "\n")

print(f"✅ Feedback labeled: {args.timestamp} → {args.label}")
print(f"✅ Saved to {FEEDBACK_FILE}")
