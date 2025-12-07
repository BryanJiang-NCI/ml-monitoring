"""
feedback_labeler.py
Label feedback samples for AI monitoring system.
"""

import os
import json
import argparse

# base directory and file paths
BASE_DIR = "./spark/data"
ANOMALY_LOG = f"{BASE_DIR}/anomaly.jsonl"
FEEDBACK_FILE = f"{BASE_DIR}/feedback_samples.jsonl"

# command-line argument parsing
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

# check if anomaly log exists
if not os.path.exists(ANOMALY_LOG):
    print("⚠️ No anomaly.jsonl found.")
    exit(1)

# search for the entry with the given timestamp
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

# add feedback label and root cause
found["feedback_label"] = args.label
found["root_cause"] = args.root_cause if args.root_cause else None
with open(FEEDBACK_FILE, "a") as f:
    f.write(json.dumps(found) + "\n")

print(f"✅ Feedback labeled: {args.timestamp} → {args.label}")
print(f"✅ Saved to {FEEDBACK_FILE}")
