import sys, json
from datetime import datetime

log_path, out_path = sys.argv[1], sys.argv[2]
with open(log_path) as f:
    lines = f.readlines()

events = {}
for l in lines:
    parts = l.strip().split()
    if len(parts) >= 3:
        system, event, time_str = parts[0], parts[1], parts[2]
        events[f"{system}_{event}"] = datetime.fromisoformat(
            time_str.replace("Z", "+00:00")
        )

t0 = events["inject_start"]


def calc(prefix):
    detect = events.get(f"{prefix}_firing") or events.get(f"{prefix}_detect_start")
    recover = events.get(f"{prefix}_inactive") or events.get(f"{prefix}_recover")
    if not (detect and recover):
        return None
    return {
        "MTTD": (detect - t0).total_seconds(),
        "MTTR": (recover - detect).total_seconds(),
        "MTTResolve": (recover - t0).total_seconds(),
    }


report = {"Prometheus": calc("PROMETHEUS"), "AI_Monitor": calc("AI_MONITOR")}

with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print("✅ Benchmark complete:", json.dumps(report, indent=2))
