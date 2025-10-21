import sys, json
from datetime import datetime

if len(sys.argv) < 3:
    print("Usage: python calc_metrics.py <log_path> <out_path>")
    sys.exit(1)

log_path, out_path = sys.argv[1], sys.argv[2]
events = {}


def parse_time(s):
    """统一时间格式"""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


with open(log_path) as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue

        # inject_start 特殊处理
        if parts[0] == "inject_start" and len(parts) >= 2:
            t = parse_time(parts[-1])
            if t:
                events["inject_start"] = t
            continue

        # 通用格式：System + Event + Timestamp
        if len(parts) >= 3:
            system, event, time_str = parts[0], parts[1], parts[-1]
            t = parse_time(time_str)
            if t:
                events[f"{system}_{event}"] = t

if "inject_start" not in events:
    print("❌ No inject_start found in log.")
    sys.exit(1)

t0 = events["inject_start"]


def find_event(system, keywords):
    """模糊匹配系统相关事件"""
    for key, val in events.items():
        if key.startswith(system) and any(k in key for k in keywords):
            return val
    return None


def calc(system):
    """计算系统的 MTTD / MTTR / MTTResolve"""
    detect = find_event(system, ["detect", "detection"])
    recover = find_event(system, ["recover", "recovered"])
    if not (detect and recover):
        return None
    return {
        "MTTD": round((detect - t0).total_seconds(), 3),
        "MTTR": round((recover - detect).total_seconds(), 3),
        "MTTResolve": round((recover - t0).total_seconds(), 3),
    }


report = {
    "Prometheus": calc("PROMETHEUS"),
    "AI_Monitor": calc("AI_MONITOR"),
}

with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print("✅ Benchmark complete:")
print(json.dumps(report, indent=2))
