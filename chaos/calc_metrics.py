import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# experiment log files
LOG_FILES = [
    "chaos/result/benchmark_cpu.log",
    "chaos/result/benchmark_http.log",
    "chaos/result/benchmark_service.log",
]


def parse_time(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def parse_log(log_path):
    """resolve log and calculate metrics"""
    events = {}
    with open(log_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            # get inject_start time
            if parts[0].lower() == "inject_start":
                events["inject_start"] = parse_time(parts[-1])

            # get other events
            elif len(parts) >= 3:
                system, event, ts = parts[0].upper(), parts[1].lower(), parts[-1]
                events[f"{system}_{event}"] = parse_time(ts)

    if "inject_start" not in events:
        print(f"⚠️ {log_path} there is no inject_start event")
        return None

    t0 = events["inject_start"]

    # calculate metrics for the pass system
    def calc(system):
        detect = next(
            (v for k, v in events.items() if k.startswith(system) and "detect" in k),
            None,
        )
        recover = next(
            (v for k, v in events.items() if k.startswith(system) and "recover" in k),
            None,
        )

        if not (detect and recover):
            print(f"⚠️ {log_path} 中 {system} lacks detect or recover events")
            return None

        return {
            "MTTD": round((detect - t0).total_seconds(), 3),
            "MTTR": round((recover - detect).total_seconds(), 3),
            "MTTResolve": round((recover - t0).total_seconds(), 3),
        }

    return {
        "Traditional Monitoring": calc("PROMETHEUS"),
        "Machine Learning-based Monitoring": calc("AI_MONITOR"),
    }


def average_results(results):
    """calculate average metrics across experiments"""
    avg = {"Traditional Monitoring": {}, "Machine Learning-based Monitoring": {}}
    systems = ["Traditional Monitoring", "Machine Learning-based Monitoring"]
    metrics = ["MTTD", "MTTR", "MTTResolve"]

    for sys_name in systems:
        valid = [r[sys_name] for r in results if r and r[sys_name]]
        for m in metrics:
            vals = [r[m] for r in valid if r and m in r]
            avg[sys_name][m] = round(sum(vals) / len(vals), 3) if vals else 0
    return avg


def main():
    all_results = []

    for log_file in LOG_FILES:
        if os.path.exists(log_file):
            print(f"📘 parse log: {log_file}")
            result = parse_log(log_file)
            if result:
                all_results.append(result)
        else:
            print(f"⚠️ file not found: {log_file}")

    if not all_results:
        print("❌ no valid results found.")
        return

    final_report = average_results(all_results)

    df = pd.DataFrame(final_report).T
    csv_path = "chaos/result/benchmark_summary.csv"
    df.to_csv(csv_path, index_label="System")
    print(f"✅ CSV file created: {csv_path}")

    plt.figure(figsize=(8, 5))
    ax = df.plot(kind="bar", figsize=(8, 5), width=0.7)
    plt.title(
        "Monitoring Performance Comparison (Traditional vs Machine Learning-based)",
        fontsize=13,
    )
    plt.ylabel("Seconds")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    plt.tight_layout()
    plt.savefig("chaos/result/benchmark_summary.png", dpi=200)
    print("✅ result png file: benchmark_summary.png")

    print("\n=== Average results ===")
    print(df)


if __name__ == "__main__":
    main()
