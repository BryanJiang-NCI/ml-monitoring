import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# === 三个实验日志路径 ===
LOG_FILES = [
    "chaos/result/benchmark_cpu.log",
    "chaos/result/benchmark_http.log",
    "chaos/result/benchmark_service.log",
]


# === 时间解析函数 ===
def parse_time(s):
    """统一时间格式"""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# === 解析单个日志，计算该实验的 MTTD / MTTR / MTTResolve ===
def parse_log(log_path):
    events = {}
    with open(log_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            # 故障注入开始时间
            if parts[0].lower() == "inject_start":
                events["inject_start"] = parse_time(parts[-1])

            # 系统事件
            elif len(parts) >= 3:
                system, event, ts = parts[0].upper(), parts[1].lower(), parts[-1]
                events[f"{system}_{event}"] = parse_time(ts)

    if "inject_start" not in events:
        print(f"⚠️ {log_path} 缺少 inject_start 事件，跳过")
        return None

    t0 = events["inject_start"]

    # 计算指定系统的各项指标
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
            print(f"⚠️ {log_path} 中 {system} 缺少检测或恢复事件")
            return None

        return {
            "MTTD": round((detect - t0).total_seconds(), 3),
            "MTTR": round((recover - detect).total_seconds(), 3),
            "MTTResolve": round((recover - t0).total_seconds(), 3),
        }

    return {
        "Prometheus": calc("PROMETHEUS"),
        "AI_Monitor": calc("AI_MONITOR"),
    }


# === 计算三个实验的平均值 ===
def average_results(results):
    avg = {"Prometheus": {}, "AI_Monitor": {}}
    systems = ["Prometheus", "AI_Monitor"]
    metrics = ["MTTD", "MTTR", "MTTResolve"]

    for sys_name in systems:
        valid = [r[sys_name] for r in results if r and r[sys_name]]
        for m in metrics:
            vals = [r[m] for r in valid if r and m in r]
            avg[sys_name][m] = round(sum(vals) / len(vals), 3) if vals else 0
    return avg


# === 主流程 ===
def main():
    all_results = []

    # 依次解析三个实验
    for log_file in LOG_FILES:
        if os.path.exists(log_file):
            print(f"📘 解析日志: {log_file}")
            result = parse_log(log_file)
            if result:
                all_results.append(result)
        else:
            print(f"⚠️ 未找到文件: {log_file}")

    if not all_results:
        print("❌ 没有可用日志文件，退出。")
        return

    # 计算平均结果
    final_report = average_results(all_results)

    # 保存为 CSV 文件
    df = pd.DataFrame(final_report).T
    csv_path = "chaos/result/benchmark_summary.csv"
    df.to_csv(csv_path, index_label="System")
    print(f"✅ 已生成 CSV 文件: {csv_path}")

    # === 生成对比图 ===
    plt.figure(figsize=(8, 5))
    ax = df.plot(kind="bar", figsize=(8, 5), width=0.7)
    plt.title("Benchmark Comparison: Prometheus vs AI_Monitor", fontsize=13)
    plt.ylabel("Seconds")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 在柱顶标注数值
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
    print("✅ 已生成图像文件: benchmark_summary.png")

    print("\n=== 平均结果 ===")
    print(df)


if __name__ == "__main__":
    main()
