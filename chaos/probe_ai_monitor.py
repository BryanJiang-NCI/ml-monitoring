import time, sys, json, datetime, os

LOG_PATH = "/Users/bryan/git/ml-monitoring/spark/data/anomaly.log"
mode = sys.argv[1] if len(sys.argv) > 1 else "detect"  # detect / recover


def log(event):
    print(f"AI_MONITOR {event} {datetime.datetime.utcnow().isoformat()}Z", flush=True)


def read_last_event():
    """读取日志中最后一条异常事件"""
    if not os.path.exists(LOG_PATH):
        return None
    with open(LOG_PATH) as f:
        lines = f.readlines()
        if not lines:
            return None
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return None


def wait_for_detection():
    """等待 AI 检测到异常"""
    while True:
        event = read_last_event()
        if event and event.get("prediction") in ("high_anomaly", "low_anomaly"):
            log("ai_detection_detected")
            return
        time.sleep(1)


def wait_for_recovery():
    """等待恢复：如果 anomaly.log 在 cooldown 秒内未更新，则认为系统恢复"""

    cooldown = 10  # 连续 10 秒无新异常即视为恢复
    last_update = time.time()

    if os.path.exists(LOG_PATH):
        last_update = os.path.getmtime(LOG_PATH)

    while True:
        # 获取当前日志文件的最后更新时间
        if os.path.exists(LOG_PATH):
            mtime = os.path.getmtime(LOG_PATH)
            if mtime > last_update:
                last_update = mtime

        now = time.time()
        # ✅ 若 cooldown 秒内未有更新，判定为恢复
        if now - last_update > cooldown:
            log("ai_recovered")
            return

        time.sleep(1)


if mode == "detect":
    wait_for_detection()
elif mode == "recover":
    wait_for_recovery()
else:
    print("Usage: python probe_ai_monitor.py [detect|recover]")
