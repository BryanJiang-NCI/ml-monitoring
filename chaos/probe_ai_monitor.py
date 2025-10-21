import time, sys, json, datetime, os

LOG_PATH = "spark/data/anomaly.log"
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


def get_last_anomaly_time():
    """返回最后一条异常的时间戳"""
    if not os.path.exists(LOG_PATH):
        return None
    with open(LOG_PATH) as f:
        lines = f.readlines()
        for line in reversed(lines):
            try:
                data = json.loads(line)
                if data.get("prediction") in ("high_anomaly", "low_anomaly"):
                    ts = data.get("timestamp")
                    if ts:
                        return datetime.datetime.fromisoformat(ts)
            except Exception:
                continue
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
    """等待恢复：检测到异常后，若 cooldown 秒内无新异常则认为恢复"""
    cooldown = 10  # 10秒无新异常即视为恢复
    check_interval = 1
    grace_period = 5  # 检测后至少等待5秒再开始判断恢复

    # 等待至少检测到一次异常
    while not get_last_anomaly_time():
        time.sleep(check_interval)

    last_anomaly_time = get_last_anomaly_time()
    last_check_time = datetime.datetime.utcnow()

    while True:
        time.sleep(check_interval)
        new_time = get_last_anomaly_time()

        # 检测到新异常则更新最后时间
        if new_time and new_time > last_anomaly_time:
            last_anomaly_time = new_time

        # 当前时间
        now = datetime.datetime.utcnow()
        elapsed = (now - last_anomaly_time).total_seconds()

        # 若异常刚检测完，先等待 grace_period 再进入判断
        if (now - last_check_time).total_seconds() < grace_period:
            continue

        # 连续 cooldown 秒无新异常则恢复
        if elapsed > cooldown:
            log("ai_recovered")
            return


if mode == "detect":
    wait_for_detection()
elif mode == "recover":
    wait_for_recovery()
else:
    print("Usage: python probe_ai_monitor.py [detect|recover]")
