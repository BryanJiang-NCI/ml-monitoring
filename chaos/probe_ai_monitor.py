import time, sys, json, os
from datetime import datetime, timezone

LOG_PATH = "spark/data/anomaly.jsonl"

# ============ 参数处理 ============
mode = sys.argv[1] if len(sys.argv) > 1 else "detect"  # detect / recover
keywords = []

# 支持多个关键字，用 , 分隔
if len(sys.argv) > 2:
    keywords = [k.strip().lower() for k in sys.argv[2].split(",") if k.strip()]


def log(event):
    print(f"AI_MONITOR {event} {datetime.now(timezone.utc).isoformat()}", flush=True)


# ============ 基础函数 ============


def parse_timestamp(ts: str):
    """将字符串转为 aware datetime"""
    try:
        if not ts:
            return None
        ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def valid_event(event):
    """
    判断事件是否为有效异常：
    1. prediction 是 high_anomaly / low_anomaly
    2. 如果设置了关键字，则必须包含关键字
    """
    if not event:
        return False

    if event.get("prediction") not in ("high_anomaly", "low_anomaly"):
        return False

    # 如果没有设置过滤关键字 → 任何异常都算
    if not keywords:
        return True

    # 获取语义文本或信息字段
    text = (
        event.get("semantic_text") or event.get("message") or json.dumps(event)
    ).lower()

    # 含任意关键字即可
    return any(kw in text for kw in keywords)


def read_last_event():
    """读取最后一条异常事件（不保证一定有效）"""
    if not os.path.exists(LOG_PATH):
        return None
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()
            if not lines:
                return None
            return json.loads(lines[-1])
    except Exception:
        return None


def get_last_valid_anomaly_time():
    """读取最后一条有效异常的时间"""
    if not os.path.exists(LOG_PATH):
        return None

    with open(LOG_PATH) as f:
        lines = f.readlines()

    for line in reversed(lines):
        try:
            data = json.loads(line)
            if valid_event(data):
                return parse_timestamp(data.get("timestamp"))
        except Exception:
            continue

    return None


# ============ 等待检测 ============
def wait_for_detection():
    """等待检测到符合关键字条件的异常"""
    while True:
        event = read_last_event()
        if valid_event(event):
            log("ai_detection_detected")
            return
        time.sleep(1)


# ============ 等待恢复 ============
def wait_for_recovery():
    """检测到异常后，若 cooldown 秒内无关键字相关异常，则认为恢复"""
    cooldown = 10  # 连续10秒无异常
    check_interval = 1
    grace_period = 5  # 第一次检测到异常后等待5秒再开始判断恢复

    # 等待首次有效异常
    while not get_last_valid_anomaly_time():
        time.sleep(check_interval)

    last_anomaly_time = get_last_valid_anomaly_time()
    first_seen_time = datetime.now(timezone.utc)

    while True:
        time.sleep(check_interval)
        new_time = get_last_valid_anomaly_time()

        if new_time and new_time > last_anomaly_time:
            last_anomaly_time = new_time

        now = datetime.now(timezone.utc)
        elapsed = (now - last_anomaly_time).total_seconds()

        # 刚检测到异常，等待 grace_period 再开始判断恢复
        if (now - first_seen_time).total_seconds() < grace_period:
            continue

        if elapsed > cooldown:
            log("ai_recovered")
            return


# ============ 主逻辑 ============
if mode == "detect":
    wait_for_detection()
elif mode == "recover":
    wait_for_recovery()
else:
    print("Usage: python probe_ai_monitor.py [detect|recover] [keywords]")
