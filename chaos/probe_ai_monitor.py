import time, sys, json, os
from datetime import datetime, timezone

LOG_PATH = "spark/data/anomaly.jsonl"

# detect / recover
mode = sys.argv[1] if len(sys.argv) > 1 else "detect"
keywords = []

if len(sys.argv) > 2:
    keywords = [k.strip().lower() for k in sys.argv[2].split(",") if k.strip()]


def log(event):
    print(f"AI_MONITOR {event} {datetime.now(timezone.utc).isoformat()}", flush=True)


def parse_timestamp(ts: str):
    """string convert to aware datetime"""
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
    check if the event is a valid anomaly event
    1. prediction is high_anomaly / low_anomaly
    """
    if not event:
        return False

    if event.get("prediction") not in ("high_anomaly", "low_anomaly"):
        return False

    # keyword matching
    if not keywords:
        return True

    # get text for keyword matching
    text = (
        event.get("semantic_text") or event.get("message") or json.dumps(event)
    ).lower()

    return any(kw in text for kw in keywords)


def read_last_event():
    """read the last event from the log file"""
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
    """read the timestamp of the last valid anomaly event"""
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


def wait_for_detection():
    """wait for detection (first valid anomaly)"""
    while True:
        event = read_last_event()
        if valid_event(event):
            log("ai_detection_detected")
            return
        time.sleep(1)


def wait_for_recovery():
    """wait for recovery (no valid anomalies for a cooldown period)"""
    cooldown = 10
    check_interval = 1
    grace_period = 5

    # wait until the first anomaly is detected
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

        # still in grace period
        if (now - first_seen_time).total_seconds() < grace_period:
            continue

        if elapsed > cooldown:
            log("ai_recovered")
            return


if mode == "detect":
    wait_for_detection()
elif mode == "recover":
    wait_for_recovery()
else:
    print("Usage: python probe_ai_monitor.py [detect|recover] [keywords]")
