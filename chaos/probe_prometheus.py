import requests, time, sys, datetime, os

PROM_URL = "http://127.0.0.1:9090/api/v1/alerts"
ALERT_NAME = "HighErrorRate"
mode = sys.argv[1]  # detect / recover
LOG_PATH = "chaos/result/benchmark.log"


def log(event):
    print(f"PROMETHEUS {event} {datetime.datetime.utcnow().isoformat()}Z", flush=True)


def get_alerts():
    try:
        r = requests.get(PROM_URL, timeout=3)
        data = r.json().get("data", {}).get("alerts", [])
        return data
    except Exception as e:
        print(f"PROMETHEUS ERROR {e}", flush=True)
        return []


def has_detected_before():
    """检查 benchmark.log 是否已有 detection 记录"""
    if not os.path.exists(LOG_PATH):
        return False
    with open(LOG_PATH) as f:
        for line in f:
            if "prometheus_detection_detected" in line:
                return True
    return False


def wait_for_detect():
    """等待 Prometheus 触发告警"""
    while True:
        alerts = get_alerts()
        for a in alerts:
            if a["labels"].get("alertname") == ALERT_NAME and a["state"] == "firing":
                log("prometheus_detection_detected")
                return
        time.sleep(1)


def wait_for_recover():
    """等待恢复：检测过异常后连续 cooldown 秒无告警则认为恢复"""
    cooldown = 10
    check_interval = 2

    while True:
        # 若尚未检测到异常则继续等待
        if not has_detected_before():
            time.sleep(check_interval)
            continue

        # 检查 Prometheus 当前告警
        alerts = get_alerts()
        firing = [
            a
            for a in alerts
            if a["labels"].get("alertname") == ALERT_NAME and a["state"] == "firing"
        ]

        if not firing:
            log("prometheus_recovered")
            return

        time.sleep(check_interval)


if mode == "detect":
    wait_for_detect()
elif mode == "recover":
    wait_for_recover()
else:
    print("Usage: python probe_prometheus.py [detect|recover]")
