import requests
import time
import sys
import datetime
import json

PROM_URL = "http://127.0.0.1:9090/api/v1/alerts"
ALERT_NAME = "HighErrorRate"
mode = sys.argv[1] if len(sys.argv) > 1 else "detect"  # detect or recover


def log(event):
    """统一日志输出"""
    print(f"PROMETHEUS {event} {datetime.datetime.utcnow().isoformat()}Z", flush=True)


def get_alerts():
    """获取当前 Prometheus firing alerts"""
    try:
        r = requests.get(PROM_URL, timeout=3)
        data = r.json().get("data", {}).get("alerts", [])
        return [a for a in data if a.get("labels", {}).get("alertname") == ALERT_NAME]
    except Exception as e:
        print(f"PROMETHEUS ERROR fetching alerts: {e}", flush=True)
        return []


def wait_for_detect():
    """等待告警进入 firing 状态"""
    while True:
        alerts = get_alerts()
        for a in alerts:
            if a.get("state") == "firing":
                log("prometheus_detection_detected")
                return
        time.sleep(1)


def wait_for_recover():
    """等待恢复（连续多次无 firing 状态）"""
    consecutive_no_firing = 0
    required_clear_count = 5  # 连续5次检查无firing（约10秒）
    interval = 2  # 检查间隔2s

    while True:
        alerts = get_alerts()
        has_firing = any(a.get("state") == "firing" for a in alerts)

        if not has_firing:
            consecutive_no_firing += 1
        else:
            consecutive_no_firing = 0

        if consecutive_no_firing >= required_clear_count:
            log("prometheus_recovered")
            return

        time.sleep(interval)


if mode == "detect":
    wait_for_detect()
elif mode == "recover":
    wait_for_recover()
else:
    print("Usage: python probe_prometheus.py [detect|recover]")
