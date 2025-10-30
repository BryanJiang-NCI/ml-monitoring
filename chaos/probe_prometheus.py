import requests, time, sys, datetime

PROM_URL = "http://127.0.0.1:9090/api/v1/alerts"


def get_alerts():
    """获取当前 Prometheus 告警"""
    try:
        r = requests.get(PROM_URL, timeout=3)
        r.raise_for_status()
        data = r.json().get("data", {}).get("alerts", [])
        return data
    except Exception as e:
        print(f"PROMETHEUS ERROR {e}", flush=True)
        return []


def wait_for_detect(alert_type):
    """等待 Prometheus 触发指定告警"""
    while True:
        alerts = get_alerts()
        for a in alerts:
            if a["labels"].get("alertname") == alert_type and a["state"] == "firing":
                print(
                    f"PROMETHEUS_MONITOR prometheus_detection_detected {datetime.datetime.utcnow().isoformat()}Z",
                    flush=True,
                )
                return
        time.sleep(1)


def wait_for_recover(alert_type):
    """等待 Prometheus 告警恢复"""
    cooldown = 10
    check_interval = 2
    stable_count = 0

    while True:
        alerts = get_alerts()
        firing = [
            a
            for a in alerts
            if a["labels"].get("alertname") == alert_type and a["state"] == "firing"
        ]

        if not firing:
            stable_count += check_interval
            if stable_count >= cooldown:
                print(
                    f"PROMETHEUS_MONITOR prometheus_recovered {datetime.datetime.utcnow().isoformat()}Z",
                    flush=True,
                )
                return
        else:
            stable_count = 0
        time.sleep(check_interval)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python probe_prometheus.py [detect|recover] [AlertName]")
        sys.exit(1)

    mode = sys.argv[1]
    alert_type = sys.argv[2]

    if mode == "detect":
        wait_for_detect(alert_type)
    elif mode == "recover":
        wait_for_recover(alert_type)
    else:
        print("Usage: python probe_prometheus.py [detect|recover] [AlertName]")
