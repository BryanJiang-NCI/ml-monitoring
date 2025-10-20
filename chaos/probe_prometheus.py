import requests, time, sys, datetime, json

PROM_URL = "http://127.0.0.1:9090/api/v1/alerts"
ALERT_NAME = "HighErrorRate"
mode = sys.argv[1]  # detect or recover


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


def wait_for_detect():
    while True:
        alerts = get_alerts()
        for a in alerts:
            if a["labels"].get("alertname") == ALERT_NAME and a["state"] == "firing":
                log("detect_start")
                return
        time.sleep(1)


def wait_for_recover():
    consecutive_empty = 0
    while True:
        alerts = get_alerts()
        if not alerts:
            consecutive_empty += 1
        else:
            firing = [
                a
                for a in alerts
                if a["labels"].get("alertname") == ALERT_NAME and a["state"] == "firing"
            ]
            if not firing:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

        if consecutive_empty >= 3:  # 连续3次为空或无firing视为恢复
            log("recover")
            return
        time.sleep(2)


if mode == "detect":
    wait_for_detect()
elif mode == "recover":
    wait_for_recover()
