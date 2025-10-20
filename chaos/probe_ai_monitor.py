import time, sys, json, datetime

LOG_PATH = "/tmp/ai_monitor_events.jsonl"
mode = sys.argv[1]  # detect / recover


def log(event):
    print(f"AI_MONITOR {event} {datetime.datetime.utcnow().isoformat()}Z", flush=True)


def wait_for_event(keyword):
    while True:
        with open(LOG_PATH) as f:
            lines = f.readlines()
            if any(keyword in l for l in lines):
                log(keyword)
                return
        time.sleep(1)


if mode == "detect":
    wait_for_event("detect_start")
elif mode == "recover":
    wait_for_event("recover")
