import os, json, requests, boto3, asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI

app = FastAPI(title="VectorFeeder")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
CACHE_FILE = os.path.join(DATA_DIR, "seen_cache.json")

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        cache = {k: datetime.fromisoformat(v) for k, v in cache.items()}
        print(f"🧠 Cache loaded: {len(cache)} records")
    except Exception as e:
        print("⚠️ Failed to load cache:", e)
        cache = {}
else:
    cache = {}

CACHE_TTL = timedelta(days=1)


def already_seen(key: str):
    now = datetime.utcnow()
    for k, t in list(cache.items()):
        if now - t > CACHE_TTL:
            cache.pop(k)
    if key in cache:
        return True
    cache[key] = now
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({k: v.isoformat() for k, v in cache.items()}, f)
    except Exception as e:
        print("⚠️ Cache save failed:", e)
    return False


def append_to_file(path, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


async def fetch_github_commits(owner, repo):
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    base_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    page = 1

    while True:
        r = requests.get(
            base_url,
            headers=headers,
            params={"per_page": 100, "page": page},
            timeout=10,
        )
        r.raise_for_status()
        commits = r.json()

        if not commits:
            break  # no more commits → stop

        for c in commits:
            repo_name = f"{owner}/{repo}"
            data = {
                "type": "github_commit",
                "author": c["commit"]["author"]["name"],
                "email": c["commit"]["author"]["email"],
                "date": c["commit"]["author"]["date"],
                # "message": c["commit"]["message"],
                "repository": repo_name,
            }
            append_to_file(os.path.join(DATA_DIR, "github_commits.jsonl"), data)

        page += 1


async def fetch_github_actions(owner, repo):
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    base_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"

    page = 1
    while True:
        r = requests.get(
            base_url,
            headers=headers,
            params={"per_page": 100, "page": page},
            timeout=10,
        )
        r.raise_for_status()
        runs = r.json().get("workflow_runs", [])

        # no more records → break pagination
        if not runs:
            break

        for run in runs:
            rid = run["id"]
            data = {
                "type": "github_action",
                "name": run.get("name"),
                "event": run.get("event"),
                "pipeline_file": run.get("path"),
                "build_branch": run.get("head_branch"),
                "status": run.get("status"),
                "commit_message": run.get("display_title"),
                "actor": run["actor"]["login"] if run.get("actor") else None,
                "conclusion": run.get("conclusion"),
                "created_at": run.get("created_at"),
                "repository": f"{owner}/{repo}",  # 更可靠的值
            }

            append_to_file(os.path.join(DATA_DIR, "github_actions.jsonl"), data)

        page += 1


async def fetch_cloudtrail(max_results=10):
    client = boto3.client(
        "cloudtrail",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    res = client.lookup_events(MaxResults=max_results)
    print("CloudTrail events fetched:", len(res["Events"]))
    for e in res["Events"]:
        # if already_seen(f"cloudtrail:{eid}"):
        #     continue
        data = {
            "type": "cloudtrail_event",
            "event_name": e["EventName"],
            "username": e.get("Username"),
            "event_time": e["EventTime"].isoformat(),
            "context": e,
        }
        append_to_file(os.path.join(DATA_DIR, "cloudtrail.jsonl"), data)


async def fetch_fastapi_health():
    url = "http://fastapi-demo:8000"
    try:
        r = requests.get(url, timeout=2)
        status_code = r.status_code
        if 200 <= status_code < 400:
            status = "running"
            value = 1.0
            message = f"Service is healthy and responding with HTTP {status_code}"
        else:
            status = "http_error"
            value = 0.5
            message = f"Service responded with error HTTP {status_code}"

    except requests.exceptions.ConnectionError:
        status = "connection_failed"
        value = 0.0
        status_code = 0
        message = "Service unreachable — connection failed"
    except requests.exceptions.Timeout:
        status = "timeout"
        value = 0.0
        status_code = 408
        message = "Service timed out — no response within 2 seconds"
    except Exception as e:
        status = "critical_failure"
        value = 0.0
        status_code = 500
        message = f"Unexpected exception occurred: {str(e)}"

    data = {
        "type": "container_status",
        "name": "fastapi_health",
        "status": status,
        "status_code": status_code,
        "url": url,
        "value": value,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "message": message,
    }
    append_to_file(os.path.join(DATA_DIR, "fastapi_health.jsonl"), data)
    print(f"🩺 FastAPI health check: {status} (code={status_code})")


async def simulate_workload():
    import random

    urls = random.sample(
        [
            "http://fastapi-demo:8000/",
            "http://fastapi-demo:8000/db_read",
            "http://fastapi-demo:8000/user_login",
        ],
        k=3,
    )
    print("🚀 Real request simulation started.")
    while True:
        try:
            for url in urls:
                status_code = 0
                try:
                    r = requests.get(url, timeout=2)
                    status_code = r.status_code
                except requests.exceptions.RequestException:
                    status_code = 0  # connection failure or timeout

                # data = {
                #     "type": "simulated_request",
                #     "url": url,
                #     "status_code": status_code,
                #     "timestamp": datetime.utcnow().isoformat() + "Z",
                # }
                # append_to_file(os.path.join(DATA_DIR, "simulated_requests.jsonl"), data)
                print(f"🌐 Simulated {url} → {status_code}")
                await asyncio.sleep(1)

        except Exception as e:
            print("❌ Simulated request error:", e)
            await asyncio.sleep(3)


async def periodic_fetch():
    while True:
        try:
            await fetch_github_commits("BryanJiang-NCI", "ml-monitoring")
            await fetch_github_actions("BryanJiang-NCI", "ml-monitoring")
            await fetch_cloudtrail()
            print(f"[{datetime.utcnow().isoformat()}] ✅ Data fetched.")
        except Exception as e:
            print("❌ Fetch error:", e)
        await asyncio.sleep(60)


async def fetch_fastapi_periodically():
    while True:
        try:
            await fetch_fastapi_health()
        except Exception as e:
            print("❌ FastAPI health fetch error:", e)
        await asyncio.sleep(10)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_fetch())
    asyncio.create_task(fetch_fastapi_periodically())
    asyncio.create_task(simulate_workload())


@app.get("/")
def root():
    return {"message": "✅ VectorFeeder running", "cached": len(cache)}
