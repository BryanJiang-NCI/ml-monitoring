import os, json, requests, boto3, asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI

app = FastAPI(title="VectorFeeder")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def append_to_file(path, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


async def fetch_github_commits(owner, repo):
    headers = {
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
                "repository": repo_name,
            }
            append_to_file(os.path.join(DATA_DIR, "github_commits.jsonl"), data)

        page += 1


async def fetch_github_actions(owner, repo):
    headers = {
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
        data = {
            "type": "cloudtrail_event",
            "event_name": e["EventName"],
            "username": e.get("Username"),
            "event_source": e["EventSource"],
            "event_time": e["EventTime"].isoformat(),
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
            "http://nginx/",
            "http://nginx/db_read",
            "http://nginx/user_login",
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

                print(f"🌐 Simulated {url} → {status_code}")
                await asyncio.sleep(0.5)

        except Exception as e:
            print("❌ Simulated request error:", e)
            await asyncio.sleep(3)


async def periodic_fetch():
    while True:
        try:
            await fetch_github_commits("BryanJiang-NCI", "ml-monitoring")
            await fetch_github_actions("BryanJiang-NCI", "ml-monitoring")
            # await fetch_cloudtrail()
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
