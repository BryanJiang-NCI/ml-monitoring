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

# 启动时加载
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
    # 写入文件
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({k: v.isoformat() for k, v in cache.items()}, f)
    except Exception as e:
        print("⚠️ Cache save failed:", e)
    return False


def append_to_file(path, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


async def fetch_github_commits(owner, repo, per_page=10):
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    r = requests.get(url, headers=headers, params={"per_page": per_page}, timeout=10)
    r.raise_for_status()
    for c in r.json():
        sha = c["sha"]
        if already_seen(f"commit:{sha}"):
            continue
        data = {
            "type": "github_commit",
            "sha": sha,
            "author": c["commit"]["author"]["name"],
            "date": c["commit"]["author"]["date"],
            "message": c["commit"]["message"],
            "url": c["html_url"],
        }
        append_to_file(os.path.join(DATA_DIR, "github_commits.jsonl"), data)


async def fetch_github_actions(owner, repo, per_page=10):
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    r = requests.get(url, headers=headers, params={"per_page": per_page}, timeout=10)
    r.raise_for_status()
    for run in r.json().get("workflow_runs", []):
        rid = run["id"]
        if already_seen(f"action:{rid}"):
            continue
        data = {
            "type": "github_action",
            "id": rid,
            "name": run["name"],
            "status": run["status"],
            "conclusion": run["conclusion"],
            "created_at": run["created_at"],
            "url": run["html_url"],
        }
        append_to_file(os.path.join(DATA_DIR, "github_actions.jsonl"), data)


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
        eid = e["EventId"]
        if already_seen(f"cloudtrail:{eid}"):
            continue
        data = {
            "type": "cloudtrail_event",
            "event_id": eid,
            "event_name": e["EventName"],
            "username": e.get("Username"),
            "event_time": e["EventTime"].isoformat(),
        }
        append_to_file(os.path.join(DATA_DIR, "cloudtrail.jsonl"), data)


async def periodic_fetch():
    while True:
        try:
            await fetch_github_commits("BryanJiang-NCI", "ml-monitoring")
            await fetch_github_actions("BryanJiang-NCI", "ml-monitoring")
            await fetch_cloudtrail()
            print(f"[{datetime.utcnow().isoformat()}] ✅ Data fetched.")
        except Exception as e:
            print("❌ Fetch error:", e)
        await asyncio.sleep(600)  # 每10分钟循环一次


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_fetch())


@app.get("/")
def root():
    return {"message": "✅ VectorFeeder running", "cached": len(cache)}
