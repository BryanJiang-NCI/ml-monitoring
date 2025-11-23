import json
import random
import datetime

# ==========================================================
# 1. Your real services (must be included exactly as-is)
# ==========================================================
real_services = [
    {
        "service_name": "fastapi-demo",
        "domain": "fastapi-demo",
        "owner": "team-fastapi",
        "language": "Python",
        "config_path": "/etc/fastapi-demo/config.yaml",
        "system": "demo-platform",
        "deployment_env": "docker",
        "dependencies": ["uvicorn", "pydantic"],
        "last_updated": "2025-10-20T06:31:05.306124549Z",
    },
    {
        "service_name": "nginx",
        "domain": "nginx",
        "owner": "team-nginx",
        "language": "C",
        "config_path": "/etc/nginx/nginx.conf",
        "system": "web-server",
        "deployment_env": "docker",
        "dependencies": ["libc"],
        "last_updated": "2025-10-18T04:20:15.123456789Z",
    },
]

# ==========================================================
# 2. Generate simulated services based on your environment
# ==========================================================
languages = ["Python", "Go", "Java", "NodeJS", "Rust", "C++"]
systems = [
    "auth",
    "gateway",
    "order",
    "payment",
    "inventory",
    "recommendation",
    "logging",
    "metrics",
    "user-profile",
    "ml-platform",
    "search-engine",
    "monitoring",
    "nginx-proxy",
]
envs = ["docker", "kubernetes", "ecs", "vm"]
teams = [
    "team-core",
    "team-app",
    "team-api",
    "team-ml",
    "team-devops",
    "team-observability",
]

base_dependencies = [
    ["redis", "mysql"],
    ["postgres"],
    ["kafka"],
    ["elasticsearch"],
    ["prometheus"],
    ["fluentd"],
    ["grpc"],
    ["protobuf"],
    ["jwt"],
]

generated = []

for i in range(98):
    service_name = f"service-{i}"
    domain = random.choice(systems)
    owner = random.choice(teams)
    lang = random.choice(languages)
    env = random.choice(envs)
    deps = random.choice(base_dependencies) + random.sample(
        ["uvicorn", "pydantic", "nginx", "fastapi", "sqlalchemy"], random.randint(0, 2)
    )

    # -------------------------------------------
    # Noise injection (simulate real CMDB mess)
    # -------------------------------------------
    if random.random() < 0.1:
        service_name = service_name.replace("-", "_")  # naming noise
    if random.random() < 0.05:
        owner = "unknown-team"  # missing metadata
    if random.random() < 0.05:
        deps.append("deprecated-lib")  # legacy dependency
    if random.random() < 0.03:
        deps = list(set(deps[1:]))  # missing first dependency

    service = {
        "service_name": service_name,
        "domain": domain,
        "owner": owner,
        "language": lang,
        "config_path": f"/etc/{service_name}/config.yaml",
        "system": domain,
        "deployment_env": env,
        "dependencies": deps,
        "last_updated": datetime.datetime.now().isoformat(),
    }

    generated.append(service)

# ==========================================================
# 3. Merge and output
# ==========================================================
cmdb = real_services + generated

output_path = "spark/data/cmdb.jsonl"

with open(output_path, "w") as f:
    for s in cmdb:
        f.write(json.dumps(s) + "\n")

print(f"✨ CMDB generated: {output_path}")
print(f"📦 Total services: {len(cmdb)} (including your real ones)")
