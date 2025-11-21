# ML monitoring
A machine learning monitoring framework using docker compose to deploy


## Folder Files introduction
```
├── /.github/workflows/      # github actions cicd pipeline config file
    ├── cloud_cicd.yml       # cloud pipeline code
├── /chaos/                  # chaos experiments code
    ├── calc_metrics.py      # calculate metrics for all experiments
    ├── probe_ai_monitor.py  # detect ai system script
    ├── probe_prometheus.py  # detect prometheus system script
├── /fastapi-demo/           # demo application code
    ├── Dockerfile           # docker build file
    ├── main.py              # demo application souce code
├── /nginx/                # nginx proxy
    ├── nginx.conf         # nginx conf
├── /prometheus/           # prometheus monitoring system
    ├── alert.rules.yml    # static alert rule file
    ├── prometheus.yml     # prometheus main configuration file
├── /spark/                # spark big data engine
    ├── /data              # parquet file
    ├── /ivy2              # java jars cache
    ├── /metrics           # some experiments result
    ├── /models            # all models dir
    ├── /script            # spark shell script
├── /vector-feeder/        # simulated real application action
    ├── /data              # feeder file
    ├── Dockerfile         # docker build file
    ├── main.py            # feeder source code
├── .env                   # docker compose env file
├── docker-compose.yml     # docker compose file
├── vector.yml             # vector main configuration file
```

## Dependency
- docker
- spark
- kafka
- prometheus
- fastapi
- chaos

## running step
### docker compose up -d
put the iFogSim code like shown below in the iFogSim folder
```
├── /org.for.smart/      
    ├── MyFogBroker.java  
    ├── Smart.java
```

## simulate

while true; do 
  echo "$(date '+%H:%M:%S') $(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8088)"
  echo "$(date '+%H:%M:%S') $(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8088/db_read)"
  echo "$(date '+%H:%M:%S') $(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8088/user_login)"
  sleep 1
done


### edge and cloud application
```
pip install -r requirements.txt
```

```
cd edge && python edge_app.py
```

```
cd cloud && python cloud_app.py
```
