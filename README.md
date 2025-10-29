# ML monitoring
A machine learning monitoring framework using docker compose to deploy


## Folder Files introduction
```
├── /.github/workflows/    # github actions cicd pipeline config file
    ├── cloud_cicd.yml      # cloud pipeline code
├── /fastapi-demo/           # cloud application code
    ├── main.py               # spark code
    ├── Dockerfile           # cloud source code
├── /nginx/                  # edge application code
    ├── nginx.conf           # edge source code
├── /prometheus/             # sensor simulation code
    ├── alert.rules.yml      # simulation data structure
    ├── prometheus.yml        # make simulation data using http post to the aws environment
├── /spark/                  # sensor simulation code
    ├── /data                # simulation data structure
    ├── /models             # make simulation data using http post to the aws environment
    ├── /script             # make simulation data using http post to the aws environment
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
