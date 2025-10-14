# ML monitoring
A machine learning monitoring framework

## Simulation tool
- iFogSim: for sensor and edge gateway device
- AWS: for edge and cloud deployment

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
- flask
- boto3
- pyspark

## running step
### sensor and gateway
put the iFogSim code like shown below in the iFogSim folder
```
├── /org.for.smart/      
    ├── MyFogBroker.java  
    ├── Smart.java
```

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


## export ecs task-define json command
aws ecs describe-task-definition \
  --task-definition edge-taskk \
  --query "taskDefinition" \
  --output json > ecs-task-def.json

## AWS Redshift sync table
```
CREATE TABLE dev.public.fog (
  device_type  VARCHAR(50),
  device_id    VARCHAR(64),
  reading      BIGINT,
  unit         VARCHAR(10),
  battery      SMALLINT,
  status       VARCHAR(32),
  "timestamp"  VARCHAR(32),
  anomaly      BOOLEAN,
  location     VARCHAR(100),
  ingest_time  DOUBLE PRECISION
)

```