import logging.config
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
import time, random

LOGGING_CONFIG_JSON = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "service_name": "fastapi-demo", "message": "%(message)s"}',
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "default": {
            "formatter": "json",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {"handlers": ["default"], "level": "INFO"},
}

logging.config.dictConfig(LOGGING_CONFIG_JSON)

for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    logging.getLogger(name).disabled = True

logger = logging.getLogger(__name__)

app = FastAPI(title="FastAPI Demo")
Instrumentator().instrument(app).expose(app)


@app.get("/")
def health():
    logger.info("Health check endpoint called")
    return {"message": "ok"}


@app.get("/db_read")
def db_read():
    logger.info("Query result fetched successfully")
    return {"message": "Database read simulated"}


@app.get("/user_login")
def user_login():
    logger.info("User authentication succeeded")
    return {"message": "User login simulated"}


@app.get("/cpu_burst")
def cpu_burst():
    logger.error("CPU burst endpoint called")
    t = time.time()
    while time.time() - t < 18:
        _ = [x**2 for x in range(300000)]
    return {"message": "CPU overload simulated"}


@app.get("/error")
def error():
    logger.error("Critical failure: unhandled exception raised inside /error endpoint")
    raise HTTPException(status_code=500, detail="Simulated internal error")
