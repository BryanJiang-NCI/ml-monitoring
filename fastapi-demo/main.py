from fastapi import FastAPI, HTTPException
import random, time
import logging
import sys
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI()

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

Instrumentator().instrument(app).expose(app)


@app.get("/")
def health():
    logging.info("Health check endpoint called")
    return {"status": "ok"}


@app.get("/cpu_burst")
def cpu_burst():
    logging.info("CPU burst endpoint called")
    t = time.time()
    while time.time() - t < 3:
        _ = [x**2 for x in range(100000)]
    return {"message": "CPU overload simulated"}


@app.get("/error")
def error():
    if random.random() < 0.7:
        logging.error("Simulated internal error occurred")
        raise HTTPException(status_code=500, detail="Simulated internal error")
    return {"status": "ok"}
