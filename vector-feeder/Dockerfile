FROM python:3.10-slim-bookworm
WORKDIR /app
COPY main.py .
RUN pip install fastapi uvicorn requests boto3
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]