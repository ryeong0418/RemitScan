FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    kafka-python \
    fastapi \
    uvicorn \
    prometheus_client \
    prometheus-fastapi-instrumentator \
    tritonclient[grpc]

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
