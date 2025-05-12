from fastapi import FastAPI, Request
from kafka import KafkaProducer
import json
import tritonclient.grpc
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import start_http_server, Gauge, Counter
import threading
import random

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Kafka Producer 생성
producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# 커스텀 메트릭
fraud_model_precision = Gauge('fraud_model_precision', 'Precision of fraud detection model')
fraud_model_recall = Gauge('fraud_model_recall', 'Recall of fraud detection model')
fraud_model_accuracy = Gauge('fraud_model_accuracy', 'Accuracy of fraud detection model')
fraud_model_feature_drift = Gauge('fraud_model_feature_drift', 'Feature drift score (e.g., population stability index)')
fraud_transactions_total = Counter('fraud_transactions_total', 'Total number of processed transactions')

def triton_infer(data: dict):
    return {"inference": "dummy"}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(data: dict):
    # 1. Kafka로 거래 메시지 전송
    producer.send("transactions", value=data)
    producer.flush()

    # 2. Triton 추론 요청 (선택)
    response = triton_infer(data)  # ← 이 함수는 너가 따로 정의해야 함

    # 3. 메트릭 업데이트 (예시)
    fraud_transactions_total.inc()

    return {"status": "sent to kafka", "triton_response": data}


from fastapi import Body

@app.post("/update_metrics")
def update_metrics(metrics: dict = Body(...)):
    fraud_model_precision.set(metrics["precision"])
    fraud_model_recall.set(metrics["recall"])
    fraud_model_accuracy.set(metrics["accuracy"])
    fraud_model_feature_drift.set(metrics["drift"])
    fraud_transactions_total.inc()
    return {"status": "metrics updated"}



# Prometheus HTTP endpoint 노출
threading.Thread(target=lambda: start_http_server(9101)).start()
