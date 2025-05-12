from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests

def retrain_model():
    # 여기에 모델 재학습 로직 넣기
    print("Relearning model...")

def update_metrics():
    try:
        response = requests.get("http://monitor:8000/update_metrics")
        print(f"Metrics updated: {response.status_code}")
    except Exception as e:
        print(f"Failed to update metrics: {e}")

with DAG(
    dag_id='model_training',
    start_date=datetime(2025, 4, 28),
    schedule_interval='@daily',
    catchup=False,
    is_paused_upon_creation=False,
) as dag:
    
    retrain_task = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model,
    )

    update_metrics_task = PythonOperator(
        task_id='update_metrics',
        python_callable=update_metrics,
    )

    # 의존성 설정: 재학습 후 메트릭 갱신
    retrain_task >> update_metrics_task
