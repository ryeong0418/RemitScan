from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType, DoubleType

# 1. Spark 세션 생성
spark = SparkSession.builder \
    .appName("AnomalyDetection") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")  # 불필요한 로그 줄이기

# 2. Kafka에서 스트리밍 데이터 수신
raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9093") \
    .option("subscribe", "transactions") \
    .load()

# 3. 메시지의 value 부분을 JSON 문자열로 파싱
json_df = raw_df.selectExpr("CAST(value AS STRING) as json_string")

# 4. JSON 스키마 정의 (Kafka 메시지에 들어오는 구조에 맞게)
schema = StructType() \
    .add("transaction_id", StringType()) \
    .add("amount", DoubleType()) \
    .add("card_type", StringType()) \
    .add("timestamp", StringType())

parsed_df = json_df.select(from_json(col("json_string"), schema).alias("data")).select("data.*")

# 5. 이상 거래 탐지 로직
# ----> 여기서부터 수정 ----
import requests

def predict_with_gnn(batch_df, batch_id):
    import torch
    from graph.gnn_model import GCN
    from graph.graph_utils import build_graph_from_df

    pandas_df = batch_df.toPandas()
    if len(pandas_df) == 0:
        return

    graph_data = build_graph_from_df(pandas_df, edge_cols=['card_type'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels=graph_data.num_node_features)
    model.load_state_dict(torch.load('client_0_final_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
        pred = out.argmax(dim=1).cpu().numpy()

    pandas_df['is_anomaly'] = pred

    # 👉 평가 지표 계산 예시 (정밀도, 재현율, 정확도)
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    # 실제 레이블이 없다면 임의로 0으로 대체
    if 'label' not in pandas_df.columns:
        pandas_df['label'] = 0  # 전부 정상이라고 가정 (실제 환경에 맞게 수정)

    precision = precision_score(pandas_df['label'], pred, zero_division=0)
    recall = recall_score(pandas_df['label'], pred, zero_division=0)
    accuracy = accuracy_score(pandas_df['label'], pred)
    drift = round(random.uniform(0.01, 0.1), 3)  # PSI나 통계 기반 drift 계산 예정

    # 👉 FastAPI 서버에 메트릭 전송
    try:
        requests.post(
            "http://localhost:8000/update_metrics",
            json={
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "accuracy": round(accuracy, 3),
                "drift": drift
            }
        )
    except Exception as e:
        print(f"⚠️ 메트릭 전송 실패: {e}")



# ✅ Spark Streaming 연결
query = parsed_df.writeStream \
    .foreachBatch(predict_with_gnn) \
    .start()

query.awaitTermination()
