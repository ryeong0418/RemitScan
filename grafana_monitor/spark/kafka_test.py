
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AnomalyDetection").getOrCreate()

# Kafka에서 데이터 읽기
df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:9093").option("subscribe", "transactions").load()

# 데이터를 처리하고 피처 추출 (예시)
df = df.selectExpr("CAST(value AS STRING)").select("value")
df.printSchema()
