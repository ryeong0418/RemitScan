import pandas as pd
import requests
import json
import time

# CSV 읽기 + 컬럼 이름 리네임
df = pd.read_csv('data/test_transaction.csv')
df.rename(columns={
    "TransactionID": "transaction_id",
    "TransactionAmt": "amount",
    "card4": "card_type",
    "TransactionDT": "timestamp"
}, inplace=True)

# 전송
for _, row in df.iterrows():
    data = {
        "transaction_id": row["transaction_id"],
        "amount": row["amount"],
        "card_type": row["card_type"],
        "timestamp": row["timestamp"]
    }

    try:
        response = requests.post(
            "http://localhost:8000/predict/",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        
        # 👇 수정: 응답 텍스트와 상태코드 먼저 확인
        print(f"✅ Sent: {data}")
        print(f"🔁 Status: {response.status_code}")
        print(f"🔁 Raw response text: {response.text}")

        # 그 다음 json 파싱 시도
        try:
            print(f"🔁 Parsed JSON: {response.json()}")
        except json.JSONDecodeError:
            print("⚠️ Response is not valid JSON")

    except Exception as e:
        print(f"❌ Error: {e}")


    time.sleep(0.5)
