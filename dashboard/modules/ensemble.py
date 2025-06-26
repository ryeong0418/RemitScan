import json
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# 모델 및 feature 정보 로드
base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_path))

model = joblib.load(f'{project_root}/dashboard/models/stack_model.pkl')
threshold = joblib.load(f'{project_root}/dashboard/models/custom_threshold.pkl')
scaler = joblib.load(f'{project_root}/dashboard/models/scaler.pkl')

with open(f'{project_root}/dashboard/models/feature_names.json') as f:
    feature_cols = json.load(f)

# 입력 데이터 로딩
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # ❗예측에 필요한 feature 컬럼만 추출
    X_raw = df[feature_cols]
    # 이미 저장된 스케일러 사용
    X = scaler.transform(X_raw)

    return df, X

def predict(df, X, threshold):

    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    df["prediction"] = y_pred
    df["risk_score"] = proba
    print(df)
    return df

if __name__ == "__main__":
    filename = 'unsupervised_pdf'
    csv_path = os.path.join(project_root, 'dashboard/data', f'{filename}.csv')
    df, X = load_data(csv_path)
    print(predict(df, X, threshold))
