import json
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os


class FraudEnsemble:

    def __init__(self):
        self.base_path=os.path.dirname(os.path.abspath(__file__))
        self.project_root=os.path.dirname(os.path.dirname(self.base_path))
        self.model = joblib.load(f'{self.project_root}/dashboard/models/stack_model.pkl')
        self.threshold = joblib.load(f'{self.project_root}/dashboard/models/custom_threshold.pkl')
        self.scaler = joblib.load(f'{self.project_root}/dashboard/models/scaler.pkl')

        with open(os.path.join(self.project_root, 'dashboard/models/feature_names.json')) as f:
            self.feature_cols = json.load(f)

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        X_raw = df[self.feature_cols] # 예측에 필요한 feature 컬럼만 추출
        X = self.scaler.transform(X_raw) # 이미 저장된 스케일러 사용

        return df, X

    def predict(self, df, X, threshold):

        proba = self.model.predict_proba(X)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        df["prediction"] = y_pred
        df["risk_score"] = proba

        return df

# if __name__ == "__main__":
#     filename = 'unsupervised_pdf'
#     csv_path = os.path.join(project_root, 'dashboard/data', f'{filename}.csv')
#     df, X = load_data(csv_path)
#     print(predict(df, X, threshold))
