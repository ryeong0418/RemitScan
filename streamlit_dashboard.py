import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dashboard.modules.eda import FraudDataEDA
from st_aggrid import AgGrid, GridOptionsBuilder
import numpy as np

import os
import joblib
import streamlit as st
from dashboard.modules.ensemble import FraudEnsemble

# Streamlit 설정
st.set_page_config(page_title="📊 이상 거래 탐지", layout="wide")
st.title("🧠 FDS")

# ✅ 고정된 CSV 경로
unsupervised_csv_path = "dashboard/data/unsupervised_pdf.csv"

# ✅ threshold 불러오기
threshold = joblib.load("dashboard/models/custom_threshold.pkl")
importances = joblib.load("dashboard/models/feature_importance.pkl")


# ✅ 데이터 로딩 및 예측
ensemble = FraudEnsemble()
df_raw, X = ensemble.load_data(unsupervised_csv_path)
df_pred = ensemble.predict(df_raw, X, threshold)

# ✅ 주요 메트릭 요약
total = len(df_pred)
predicted_fraud = df_pred["prediction"].sum()
df_pred["true_label"] = df_pred["pseudo_label_avg"]

if "true_label" in df_pred.columns:
    true_fraud = df_pred["true_label"].sum()
    true_positive = ((df_pred["prediction"] == 1) & (df_pred["true_label"] == 1)).sum()
    recall = round((true_positive / true_fraud) * 100, 2) if true_fraud else 0
    recall_display = f"{recall}%"
else:
    recall_display = "N/A"

# ✅ 메트릭 시각화 -> 테두리 설정하는 부분
st.subheader("📌 주요 탐지 지표")
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown(
            """
            <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px;">
                <p style="margin: 0; font-weight: bold;">전체 거래 수</p>
                <h3 style="margin: 0;">{:,.0f}</h3>
            </div>
            """.format(total),
            unsafe_allow_html=True
        )

with col2:
    with st.container():
        st.markdown(
            """
            <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px;">
                <p style="margin: 0; font-weight: bold;">탐지된 이상 거래 수</p>
                <h3 style="margin: 0;">{:,.0f}</h3>
            </div>
            """.format(predicted_fraud),
            unsafe_allow_html=True
        )

with col3:
    with st.container():
        st.markdown(
            f"""
        <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px;">
            <p style="margin: 0; font-weight: bold;">탐지율 (Recall)</p>
            <h3 style="margin: 0;">{recall_display}</h3>
        </div>
        """,
            unsafe_allow_html=True
        )

# 한글 폰트 설정 (Windows에서 'Malgun Gothic' 사용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

st.subheader("💰 거래 금액 분포 비교")

fig, ax = plt.subplots()

df_pred[df_pred['prediction'] == 0]['TransactionAmt'].hist(
    ax=ax, bins=50, alpha=0.5, label="정상"
)
df_pred[df_pred['prediction'] == 1]['TransactionAmt'].hist(
    ax=ax, bins=50, alpha=0.5, label="이상"
)
# ✅ x축 포맷: 천 단위 콤마
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# ✅ x축 범위 제한: 0 ~ 5,000,000
ax.set_xlim(0, 5_000_000)
ax.set_title("거래 금액 분포 (정상 vs 이상)")
ax.set_xlabel("거래 금액")
ax.set_ylabel("거래 수 (빈도)")
ax.legend()
st.pyplot(fig)


eda = FraudDataEDA()
fds_df = eda.extract_rule_combo()

# 조합별 비율 계산
col1, col2 = st.columns([1,1]) # 좌우 50:50 비율

with col1:

    st.markdown(
        """
        <div style="border: 2px solid #FF6B6B; border-radius: 10px; padding: 15px; background-color: #FFF0F0;">
            <h4>📌 Rule 기반 이상 거래 탐지 조건</h4>
            <ul>
                <li><b>Rule 1</b>: 거래 금액 ≥ 700,000원</li>
                <li><b>Rule 2</b>: 계좌 평균 대비 2배 이상 거래</li>
                <li><b>Rule 3</b>: VPN 사용, 신규 기기, 루팅 중 하나라도 포함</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    # Pie Chart: Rule 충족 여부
    rule_hit = fds_df[["is_high_amt", "is_high_ratio", "has_risk_factor"]].all(axis=1)
    hit = rule_hit.sum()
    print(hit)
    miss = len(rule_hit) - hit



    fig, ax = plt.subplots()
    ax.pie(
        [hit, miss],
        labels=["Rule 충족", "충족 안함"],
        autopct="%1.1f%%",
        colors=["#FF6B6B", "#C0C0C0"],
        startangle=90
    )
    # ax.set_title("전체 거래 중 Rule 충족 여부")
    ax.axis("equal")  # 원형 유지
    st.pyplot(fig)

top_features = importances.sort_values(ascending=False).head(10)

st.subheader("🎯 중요 피처 TOP 10")
# st.bar_chart(top_features)
fig, ax = plt.subplots()
top_features.sort_values().plot(
    kind="barh",
    ax=ax,
    color="#FF6B6B"
)

ax.set_xlabel("중요도 (feature importance)")
ax.set_ylabel("feature name")
st.pyplot(fig)

# ✅ 통계 리포트 (true_label이 존재할 경우에만)
if "true_label" in df_raw.columns:
    from sklearn.metrics import classification_report, confusion_matrix

    st.subheader("📊 Confusion Matrix")

    matrix = confusion_matrix(df_raw["true_label"], df_pred["prediction"])
    matrix_df = pd.DataFrame(
        matrix,
        index=["실제:정상", "실제:이상"],
        columns=["예측:정상", "예측:이상"]
    )
    st.table(matrix_df)

    st.subheader("📈 Classification Report")
    report_dict = classification_report(
        df_raw["true_label"],
        df_pred["prediction"],
        target_names=["정상", "이상"],
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T.round(3)
    st.table(report_df)

df_pred["transaction_time"] = pd.to_datetime(df_pred["transaction_time"])
#
# ✅ 이상 거래만 필터링
anomaly_df = df_pred[df_pred["prediction"]==1].copy()

# ✅ 일별 이상 거래 수
daily_anomalies = anomaly_df.groupby(
    anomaly_df["transaction_time"].dt.date
).size().reset_index(name="anomaly_count")
daily_anomalies.columns=["일자", "이상거래 수"]

# ✅ 월별 이상 거래 수
monthly_anomalies = anomaly_df.groupby(
    anomaly_df["transaction_time"].dt.to_period("M")
).size().reset_index(name="anomaly_count")
monthly_anomalies["월"] = monthly_anomalies["transaction_time"].dt.strftime("%Y-%m")
monthly_anomalies = monthly_anomalies[["월", "anomaly_count"]]
monthly_anomalies = monthly_anomalies.set_index("월")

st.subheader("📈 월별 이상 거래 수 (선 그래프)")
st.line_chart(
    data=monthly_anomalies.reset_index(),
    x="월",
    use_container_width=True
)

st.subheader("📋 일별 이상 거래 수 (표)")
st.dataframe(daily_anomalies, use_container_width=True)


st.subheader("⚠️ 탐지된 이상 거래")
anomalies = df_pred[df_pred["prediction"] == 1]
gb = GridOptionsBuilder.from_dataframe(anomalies)
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
grid_options = gb.build()
AgGrid(anomalies, gridOptions=grid_options, height=500)