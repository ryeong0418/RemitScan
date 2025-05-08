# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 각 기능별 라우터 import
from routes import (
    registration_risk,
    # account_risk,
    # transaction_risk,
    # aml_evaluation,
    # behavioral_score,
    # system_monitor,
    # audit_log
)

app = FastAPI(
    title="FDS API - 이상거래탐지 시스템",
    description="신규 가입부터 이상거래 리포트까지 전 과정 탐지 API",
    version="1.0.0"
)

# CORS 설정 (내부망일 경우 생략 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안을 위해 실제 운영 시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 각 탐지 기능별 라우터 등록
app.include_router(registration_risk.router, prefix="/risk", tags=["가입 위험 탐지"])
# app.include_router(account_risk.router, prefix="/risk", tags=["계좌 등록 위험 탐지"])
# app.include_router(transaction_risk.router, prefix="/risk", tags=["송금 거래 탐지"])
# app.include_router(behavioral_score.router, prefix="/risk", tags=["행동 기반 위험 점수"])

# app.include_router(aml_evaluation.router, prefix="/aml", tags=["AML 룰 탐지"])
# app.include_router(system_monitor.router, prefix="/system", tags=["시스템 모니터링"])
# app.include_router(audit_log.router, prefix="/audit", tags=["감사/리포트 관리"])
