# routes/registration_risk.py

from fastapi import APIRouter
from schemas.registration import RegistrationRiskRequest, RegistrationRiskResponse
from services.registration_checker import check_registration_risk

router = APIRouter()

@router.post("/check-registration", response_model=RegistrationRiskResponse)
def registration_risk_check(request: RegistrationRiskRequest):
    return check_registration_risk(request)
