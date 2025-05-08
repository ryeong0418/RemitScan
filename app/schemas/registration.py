from pydantic import BaseModel
from typing import List

class RegistrationRiskRequest(BaseModel):
    ip: str
    device_id: str
    country_code: str
    phone_number: str
    timestamp: str

class RegistrationRiskResponse(BaseModel):
    is_risky: bool
    reasons: List[str]
