# services/registration_checker.py

HIGH_RISK_COUNTRIES = {"KP", "IR", "SY", "RU"}
BLOCKED_IPS = {"10.0.0.1", "192.168.0.99"}
DEVICE_BLACKLIST = {"emulator_test", "rooted_device"}
SANCTIONED_PHONE_PREFIXES = {"+850", "+98"}  # 북한, 이란 등

def check_registration_risk(data):
    #시장 조사 보완 예정
    
    reasons = []

    if data.ip in BLOCKED_IPS:
        reasons.append("차단된 IP에서 접속")

    if data.device_id in DEVICE_BLACKLIST:
        reasons.append("블랙리스트 단말기 사용")

    if data.country_code.upper() in HIGH_RISK_COUNTRIES:
        reasons.append("고위험 국가에서 접속")

    if any(data.phone_number.startswith(prefix) for prefix in SANCTIONED_PHONE_PREFIXES):
        reasons.append("제재 국가 번호 사용")

    # RBA - 위험 점수 간단 계산 예시
    rba_score = 0
    if "vpn" in data.device_id.lower():
        rba_score += 3
        reasons.append("VPN 접속 의심")
    if "root" in data.device_id.lower():
        rba_score += 2
        reasons.append("루팅된 단말기 의심")

    if rba_score >= 5:
        reasons.append("위험 점수 초과")

    return {
        "is_risky": len(reasons) > 0,
        "reasons": reasons
    }
