from typing import Optional, List, Literal
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Predict API", version="1.0.0")

# ======================
# 요청 스키마 (워커에서 보내는 필드명과 동일)
# ======================
## test
## test2


class VitalReq(BaseModel):
    patientId: int
    temperature: Optional[float] = None
    heartrate: Optional[float] = None
    resprate: Optional[float] = None
    o2sat: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    map: Optional[float] = None

    source: Optional[str] = "vital"
    vital_pk: Optional[int] = None
    measured_at: Optional[datetime] = None  # worker에서 record_time을 여기에 넣어줌

class FullReq(BaseModel):
    patientId: int

    # lab
    lactate: Optional[float] = None
    glucose: Optional[float] = None
    pH: Optional[float] = None
    bicarbonate: Optional[float] = None
    aniongap: Optional[float] = None
    troponin: Optional[float] = None
    wbc: Optional[float] = None
    creatinine: Optional[float] = None
    potassium: Optional[float] = None
    bnp: Optional[float] = None
    uketone: Optional[int] = None           # TINYINT(1) → 0/1 정수로 받음
    hydroxybutyrate: Optional[float] = None

    # (옵션) 최신 vital 동봉
    temperature: Optional[float] = None
    heartrate: Optional[float] = None
    resprate: Optional[float] = None
    o2sat: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    map: Optional[float] = None

    source: Optional[str] = "lab"
    lab_pk: Optional[int] = None
    lab_measured_at: Optional[datetime] = None
    vital_pk: Optional[int] = None
    vital_measured_at: Optional[datetime] = None
    vital_attached: bool = False

# ======================
# 응답 스키마 (프론트가 받길 원하는 형태 그대로)
# ======================

class VitalRes(BaseModel):
    success: bool = True
    predict: Literal["고위험군", "저위험군"]

class Disease(BaseModel):
    name: str
    percent: int
    basis: str

class FullRes(BaseModel):
    success: bool = True
    ICU_percent: int
    ICU_hours: int
    diseases: List[Disease]

# ======================
# 엔드포인트
# ======================

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict/vital", response_model=VitalRes)
def predict_vital(req: VitalReq):
    """
    TODO: 여기에서 실제 모델 호출로 교체.
    지금은 테스트용으로 간단 규칙:
      - 체온 >= 38.0 또는 심박수 >= 120 이면 고위험군
    """
    high = False
    if req.temperature is not None and req.temperature >= 38.0:
        high = True
    if req.heartrate is not None and req.heartrate >= 120:
        high = True

    return VitalRes(success=True, predict=("고위험군" if high else "저위험군"))

@app.post("/predict/full", response_model=FullRes)
def predict_full(req: FullReq):
    """
    TODO: 여기에서 실제 모델 호출로 교체.
    지금은 테스트용으로 임의/간단 계산:
      - 몇몇 lab이 채워져 있으면 ICU_percent를 대충 60~80 사이로
      - ICU_hours는 3으로 고정
      - diseases는 예시 두 개 리턴
    """
    filled_lab_fields = [
        x for x in [
            req.lactate, req.troponin, req.wbc, req.creatinine, req.bnp
        ] if x is not None
    ]
    base = 50 + min(len(filled_lab_fields) * 6, 30)  # 50~80
    icu_percent = max(0, min(int(round(base)), 100))

    diseases = [
        Disease(name="sepsis", percent=92, basis="sbp 250 및 뭐시기"),
        Disease(name="aki",    percent=75, basis="뭐시기뭐사기 하니까 영상 검사 하aasdfadsfasdfasdfasfasdfasdfasdfasdfasdfasdfadsfasdfaWsdfvadsdfvaWefvaew4fvxzcdvzsef니까"),
        Disease(name="aki",    percent=75, basis="뭐시기뭐사기 하니까 영상 검사 필요"),
        Disease(name="aki",    percent=75, basis="뭐시기뭐사기 하니까 영상 검사 필요"),
        Disease(name="aki",    percent=75, basis="뭐시기뭐사기 하aasdfadsfasdfasdfasfasdfasdfasdfasdfasdfasdfadsfasdfaWsdfvadsdfvaWefvaew4fvxzcdvzsef니까 영상 검사 필요"),
    ]

    return FullRes(
        success=True,
        ICU_percent=icu_percent,
        ICU_hours=3,
        diseases=diseases
    )
