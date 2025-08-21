from typing import Optional, List, Literal
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
app = FastAPI(title="Predict API", version="1.0.0")

# ======================
# 요청 스키마 (워커에서 보내는 필드명과 동일)
# ======================
## test
## test2

# ---------- 0) 로딩 ----------
MODEL_PATH = "./icu_model.h5"
COLS_PATH = "./feature_columns.json"
THR_PATH = "./thresholds.json"

model = load_model(MODEL_PATH)
feature_cols = json.load(open(COLS_PATH, "r"))  # list[str]
thresholds = json.load(open(THR_PATH, "r"))     # {"icu_transfer": 0.xx}
THR_ICU = float(thresholds.get("icu_transfer", 0.5))

class VitalReq(BaseModel):
    patientId: int
    temperature: Optional[float] = None
    heartrate: Optional[float] = None
    resprate: Optional[float] = None
    o2sat: Optional[float] = None
    sbp: Optional[float] = None
    # dbp: Optional[float] = None
    map: Optional[float] = None

    source: Optional[str] = "vital"
    vital_pk: Optional[int] = None
    measured_at: Optional[datetime] = None  # worker에서 record_time을 여기에 넣어줌


class VitalPayload(BaseModel):
    # 시간
    first_vs_time: Optional[str] = None
    near_vs_time:  Optional[str] = None
    delta_time_hr: Optional[float] = 0.0

    # 바이탈 (두 시점)
    sbp_first: Optional[float] = None; sbp_near: Optional[float] = None
    map_first: Optional[float] = None; map_near: Optional[float] = None
    hr_first:  Optional[float] = None; hr_near:  Optional[float] = None
    rr_first:  Optional[float] = None; rr_near:  Optional[float] = None
    spo2_first:Optional[float] = None; spo2_near:Optional[float] = None
    temp_first: Optional[float] = None; temp_near: Optional[float] = None

    # 선택: 승압제/바소프레서
    norepinephrine: Optional[float] = 0
    epinephrine:   Optional[float] = 0
    dopamine:      Optional[float] = 0
    dobutamine:    Optional[float] = 0
    phenylephrine: Optional[float] = 0
    vasopressin:   Optional[float] = 0
    milrinone:     Optional[float] = 0



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
    # dbp: Optional[float] = None
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

from datetime import datetime

def to_vital_payload_same(req: "VitalReq") -> "VitalPayload":
    t = req.measured_at or datetime.utcnow()
    iso = t.isoformat()
    return VitalPayload(
        first_vs_time=iso,
        near_vs_time=iso,
        delta_time_hr=0.0,

        sbp_first=req.sbp,  sbp_near=req.sbp,
        map_first=req.map,  map_near=req.map,
        hr_first=req.heartrate, hr_near=req.heartrate,
        rr_first=req.resprate,  rr_near=req.resprate,
        spo2_first=req.o2sat,   spo2_near=req.o2sat,
        temp_first=req.temperature, temp_near=req.temperature,

        # 바소프레서는 입력이 없으니 0
        norepinephrine=0, epinephrine=0, dopamine=0, dobutamine=0,
        phenylephrine=0, vasopressin=0, milrinone=0,
    )


# ---------- 2) 전처리: add_time_delta_features와 동일 로직(요약 구현) ----------
def compute_delta_time_hr(first_time: Optional[str], near_time: Optional[str], delta_time_hr: Optional[float]) -> float:
    if delta_time_hr is not None:
        return max(0.0, float(delta_time_hr))
    if first_time and near_time:
        try:
            a = datetime.fromisoformat(first_time)
            b = datetime.fromisoformat(near_time)
            return max(0.0, (b - a).total_seconds() / 3600.0)
        except Exception:
            return 0.0
    return 0.0

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # delta_time_hr
    df["delta_time_hr"] = df.apply(
        lambda r: compute_delta_time_hr(r.get("first_vs_time"), r.get("near_vs_time"), r.get("delta_time_hr")),
        axis=1
    )

    # first/near 쌍에서 delta/rate 생성
    pairs = []
    for base in ["sbp","map","hr","rr","spo2","temp"]:
        f, n = f"{base}_first", f"{base}_near"
        if f in df.columns and n in df.columns:
            pairs.append((base, f, n))

    for base, fcol, ncol in pairs:
        dcol = f"delta_{base}"
        rcol = f"rate_{base}_per_hr"
        df[dcol] = pd.to_numeric(df[ncol], errors="coerce") - pd.to_numeric(df[fcol], errors="coerce")
        # 시간당 변화율
        safe_dt = df["delta_time_hr"].replace(0, np.nan)
        rate = df[dcol] / safe_dt
        df[rcol] = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 임상 플래그(원 코드 기준)
    if "delta_sbp" in df:  df["flag_worse_hypotension_sbp"] = (df["delta_sbp"] <= -20).astype(int)
    if "delta_map" in df:  df["flag_worse_hypotension_map"] = (df["delta_map"] <= -10).astype(int)
    if "delta_spo2" in df: df["flag_worse_spo2"]            = (df["delta_spo2"] <= -3).astype(int)
    if "delta_rr" in df:   df["flag_worse_tachypnea"]       = (df["delta_rr"]   >= 5).astype(int)
    if "delta_hr" in df:   df["flag_worse_tachycardia"]     = (df["delta_hr"]   >= 20).astype(int)
    if "delta_temp" in df: df["flag_worse_fever"]           = (df["delta_temp"] >= 1.0).astype(int)

    return df


# ======================
# 엔드포인트
# ======================

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict/vital", response_model=VitalRes)
def predict_vital(req: VitalReq):

    payload = to_vital_payload_same(req)

    # 3-1) 입력 → DF
    row = pd.DataFrame([payload.dict()])

    # 3-2) 특징 생성(시간/변화/플래그)
    row = make_features(row)

    # 3-3) (중요) feature_cols 순으로 정렬
    #     실무에서는 scaler가 '원시 입력 컬럼들' 기준인지, '파생특징까지 포함한 최종 컬럼' 기준인지에 맞춰서 처리해야 합니다.
    #     여기서는 '최종 입력 컬럼(feature_cols)'이 스케일러 대상이라고 가정합니다.
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0.0  # 누락된 컬럼은 0으로 채움(학습 시 NaN→0 처리에 맞춤)
    X = row[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 3-4) 표준화
    Xs = X.values

    # 3-5) 예측
    prob = float(model.predict(Xs, batch_size=1)[0][0])  # 출력이 단일(icu_transfer)이라 가정
    label = int(prob >= THR_ICU)
    predict_res = ''

    if label :
        predict_res = "고위험군"
    else :
        predict_res = "저위험군"
    
    return VitalRes(success=True, predict=predict_res)


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
