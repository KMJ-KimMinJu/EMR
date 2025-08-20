# app.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model

# ---------- 0) 로딩 ----------
MODEL_PATH = "./icu_model.h5"
COLS_PATH = "./feature_columns.json"
THR_PATH = "./thresholds.json"

model = load_model(MODEL_PATH)
feature_cols = json.load(open(COLS_PATH, "r"))  # list[str]
thresholds = json.load(open(THR_PATH, "r"))     # {"icu_transfer": 0.xx}
THR_ICU = float(thresholds.get("icu_transfer", 0.5))

# ---------- 1) 입력 스키마 ----------
class VitalPayload(BaseModel):
    # 시간
    first_vs_time: Optional[str] = Field(None, description="예: 2025-08-21T10:00:00")
    near_vs_time:  Optional[str] = Field(None, description="예: 2025-08-21T12:15:00")
    delta_time_hr: Optional[float] = None

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

app = FastAPI(title="ICU Risk API (Vitals Only)")

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

# ---------- 3) 추론 엔드포인트 ----------
@app.post("/predict/icu")
def predict_icu(payload: VitalPayload):
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

    return {
        "icu_prob": prob,
        "icu_high_risk": bool(label),
        "threshold": THR_ICU
    }

# 로컬 실행용
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
