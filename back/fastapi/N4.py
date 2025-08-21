# N version + 태그 코드 정확도 점검 + 학습 전 데이터 준비 코드 수정 
# + 시간에 따른 바이탈 변화 (가이드 라인 제외)

"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""

import random

import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.optimizers import Adam
from keras.initializers import VarianceScaling
from keras.layers import Input, Dense, Concatenate, Lambda, Layer
from keras.models import Model
from keras.callbacks import Callback
from keras.metrics import AUC

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, mean_absolute_error

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from typing import Dict, Optional, List, Tuple, Any, Union

from mmoe import MMoE

#시드 고정
SEED = 1
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            print(
                'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                    output_name, round(train_roc_auc, 4),
                    output_name, round(validation_roc_auc, 4),
                    output_name, round(test_roc_auc, 4)
                )
            )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# 태그 생성
def add_er_icd_tags(df, icd_col="primary_icd_code", ver_col="primary_icd_version"):
    """
    응급실 주진단 ICD(9/10) 코드를 tag_* 멀티라벨로 변환.
    - chest pain, sepsis, depression, alcohol intox/abuse, hearing loss, pneumonia
    - dka, mi_acute, shock_general, metabolic_acidosis, hyperkalemia, chf, pulmonary_edema
    """

    # 1) 코드 정규화: 대문자, 공백 제거, 점(.) 제거
    def norm_code(x: str) -> str:
        if pd.isna(x): return ""
        return str(x).strip().upper().replace(".", "")

    code_norm = df[icd_col].astype(str).map(norm_code) if icd_col in df.columns else pd.Series("", index=df.index)

    # 2) 매핑 정의 
    MAP = {
        "tag_chest_pain": { "R079", "78650", "78659", },
        "tag_sepsis": { "A419", "0389", "038", },
        "tag_depression": { "F329", "311", },
        "tag_alcohol_intox": { "F10129", },
        "tag_alcohol_abuse": { "30500", },
        "tag_hearing_loss": { "389", },
        "tag_pneumonia": { "486", "J189", },
        "tag_dka": { "25010", "25012", "25013" },
        "tag_mi_acute": { "I210", "I219", "410", },
        "tag_shock_general": { "R57", "7855", },
        "tag_metabolic_acidosis": { "E872", "2762", },
        "tag_hyperkalemia": { "E875", "2767", },
        "tag_chf": { "I50", "428", },
        "tag_pulmonary_edema": { "J81", "5184", },
        "tag_septic_shock": { "R6521","78552" },
        "tag_aki": { "N17","584" },
        "tag_chf_chronic": { "I5022","I5032","I5042","42822","42832","42842" },
        "tag_hf_acute": { "I5021","I5031","I5041","42821","42831","42841" },
        "tag_acidosis_unspecified": { "E8720","2762" },
    }

    # 3) 일부 코드는 "카테고리(3자리/4자리) + 세부자리"가 섞여 있어 prefix 매칭도 지원
    PREFIX_MAP = {
        "tag_hearing_loss":        ["389"],                        # 389.xx
        "tag_sepsis":              ["038", "A40", "A41", "R652"],  # septicemia / sepsis / severe sepsis
        "tag_pneumonia":           ["J18", "482"],                 # J18.x, 482.x
        "tag_dka":                 ["2501", "E0810","E0910","E1010","E1110","E1310","E0811","E0911","E1011","E1111","E1311"],
        "tag_mi_acute":            ["I21","I22","410"],            # Acute MI
        "tag_metabolic_acidosis":  ["E872","2762"],                # Metabolic acidosis
        "tag_hyperkalemia":        ["E875","2767"],                # Hyperkalemia
        "tag_chf":                 ["I50","428"],                  # Heart failure
        "tag_pulmonary_edema":     ["J81","5184"],                 # Pulmonary edema
        "tag_chest_pain":          ["R07"],                        # Chest pain category
        "tag_depression":          ["F32","F33"],                  # Depressive disorders 범주 확장
        "tag_alcohol_intox":       ["F101, F100"],                 # Alcohol related, intoxication 포함
        "tag_alcohol_abuse":       ["3050"],                       # Alcohol abuse 범주
        "tag_aki":                 ["N17","584"],                  # AKI
        "tag_chf_chronic":         ["I5022","I5032","I5042","42822","42832","42842"],
        "tag_hf_acute":            ["I5021","I5031","I5041","42821","42831","42841"],
        "tag_acidosis_unspecified":["E8720","2762"],               # '미분류'를 우선 특정
    }

    # 4) 태그 생성
    #    - 정확일치(hit)와 prefix(hit)를 OR로 결합
    for tag, exact_set in MAP.items():
        if tag in df.columns:
            # 기존 컬럼이 있으면 덮어쓰기 (일관성 보장)
            df.drop(columns=[tag], inplace=True)
        exact_hit = code_norm.isin(exact_set) if len(exact_set) > 0 else pd.Series(False, index=df.index)
        prefix_hit = pd.Series(False, index=df.index)
        for p in PREFIX_MAP.get(tag, []):
            prefix_hit = prefix_hit | code_norm.str.startswith(p)
        df[tag] = (exact_hit | prefix_hit).astype(int)

    return df


# 시간 변화에 따른 특징 생성 (바이탈 변화 계산)
def add_time_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    first_vs_time, near_vs_time 기반으로 시간차와 바이탈 변화량/변화율/임상 플래그를 자동 생성.
    - delta_time_hr: (near - first) 시간(시간 단위; datetime이면 실시간 계산, 숫자면 단순 차)
    - delta_<feat>: near - first (절대 변화량)
    - rate_<feat>_per_hr: 시간당 변화율 (delta / 시간)
    - 임상 플래그: 악화 신호(저혈압/저산소/빈호흡/빈맥/고열)의 threshold 반영
    """
    # 0) 시간 컬럼 파싱 (문자열 → datetime)
    if "first_vs_time" in df.columns and df["first_vs_time"].dtype == object:
        df["first_vs_time"] = pd.to_datetime(df["first_vs_time"], errors="coerce")
    if "near_vs_time" in df.columns and df["near_vs_time"].dtype == object:
        df["near_vs_time"] = pd.to_datetime(df["near_vs_time"], errors="coerce")

    if "first_vs_time" not in df.columns or "near_vs_time" not in df.columns:
        return df  # 시간 컬럼 없으면 스킵

    # 1) delta_time_hr (float 보장)
    a, b = df["first_vs_time"], df["near_vs_time"]
    if np.issubdtype(a.dtype, np.datetime64) and np.issubdtype(b.dtype, np.datetime64):
        # datetime → Timedelta → seconds → hours
        df["delta_time_hr"] = ((b - a).dt.total_seconds() / 3600.0).astype(float)
    else:
        # 숫자일 경우: 단순 차이를 "시간"으로 간주(필요시 이후에/60 등 조정)
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        df["delta_time_hr"] = (b_num - a_num).astype(float)

    # 음수 방지(측정 1회나 정렬 오류)
    df["delta_time_hr"] = df["delta_time_hr"].clip(lower=0.0).fillna(0.0)

    # 2) first_/near_ 쌍 자동 탐색
    first_cols = [c for c in df.columns if c.endswith("_first")]
    pairs = []
    for fcol in first_cols:
        base = fcol[:-len("_first")]
        ncol = base + "_near"
        if ncol in df.columns:
            pairs.append((base, fcol, ncol))

    # 3) 변화량/변화율 생성 (+ 이상치 완화)
    for base, fcol, ncol in pairs:
        dcol = f"delta_{base}"
        rcol = f"rate_{base}_per_hr"

        # 절대 변화량
        df[dcol] = pd.to_numeric(df[ncol], errors="coerce") - pd.to_numeric(df[fcol], errors="coerce")

        # 시간당 변화율 (delta / 시간; 0시간이면 NaN→0)
        safe_dt = df["delta_time_hr"].where(df["delta_time_hr"] > 0, np.nan)
        rate = df[dcol] / safe_dt
        df[rcol] = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 윈저라이즈(0.1~99.9 분위)
        for c in (dcol, rcol):
            s = pd.to_numeric(df[c], errors="coerce")
            lo, hi = s.quantile(0.001), s.quantile(0.999)
            df[c] = s.clip(lo, hi).fillna(0.0)

    # 4) 임상 플래그(악화 신호)
    def _has(c): return c in df.columns
    def _mk_flag(name, cond): df[name] = cond.astype(int)

    if _has("delta_sbp"):   _mk_flag("flag_worse_hypotension_sbp", df["delta_sbp"] <= -20)
    if _has("delta_map"):   _mk_flag("flag_worse_hypotension_map", df["delta_map"] <= -10)
    if _has("delta_spo2"):  _mk_flag("flag_worse_spo2",            df["delta_spo2"] <= -3)
    if _has("delta_rr"):    _mk_flag("flag_worse_tachypnea",       df["delta_rr"]   >= 5)
    if _has("delta_hr"):    _mk_flag("flag_worse_tachycardia",     df["delta_hr"]   >= 20)
    if _has("delta_temp"):  _mk_flag("flag_worse_fever",           df["delta_temp"] >= 1.0)

    # 5) 측정 1회(=시간차 0) → rate_* 0, flag_* 0
    zero_dt = (df["delta_time_hr"] <= 0)
    if zero_dt.any():
        rate_cols = [c for c in df.columns if c.startswith("rate_") and c.endswith("_per_hr")]
        flag_cols = [c for c in df.columns if c.startswith("flag_worse_")]
        if rate_cols: df.loc[zero_dt, rate_cols] = 0.0
        if flag_cols: df.loc[zero_dt, flag_cols] = 0

    return df


# 학습에 필요한 데이터 준비 
def data_preparation():

    base_csv = os.path.join("./", "medi_data.csv")
    df = pd.read_csv(base_csv, low_memory=False)
    
    # ICD 태깅
    df = add_er_icd_tags(df, icd_col="primary_icd_code", ver_col="primary_icd_version")
    
    # 시간 변화에 따른 특징 생성
    df = add_time_delta_features(df)

    # 0) has_* → tag_* 자동 변환 (멀티라벨 라벨 만들기)
    has_cols = [c for c in df.columns if c.startswith("has_")]

    for c in has_cols:
        newc = "tag_" + c[len("has_"):]           # has_sepsis -> tag_sepsis
        df[newc] = (df[c].astype(float) > 0).astype(int)

    tag_cols = sorted([c for c in df.columns if c.startswith("tag_")])

    # 0-1) 라벨이 하나도 없으면 바로 에러
    icu_col = "icu_transfer" if "icu_transfer" in df.columns else None
    hh_col  = "HH" if "HH" in df.columns else None
    if len(tag_cols) == 0 and (icu_col is None) and (hh_col is None):
        raise ValueError("라벨이 없습니다. has_* → tag_* 생성 또는 icu_transfer/HH가 필요합니다.")

    # 1) 입력/제외 컬럼 정의
    drop_cols = set(tag_cols + has_cols + [
        "subject_id","hadm_id","ed_stay_id",
        "first_vs_time","near_vs_time","lab_charttime",
        "other_disease","hosp_primary_long_title",
        "primary_icd_code","primary_icd_version"
    ])

    if icu_col: drop_cols.add(icu_col)
    if hh_col: drop_cols.add(hh_col)

    # 2) X 선택: *_z(표준화 수치) + 약물 투여 플래그
    z_cols = [c for c in df.columns if c.endswith("_z")]
    drug_cols_all = ["norepinephrine","epinephrine","dopamine","dobutamine","phenylephrine","vasopressin","milrinone"]
    drug_cols = [c for c in drug_cols_all if c in df.columns]

    USE_DRUG_FLAGS = True
    candidate_X = z_cols + (drug_cols if USE_DRUG_FLAGS else [])
    
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    rate_cols  = [c for c in df.columns if c.startswith("rate_") and c.endswith("_per_hr")]
    flag_cols  = [c for c in df.columns if c.startswith("flag_worse_")]
    time_cols  = ["delta_time_hr"] if "delta_time_hr" in df.columns else []
    candidate_X += delta_cols + rate_cols + flag_cols + time_cols

    # 혹시 비어버릴 경우를 대비해, 드롭 목록에 없다면 원래 연속형도 조금 추가
    if len(candidate_X) == 0:
        candidate_X = [c for c in df.columns if c not in drop_cols and df[c].dtype != "object"]

    X = (df[candidate_X]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0))
    
    # 비수치 컬럼 제거(안전장치)
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:    
        X = X.drop(columns=non_numeric_cols)

    # 3) 타깃 구성: 멀티라벨(tag_*), + ICU 이진, + HH 회귀(마스크)
    y_list = [df[c].astype(int).values for c in tag_cols]
    output_names = tag_cols.copy()

    if icu_col:
        y_list.append(df[icu_col].astype(int).values)
        output_names.append(icu_col)

    def _make_class_weights(y):
        # y: (N,) binary
        p = float(np.mean(y))
        if p <= 0 or p >= 1: return np.ones_like(y, dtype=float)
        w_pos = 0.5 / p
        w_neg = 0.5 / (1 - p)
        return np.where(y==1, w_pos, w_neg)

    # data_preparation() 끝부분, sample_weights 구성 전에:
    task_weights = [_make_class_weights(y) for y in y_list]
    sample_weights = np.column_stack(task_weights)  # 회귀 태스크는 후에 그대로 1 또는 mask 적용

    if hh_col:
        y_hh_raw = df[hh_col].astype(float).values
        mask_hh = (df[icu_col].astype(int).values == 1).astype(float) if icu_col else (~np.isnan(y_hh_raw)).astype(float)
        y_hh = np.nan_to_num(y_hh_raw, nan=0.0)

        # 스케일링
        y_hh = np.log1p(y_hh)
        y_list.append(y_hh)
        output_names.append("transfer_time")

        # 태스크별 sample_weight (태그/ICU=1, HH는 mask)
        sw_list = [np.ones(len(X), dtype=float) for _ in output_names]
        sw_list[-1] = mask_hh
        sample_weights = np.column_stack(sw_list)

    # --- 멀티라벨 분할 (train/valid/test) ---
    Y_all = np.column_stack([y.astype(int) for y in y_list])
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, temp_idx = next(msss.split(X.values, Y_all))

    # Train / Temp
    X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
    y_trains = [y[train_idx] for y in y_list]
    y_temps  = [y[temp_idx]  for y in y_list]
    SW_train, SW_temp = sample_weights[train_idx], sample_weights[temp_idx]

    # Temp → Valid / Test (50:50)
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    valid_idx, test_idx = next(msss2.split(X_temp.values, np.column_stack(y_temps)))

    X_valid, X_test = X_temp.iloc[valid_idx], X_temp.iloc[test_idx]
    y_valids = [y[valid_idx] for y in y_temps]
    y_tests  = [y[test_idx]  for y in y_temps]
    SW_valid, SW_test = SW_temp[valid_idx], SW_temp[test_idx]

    # --- sample_weights 분리 ---
    train_sw = [SW_train[:, i] for i in range(SW_train.shape[1])]
    valid_sw = [SW_valid[:, i] for i in range(SW_valid.shape[1])]
    test_sw  = [SW_test[:,  i] for i in range(SW_test.shape[1])]

    MIN_POS = 3
    keep_idx = []
    for i, name in enumerate(output_names):
        if name == 'transfer_time':   # 회귀 태스크는 패스
            keep_idx.append(i)
            continue
        pos_tr = int(np.sum(y_trains[i]))
        pos_va = int(np.sum(y_valids[i]))
        if pos_tr >= MIN_POS and pos_va >= MIN_POS:
            keep_idx.append(i)
        else:
            print(f"[DROP TAG] {name}: train_pos={pos_tr}, valid_pos={pos_va}")

    # 필터 적용
    output_names = [output_names[i] for i in keep_idx]
    y_trains     = [y_trains[i]     for i in keep_idx]
    y_valids     = [y_valids[i]     for i in keep_idx]
    y_tests      = [y_tests[i]      for i in keep_idx]

    if sample_weights is not None:
        # sw도 같은 순서로 줄여야 함
        train_sw = [train_sw[i] for i in keep_idx]
        valid_sw = [valid_sw[i] for i in keep_idx]
        if test_sw is not None:
            test_sw  = [test_sw[i]  for i in keep_idx]

    # 5) output_info: (units, name) — 분류(1, sigmoid) / 회귀(1, linear)
    output_info = [(1, name) for name in output_names]

    return (X_train, y_trains,
            X_valid, y_valids,
            X_test,  y_tests,
            output_info,
            (train_sw, valid_sw, test_sw),
            {"df_raw": df})


# 모델 학습
def main():
    # 모델 생성 전에 session clear
    tf.keras.backend.clear_session()

    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info, sw_tuple, raw_ctx = data_preparation()
    sw_train, sw_valid, sw_test = sw_tuple
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    # 입력층
    input_layer = Input(shape=(num_features,))

    # 1) 동적 태스크 수
    num_tasks = len(output_info)

    print("num_tasks:", len(output_info), "names:", [n for _, n in output_info])

    if num_tasks == 0:
        raise RuntimeError("태스크가 0개입니다. data_preparation()에서 라벨 생성 확인 필요.")

    # 2) MMoE: 태스크 수를 동적으로
    MMOE_UNITS = 16
    mmoe_layers = MMoE(
        units=MMOE_UNITS,
        num_experts=8,
        num_tasks=num_tasks
    )(input_layer)

    if isinstance(mmoe_layers, (list, tuple)):
        mmoe_layers = list(mmoe_layers)
    else:
        mmoe_layers = [mmoe_layers]

    print("MMoE outputs shapes:", [t.shape for t in mmoe_layers])

    # 태스크별 타워 + 출력층
    output_layers = []

    # 3) 출력층: 질병/ICU는 sigmoid(1유닛), HH는 linear(1유닛)
    for i, task_layer in enumerate(mmoe_layers):
      units_i, name_i = output_info[i]

      # 1) 태스크별 attention 
      att_i = FeatureAttention(name=f"att_{name_i}")(input_layer)     

      # 2) attention 요약을 작은 차원으로 압축해 결합 
      att_i_proj = Dense(16, activation='relu',
                        kernel_initializer=VarianceScaling(),
                        name=f"att_proj_{name_i}")(att_i)            

      # 3) MMoE 태스크 표현(task_layer)와 결합
      merged = Concatenate(name=f"concat_{name_i}")([task_layer, att_i_proj])

      tower_layer = Dense(
          units=8,
          activation='relu',
          kernel_initializer=VarianceScaling(),
          name=f"tower_{name_i}"
      )(merged)

      if name_i == 'transfer_time':
          output_layer = Dense(
              units=1,
              name=name_i,
              activation='linear',
              kernel_initializer=VarianceScaling()
          )(tower_layer)
      else:
          output_layer = Dense(
              units=1,
              name=name_i,
              activation='sigmoid',
              kernel_initializer=VarianceScaling()
          )(tower_layer)

      output_layers.append(output_layer)

    # 모델 컴파일
    model = Model(inputs=[input_layer], outputs=output_layers)

    # 4) 손실/지표를 태스크별 dict로 구성
    losses = {name: ('mse' if name=='transfer_time' else 'binary_crossentropy') for (_, name) in output_info}
    metrics = {name: ([] if name=='transfer_time' else [AUC(name='AUC')]) for (_, name) in output_info}
    loss_weights = {name: (0.2 if name == 'transfer_time' else 1.0)
                for (_, name) in output_info}

    adam_optimizer = Adam()
    model.compile(
        loss=losses,
        optimizer=adam_optimizer,
        metrics=metrics,
        loss_weights=loss_weights
    )

    # 모델 구조 출력
    model.summary()

    # 5) 콜백 제거 (회귀 섞여서 기존 ROC 콜백은 에러 위험)
    # 변경 (sample_weight/val용 가중치 전달)
    if sw_train is not None:
        model.fit(
            x=train_data,
            y=train_label,
            validation_data=(validation_data, validation_label, sw_valid),
            sample_weight=sw_train,
            epochs=100,
            batch_size=256
        )
    else:
        model.fit(
            x=train_data,
            y=train_label,
            validation_data=(validation_data, validation_label),
            epochs=100,
            batch_size=256
        )

    task_names = [name for _, name in output_info]


    #########################################################################################

    # 1) 검증셋으로 각 태스크 최적 임계값(threshold) 찾기 (F1 최대)
    pred_valid = model.predict(validation_data, batch_size=1024)
    best_thr = {}

    for i, name in enumerate(task_names):
        if name == "transfer_time":
            continue  # 회귀는 threshold 불필요
        yv = validation_label[i].ravel()
        pv = pred_valid[i].ravel()
        p, r, t = precision_recall_curve(yv, pv)
        f1 = (2*p*r)/(p+r+1e-9)
        thr = 0.5 if len(t) == 0 else t[np.argmax(f1)]
        best_thr[name] = float(thr)


    # 2) 테스트셋 성능 지표
    pred_test = model.predict(test_data, batch_size=1024)

    print("\n=== Test metrics ===")

    for i, name in enumerate(task_names):

        pt = pred_test[i].ravel()

        if name == "transfer_time":

            # ICU 간 케이스만 회귀 평가 (log1p 스케일 복원)
            if "icu_transfer" in task_names:
                icu_idx = task_names.index("icu_transfer")
                icu_mask = (test_label[icu_idx].ravel() == 1)
            else:
                icu_mask = np.ones_like(pt, dtype=bool)

            yt = test_label[i].ravel()
            yt_hours = np.expm1(yt)              # log1p 역변환
            pt_hours = np.expm1(pt)

            if icu_mask.sum() > 0:
                mae = mean_absolute_error(yt_hours[icu_mask], pt_hours[icu_mask])
                print(f"[{name}] MAE (hours) on ICU cases: {mae:.3f}  (N={icu_mask.sum()})")
            else:
                print(f"[{name}] No ICU cases in test set to evaluate.")
        else:
            yt = test_label[i].ravel()

            auroc = roc_auc_score(yt, pt) if len(np.unique(yt)) > 1 else float("nan")
            ap    = average_precision_score(yt, pt) if len(np.unique(yt)) > 1 else float("nan")
            thr   = best_thr.get(name, 0.5)
            yhat  = (pt >= thr).astype(int)

            tn, fp, fn, tp = confusion_matrix(yt, yhat).ravel()

            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)

            print(f"[{name}] AUROC={auroc:.3f}  AP={ap:.3f}  thr={thr:.3f}  "
                  f"Prec={prec:.3f}  Rec={rec:.3f}  (TP={tp}, FP={fp}, FN={fn}, TN={tn})")

    # 3) 샘플 환자 예측 보기 (테스트 상위 5명)
    n_show = 5
    X_show = test_data.iloc[:n_show]
    pred_show = model.predict(X_show, batch_size=32)

    # 분류 확률 표
    rows = []

    for i in range(n_show):
        row = {"row_idx": int(X_show.index[i])}

        for j, name in enumerate(task_names):

            if name == "transfer_time":
                continue

            row[name] = float(pred_show[j][i, 0])

        rows.append(row)

    df_probs = pd.DataFrame(rows).set_index("row_idx").round(4)

    print("\n=== Sample classification probabilities (first 5 test rows) ===")
    print(df_probs.to_string())

    # ICU 확률 & 이송시간 함께 보기
    out = []
    icu_idx = task_names.index("icu_transfer") if "icu_transfer" in task_names else None
    tt_idx  = task_names.index("transfer_time") if "transfer_time" in task_names else None

    for i in range(n_show):
        item = {"row_idx": int(X_show.index[i])}

        if icu_idx is not None:
            item["icu_prob"] = float(pred_show[icu_idx][i, 0])

        if tt_idx is not None:
            item["transfer_time_hours_pred"] = float(np.expm1(pred_show[tt_idx][i, 0]))

        out.append(item)

    df_icu_tt = pd.DataFrame(out).set_index("row_idx").round(3)

    print("\n=== ICU prob & Predicted transfer time (hours) ===")
    print(df_icu_tt.to_string())
    
    ############################################
    # Guideline-based evidence for TOP disease #
    ############################################
    cls_task_idx = [i for i, name in enumerate(task_names) if name != "transfer_time"]
    raw_df = raw_ctx["df_raw"]
    colmap = _pick_raw_feature_columns(raw_df)

    print("\n=== Guideline-based evidence for TOP disease (multi-disease) ===")
    for i in range(n_show):
        ridx = int(X_show.index[i])

        prob_map = {task_names[j]: float(pred_show[j][i, 0]) for j in cls_task_idx}
        top_task, top_prob = max(prob_map.items(), key=lambda kv: kv[1])

        print(f"\n[Row {ridx}] Top task: {top_task}  (p={top_prob:.3f})")

        if ridx not in raw_df.index:
            print("  • 원시 DF에서 해당 행을 찾을 수 없습니다(인덱스 불일치).")
            continue

        row_raw = raw_df.loc[ridx]
        tt = top_task.lower()

        def _print_used(vals: Dict[str, Any]) -> str:
            out = []
            for k, v in vals.items():
                if v is None: out.append(f"{k}=?")
                elif isinstance(v, float) or isinstance(v, int): out.append(f"{k}={v:.3f}" if isinstance(v,float) else f"{k}={v}")
                else: out.append(f"{k}={v}")
            return ", ".join(out)

        if tt in {"tag_sepsis","sepsis","has_sepsis"}:
            flags, vals = sepsis_guideline_evidence_for_row(row_raw, colmap)
        elif tt in {"tag_septic_shock","septic_shock","tag_shock_septic"} or tt in {"tag_shock_general","shock_general","shock"}:
            flags, vals = septic_shock_guideline_evidence_for_row(row_raw, colmap)
        elif tt in {"tag_dka","dka"}:
            flags, vals = dka_guideline_evidence_for_row(row_raw, colmap)
        elif tt in {"tag_aki","aki"}:
            flags, vals = aki_guideline_evidence_for_row(row_raw, colmap)
        elif tt in {"tag_metabolic_acidosis","metabolic_acidosis","tag_acidosis","acidosis"}:
            flags, vals = meta_acid_guideline_evidence_for_row(row_raw, colmap)
        elif tt in {"tag_hyperkalemia","hyperkalemia"}:
            flags, vals = hyperk_guideline_evidence_for_row(row_raw, colmap)
        elif tt in {"tag_pneumonia","pneumonia","cap"}:
            flags, vals = pneumonia_guideline_evidence_for_row(row_raw, colmap)
        elif tt in {"tag_chf_chronic","tag_hf_acute","chf","hf","acute_hf"}:
            flags, vals = chf_guideline_evidence_for_row(row_raw, colmap)
        elif tt in {"tag_pulmonary_edema","pulmonary_edema"}:
            flags, vals = pulm_edema_guideline_evidence_for_row(row_raw, colmap)
        elif tt in {"tag_mi_acute","mi_acute","mi","udmi"}:
            flags, vals = mi_udmi_guideline_evidence_for_row(row_raw, colmap)
        else:
            print("  • 해당 질환에 대한 가이드라인 모듈이 아직 연결되지 않았습니다.")
            continue

        print(f"  • Raw used: {_print_used(vals)}")
        print("  • Applied criteria:")
        for tline in flags:
            print(f"    - {tline}")

################################
''' Attention Layer '''
class FeatureAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = None
        self.b = None

    def build(self, input_shape):
        d_features = int(input_shape[-1])
        # 점수 e = tanh(x W + b), alpha = softmax(e), attended = x * alpha
        self.W = self.add_weight(
            name="W",
            shape=(d_features, d_features),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(d_features,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.linalg.matmul(x, self.W) + self.b)   # (B, F)
        alpha = tf.nn.softmax(e, axis=-1)                   # (B, F)
        return x * alpha                                    # (B, F)

    def compute_output_shape(self, input_shape):
        # 입력과 동일 shape 반환
        return input_shape



####################
#### 가이드 라인 ####

def _pick_raw_feature_columns(df: pd.DataFrame) -> dict[str, Any]:
    """
    데이터셋 변수명 기준으로 가이드라인 평가에 필요한 컬럼 자동 매핑.
    """
    def _first(cands): 
        for c in cands:
            if c in df.columns: return c
        return None

    # Vital / Lab / Imaging
    sbp   = _first(["sbp","sbp_mean","sbp_first","sbp_min","sbp_max"])
    mapc  = _first(["map","map_mean","map_first","map_min","map_max","mean_arterial_pressure","mean_arterial_bp"])
    rr    = _first(["resp_rate","resp_rate_mean","resp_rate_first","resp_rate_min","resp_rate_max"])
    gcs   = _first(["gcs","gcs_mean","gcs_first","gcs_min","gcs_max"])
    hr    = _first(["hr","heart_rate","pulse","hr_mean","heart_rate_mean"])
    hb    = _first(["hb","hemoglobin","hgb","hb_mean"])
    spo2  = _first(["spo2","o2_sat","spo2_mean"])
    tempc = _first(["temp","temperature","body_temp","temp_mean"])
    wbc   = _first(["wbc","wbc_count","white_blood_cells"])
    na    = _first(["sodium","na"])
    cl    = _first(["chloride","cl"])
    hco3  = _first(["hco3","bicarbonate","tco2"])
    lact  = _first(["lactate","lactate_mean","lactate_first","lactate_min","lactate_max"])
    k     = _first(["potassium","k"])
    glu   = _first(["glucose","glu","blood_glucose"])
    bhb   = _first(["beta_hydroxybutyrate","bhb","beta_hb"])
    uket  = _first(["urine_ketone","uketone","urine_ketone_strip"])
    ph    = _first(["ph","arterial_ph","venous_ph"])
    bnp   = _first(["bnp"])
    ntp   = _first(["ntprobnp","nt_pro_bnp"])
    cxr_infil = _first(["cxr_infiltrate","cxr_infil","pna_infiltrate"])
    cxr_bilat = _first(["cxr_bilateral","cxr_bilat"])
    cxr_cardm = _first(["cxr_cardiomegaly","cardiomegaly"])
    congestion = _first(["congestion","pulmonary_congestion","rales_crackles"])
    troponin = _first(["troponin","trop","troponin_i","troponin_t"])
    trop_url99 = _first(["troponin_url99","trop_url99","troponin_uln99"])
    ecg_iscm = _first(["ecg_ischemia","ecg_st_depression","ecg_st_elevation","ischemic_evidence"])
    dm_hist = _first(["dm_history","diabetes_history","dm","t2dm","t1dm"])

    # AKI / 소변량
    cr_now = _first(["creatinine","creat","serum_creatinine"])
    cr_prev48 = _first(["creatinine_prev_48h","creat_prev_48h","cr_48h_ago"])
    cr_delta48 = _first(["creatinine_delta_48h","cr_delta_48h"])
    cr_base7d = _first(["creatinine_baseline_7d","cr_baseline_7d","creatinine_min_7d"])
    uo_mlkgh = _first(["uo_ml_kg_h","urine_ml_kg_h","uop_ml_kg_h"])
    uo_ml_h  = _first(["uo_ml_h","urine_ml_h","uop_ml_h"])
    weight_kg = _first(["weight","weight_kg","body_weight_kg"])

    # 승압제(여러 컬럼 OR)
    vaso_candidates = [
        "vasopressor","vasopressor_flag",
        "norepinephrine","epinephrine","dopamine","dobutamine",
        "phenylephrine","vasopressin","milrinone"
    ]
    vaso_cols = [c for c in vaso_candidates if c in df.columns]

    return {
        "SBP": sbp, "MAP": mapc, "RR": rr, "GCS": gcs, "HR": hr, "Hb": hb, "SpO2": spo2, "Temp": tempc, "WBC": wbc,
        "Na": na, "Cl": cl, "HCO3": hco3, "Lactate": lact, "K": k,
        "Glucose": glu, "BHB": bhb, "Uketone": uket, "pH": ph,
        "BNP": bnp, "NTproBNP": ntp, "CXR_infiltrate": cxr_infil, "CXR_bilateral": cxr_bilat,
        "CXR_cardiomegaly": cxr_cardm, "congestion": congestion,
        "Troponin": troponin, "Troponin_URL99": trop_url99, "ECG_ischemia": ecg_iscm,
        "DM_history": dm_hist,
        "Cr_now": cr_now, "Cr_prev48": cr_prev48, "Cr_delta48": cr_delta48,
        "Cr_baseline7d": cr_base7d, "UO_mLkg_h": uo_mlkgh, "UO_mL_h": uo_ml_h, "Weight_kg": weight_kg,
        "VasoCols": vaso_cols if vaso_cols else None,
    }

# ---------- 공통 ----------
def _get_float(row: pd.Series, col: Optional[str]) -> Optional[float]:
    if col is None or col not in row.index: return None
    try:
        v = float(row[col])
        if np.isnan(v): return None
        return v
    except Exception:
        return None

def _row_has_vasopressor(row: pd.Series, vaso_cols: Optional[List[str]]) -> Optional[int]:
    if not vaso_cols: return None
    any_seen = False
    any_pos = False
    for c in vaso_cols:
        if c not in row.index: continue
        any_seen = True
        val = row[c]
        # 숫자: >0 → 투여
        try:
            fv = float(val)
            if not np.isnan(fv) and fv > 0: any_pos = True; break
            continue
        except Exception:
            pass
        # 문자: yes/true/1 등
        if isinstance(val, str) and val.strip().lower() in {"1","y","yes","true","t"}:
            any_pos = True; break
    if not any_seen: return None
    return 1 if any_pos else 0

def _row_yesno(row: pd.Series, col: Optional[str]) -> Optional[int]:
    if col is None or col not in row.index: return None
    v = row[col]
    try:
        fv = float(v)
        if not np.isnan(fv): return 1 if fv != 0 else 0
    except Exception:
        pass
    if isinstance(v, str) and v.strip() != "":
        s = v.strip().lower()
        if s in {"1","y","yes","true","t","positive","pos"}: return 1
        if s in {"0","n","no","false","f","negative","neg"}: return 0
    return None

def _parse_urine_ketone_level(v: Any) -> Optional[float]:
    """ '2+' → 2, 'trace'→0.5 등 느슨히 파싱 """
    if v is None: return None
    try:
        fv = float(v); 
        if not np.isnan(fv): return fv
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"neg","negative","none"}: return 0.0
        if s in {"trace","tr"}: return 0.5
        m = re.match(r"(\d+)\s*\+?", s)
        if m: return float(m.group(1))
        if s in {"+","+1"}: return 1.0
        if s in {"++","+2","2+"}: return 2.0
        if s in {"+++","+3","3+"}: return 3.0
        if s in {"++++","+4","4+"}: return 4.0
    return None

def _compute_anion_gap(row: pd.Series, colmap: Dict[str, Any]) -> Optional[float]:
    na = _get_float(row, colmap.get("Na"))
    cl = _get_float(row, colmap.get("Cl"))
    hco3 = _get_float(row, colmap.get("HCO3"))
    if None in (na, cl, hco3): return None
    return na - (cl + hco3)

def _uo_mlkgh(row: pd.Series, colmap: Dict[str, Any]) -> Optional[float]:
    uo_direct = _get_float(row, colmap.get("UO_mLkg_h"))
    if uo_direct is not None: return uo_direct
    # 대안: mL/h 와 체중이 있으면 환산
    uo_h = _get_float(row, colmap.get("UO_mL_h"))
    wt = _get_float(row, colmap.get("Weight_kg"))
    if None in (uo_h, wt) or wt == 0: return None
    return uo_h / wt

# 1. Sepsis
def sepsis_guideline_evidence_for_row(row: pd.Series,
                                      colmap: Dict[str, Optional[str]]) -> Tuple[List[str], Dict[str, Optional[float]]]:
    """
    한 명의 환자(row)에 대해 Sepsis 가이드라인 근거를 평가하고,
    적용된 근거 텍스트 리스트와 사용된 값들을 반환합니다.
    - qSOFA = (RR>=22) + (SBP<=100) + (GCS<15)
    - Lactate > 2 mmol/L : 예후불량 표지
    """
    vals = {}
    def _get(name):
        col = colmap.get(name)
        if col is None or col not in row.index:
            return None
        try:
            v = float(row[col])
            if np.isnan(v):
                return None
            return v
        except Exception:
            return None

    sbp = _get("SBP"); rr = _get("RR"); gcs = _get("GCS"); lact = _get("Lactate")
    vals = {"SBP": sbp, "RR": rr, "GCS": gcs, "Lactate": lact}

    flags = []
    q_rr   = int(rr  is not None and rr  >= 22)
    q_sbp  = int(sbp is not None and sbp <= 100)
    q_gcs  = int(gcs is not None and gcs < 15)
    known_components = sum(x is not None for x in [rr, sbp, gcs])

    # 개별 조건 문구
    if rr   is not None and rr  >= 22:  flags.append(f"호흡수(RR) ≥ 22/min (현재 {rr:.1f})")
    if sbp  is not None and sbp <= 100: flags.append(f"수축기혈압(SBP) ≤ 100 mmHg (현재 {sbp:.1f})")
    if gcs  is not None and gcs < 15:   flags.append(f"GCS < 15 (현재 {gcs:.1f})")

    # qSOFA 요약
    if known_components >= 2:  # 구성요소가 2개 이상 있으면 qSOFA 합산
        qsofa = q_rr + q_sbp + q_gcs
        if qsofa >= 2:
            flags.append(f"qSOFA = {qsofa} (≥2 → 고위험)")
        else:
            flags.append(f"qSOFA = {qsofa}")
    else:
        flags.append("qSOFA 계산 불충분(구성요소 부족)")

    # Lactate
    if lact is not None:
        if lact > 2:
            flags.append(f"Lactate > 2 mmol/L (현재 {lact:.2f}) — 예후 불량 표지")
        else:
            flags.append(f"Lactate ≤ 2 mmol/L (현재 {lact:.2f}) — 단독 배제 불가")

    # 주의 문구
    flags.append("주의: qSOFA 단독 진단 비권고. 확진은 감염 + ΔSOFA≥2 필요(라크테이트 정상이어도 배제 불가).")

    return flags, vals

# 2. Septic shock
def septic_shock_guideline_evidence_for_row(
    row: pd.Series,
    colmap: Dict[str, Optional[Union[str, List[str]]]]
) -> Tuple[List[str], Dict[str, Optional[float]]]:
    """
    Septic shock 기준:
    - Sepsis + MAP < 65 + vasopressor 필요 + Lactate > 2
    여기서는 모델 단계에서 'Sepsis 의심/진단 상황'을 전제로, MAP/vasopressor/Lactate를 평가.
    """
    map_val = _get_float(row, colmap.get("MAP"))
    lact = _get_float(row, colmap.get("Lactate"))
    vaso = _row_has_vasopressor(row, colmap.get("VasoCols"))

    flags = []
    used = {"MAP": map_val, "Lactate": lact, "Vasopressor": float(vaso) if vaso is not None else None}

    # 개별 조건 문구
    if map_val is not None:
        if map_val < 65:
            flags.append(f"MAP < 65 mmHg (현재 {map_val:.1f})")
        else:
            flags.append(f"MAP ≥ 65 mmHg (현재 {map_val:.1f})")
    else:
        flags.append("MAP 값 없음")

    if vaso is None:
        flags.append("승압제 투여 여부 확인 불가(관련 컬럼 부재)")
    elif vaso == 1:
        flags.append("승압제 필요/투여 중 (하나 이상의 관련 약물 컬럼 양성)")
    else:
        flags.append("승압제 투여 신호 없음")

    if lact is not None:
        if lact > 2:
            flags.append(f"Lactate > 2 mmol/L (현재 {lact:.2f})")
        else:
            flags.append(f"Lactate ≤ 2 mmol/L (현재 {lact:.2f})")
    else:
        flags.append("Lactate 값 없음")

    # 종합 로직 평가 (Sepsis 전제)
    cond_map = (map_val is not None and map_val < 65)
    cond_vaso = (vaso == 1) if vaso is not None else False
    cond_lact = (lact is not None and lact > 2)

    if cond_map and cond_vaso and cond_lact:
        flags.append("⇒ Septic shock 논리 기준 충족 (Sepsis 전제).")
    else:
        missing = []
        if not cond_map:  missing.append("MAP<65")
        if not cond_vaso: missing.append("승압제 필요")
        if not cond_lact: missing.append("Lactate>2")
        if missing:
            flags.append("⇒ Septic shock 기준 일부 미충족: " + ", ".join(missing))

    # 주의 문구
    flags.append("주의: Septic shock은 수액 후에도 저혈압 지속 + 승압제 필요 상황. 저혈량성 등 다른 쇼크와 감별 필요.")

    return flags, used

# 3. DKA
def dka_guideline_evidence_for_row(row: pd.Series, colmap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    glu = _get_float(row, colmap.get("Glucose"))
    dm  = _row_yesno(row, colmap.get("DM_history"))
    bhb = _get_float(row, colmap.get("BHB"))
    ukt_col = colmap.get("Uketone")
    ukt = _parse_urine_ketone_level(row[ukt_col]) if (ukt_col and ukt_col in row.index) else None
    ph  = _get_float(row, colmap.get("pH"))
    hco3 = _get_float(row, colmap.get("HCO3"))

    flags = []
    used = {"Glucose": glu, "DM": dm, "BHB": bhb, "Uketone": ukt, "pH": ph, "HCO3": hco3}

    c1 = ((glu is not None and glu >= 200) or (dm == 1))
    c2 = ((bhb is not None and bhb >= 3.0) or (ukt is not None and ukt >= 2))
    c3 = ((ph is not None and ph < 7.30) or (hco3 is not None and hco3 < 18))

    if glu is not None: flags.append(f"Glucose {'≥' if glu>=200 else '<'}200 (현재 {glu:.1f} mg/dL)")
    else: flags.append("Glucose 값 없음")
    if dm is not None: flags.append("DM 병력 있음" if dm==1 else "DM 병력 없음")
    else: flags.append("DM 병력 정보 없음")

    if bhb is not None: flags.append(f"βHB {'≥' if bhb>=3.0 else '<'}3.0 (현재 {bhb:.2f} mmol/L)")
    else: flags.append("βHB 값 없음")
    if ukt is not None: flags.append(f"요 케톤 {'≥' if ukt>=2 else '<'} 2+ (현재 {ukt})")
    else: flags.append("요 케톤 값 없음")

    if ph is not None: flags.append(f"pH {'<' if ph<7.30 else '≥'}7.30 (현재 {ph:.2f})")
    else: flags.append("pH 값 없음")
    if hco3 is not None: flags.append(f"HCO₃ {'<' if hco3<18 else '≥'}18 (현재 {hco3:.1f} mmol/L)")
    else: flags.append("HCO₃ 값 없음")

    if c1 and c2 and c3:
        flags.append("⇒ DKA 기준 충족.")
    else:
        miss = []
        if not c1: miss.append("Glu≥200 또는 DM=1")
        if not c2: miss.append("βHB≥3.0 또는 Uketone≥2+")
        if not c3: miss.append("pH<7.30 또는 HCO₃<18")
        flags.append("⇒ DKA 기준 일부 미충족: " + ", ".join(miss))
    flags.append("주의: 정상혈당성 DKA 존재, AG는 필수 기준에서 제외됨. 케톤 정량 권장.")
    return flags, used

# 4. AKI
def aki_guideline_evidence_for_row(row: pd.Series, colmap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    cr_now = _get_float(row, colmap.get("Cr_now"))
    cr_prev48 = _get_float(row, colmap.get("Cr_prev48"))
    cr_delta48 = _get_float(row, colmap.get("Cr_delta48"))
    cr_base7d = _get_float(row, colmap.get("Cr_baseline7d"))
    uo = _uo_mlkgh(row, colmap)

    flags = []
    used = {"Cr_now": cr_now, "Cr_prev48": cr_prev48, "Cr_delta48": cr_delta48, "Cr_baseline7d": cr_base7d, "UO_mLkg_h": uo}

    # ΔCr 기준
    cond_delta = False
    if cr_delta48 is not None:
        cond_delta = cr_delta48 >= 0.3
        flags.append(f"ΔCr(48h) {'≥' if cr_delta48>=0.3 else '<'}0.3 (현재 {cr_delta48:.3f} mg/dL)")
    elif None not in (cr_now, cr_prev48):
        delta = cr_now - cr_prev48
        cond_delta = delta >= 0.3
        flags.append(f"ΔCr(48h) 추정 {'≥' if delta>=0.3 else '<'}0.3 (현재 {delta:.3f} mg/dL)")
    else:
        flags.append("ΔCr(48h) 계산 불가")

    # 1.5x 기준
    cond_ratio = False
    if None not in (cr_now, cr_base7d) and cr_base7d > 0:
        ratio = cr_now / cr_base7d
        cond_ratio = ratio >= 1.5
        flags.append(f"Cr/기저(7d) {'≥' if ratio>=1.5 else '<'}1.5 (현재 {ratio:.2f}x)")
    else:
        flags.append("Cr/기저(7d) 평가 불가")

    # 소변량 기준
    cond_uo = False
    if uo is not None:
        cond_uo = uo < 0.5  # (≥6h 가정; 시간 정보 없으면 보수적 코멘트)
        flags.append(f"UO {'<' if uo<0.5 else '≥'}0.5 mL/kg/h (현재 {uo:.2f})")
    else:
        flags.append("UO(mL/kg/h) 평가 불가")

    if cond_delta or cond_ratio or cond_uo:
        flags.append("⇒ AKI 기준 충족(세 기준 중 하나 이상).")
    else:
        flags.append("⇒ AKI 기준 미충족(모든 기준 불충족).")
    flags.append("주의: 소변량 측정 정확성 및 급성/만성 감별 필요.")
    return flags, used

# 5. Metabolic acidosis
def meta_acid_guideline_evidence_for_row(row: pd.Series, colmap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    ph = _get_float(row, colmap.get("pH"))
    hco3 = _get_float(row, colmap.get("HCO3"))
    lact = _get_float(row, colmap.get("Lactate"))
    ag = _compute_anion_gap(row, colmap)

    flags = []
    used = {"pH": ph, "HCO3": hco3, "AG": ag, "Lactate": lact}

    cond_ph = (ph is not None and ph < 7.35)
    cond_hco3 = (hco3 is not None and hco3 < 20)
    cond_ag = (ag is not None and ag > 12)
    cond_lac = (lact is not None and lact > 2)

    if ph is not None: flags.append(f"pH {'<' if cond_ph else '≥'}7.35 (현재 {ph:.2f})")
    else: flags.append("pH 값 없음")
    if hco3 is not None: flags.append(f"HCO₃ {'<' if cond_hco3 else '≥'}20 (현재 {hco3:.1f})")
    else: flags.append("HCO₃ 값 없음")
    if ag is not None: flags.append(f"AG {'>' if cond_ag else '≤'}12 (현재 {ag:.1f})")
    else: flags.append("AG 계산 불가")
    if lact is not None: flags.append(f"Lactate {'>' if cond_lac else '≤'}2 (현재 {lact:.2f})")
    else: flags.append("Lactate 값 없음")

    if cond_ph and cond_hco3 and (cond_ag or cond_lac):
        flags.append("⇒ 대사성 산증 기준 충족.")
    else:
        miss = []
        if not cond_ph: miss.append("pH<7.35")
        if not cond_hco3: miss.append("HCO₃<20")
        if not (cond_ag or cond_lac): miss.append("AG>12 또는 Lactate>2")
        flags.append("⇒ 대사성 산증 기준 일부 미충족: " + ", ".join(miss))
    flags.append("주의: DKA·패혈증·신부전 등 감별 필요.")
    return flags, used

# 6. Hyperkalemia
def hyperk_guideline_evidence_for_row(row: pd.Series, colmap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    k = _get_float(row, colmap.get("K"))
    flags = []
    used = {"K": k}
    if k is not None:
        flags.append(f"K⁺ {'>' if k>5.5 else '≤'}5.5 mmol/L (현재 {k:.2f})")
        flags.append("⇒ 고칼륨혈증 " + ("충족." if k>5.5 else "미충족."))
    else:
        flags.append("K⁺ 값 없음 (평가 불가).")
    flags.append("주의: 용혈 등 가성고칼륨 배제 필요. ECG 변화 시 응급.")
    return flags, used

# 7. Shock(기타)
def shock_other_guideline_evidence_for_row(row: pd.Series, colmap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    sbp = _get_float(row, colmap.get("SBP"))
    mapv = _get_float(row, colmap.get("MAP"))
    lact = _get_float(row, colmap.get("Lactate"))
    gcs  = _get_float(row, colmap.get("GCS"))
    uo   = _uo_mlkgh(row, colmap)

    flags = []
    used = {"SBP": sbp, "MAP": mapv, "Lactate": lact, "GCS": gcs, "UO_mLkg_h": uo}

    cond_press = ((sbp is not None and sbp < 90) or (mapv is not None and mapv < 65))
    cond_hypoperf = ((lact is not None and lact > 2) or (uo is not None and uo < 0.5) or (gcs is not None and gcs < 15))

    if sbp is not None: flags.append(f"SBP {'<' if sbp<90 else '≥'}90 (현재 {sbp:.1f})")
    else: flags.append("SBP 없음")
    if mapv is not None: flags.append(f"MAP {'<' if mapv<65 else '≥'}65 (현재 {mapv:.1f})")
    else: flags.append("MAP 없음")

    if lact is not None: flags.append(f"Lactate {'>' if lact>2 else '≤'}2 (현재 {lact:.2f})")
    else: flags.append("Lactate 없음")
    if uo is not None: flags.append(f"UO {'<' if uo<0.5 else '≥'}0.5 mL/kg/h (현재 {uo:.2f})")
    else: flags.append("UO(mL/kg/h) 없음")
    if gcs is not None: flags.append(f"GCS {'<' if gcs<15 else '≥'}15 (현재 {gcs:.1f})")
    else: flags.append("GCS 없음")

    if cond_press and cond_hypoperf:
        flags.append("⇒ 기타 쇼크(저관류 동반) 기준 충족.")
    else:
        miss = []
        if not cond_press: miss.append("SBP<90 또는 MAP<65")
        if not cond_hypoperf: miss.append("저관류 징후(락테이트↑/핍뇨/의식저하)")
        flags.append("⇒ 기타 쇼크 기준 일부 미충족: " + ", ".join(miss))
    flags.append("주의: 원인 분류(저혈량·심인성·폐쇄성·분포성) 필요. Hb 단독 해석 금지.")
    return flags, used

# 8. Pneumonia (CAP)
def pneumonia_guideline_evidence_for_row(row: pd.Series, colmap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    cxr = _row_yesno(row, colmap.get("CXR_infiltrate"))
    tempc = _get_float(row, colmap.get("Temp"))
    rr = _get_float(row, colmap.get("RR"))
    spo2 = _get_float(row, colmap.get("SpO2"))
    wbc = _get_float(row, colmap.get("WBC"))

    flags = []
    used = {"CXR_infiltrate": cxr, "Temp": tempc, "RR": rr, "SpO2": spo2, "WBC": wbc}

    cond_cxr = (cxr == 1)
    sxs = []
    if tempc is not None and tempc >= 38: sxs.append("발열(Temp≥38)")
    if rr is not None and rr > 20: sxs.append("빈호흡(RR>20)")
    if spo2 is not None and spo2 < 94: sxs.append("저산소(SpO2<94)")
    if wbc is not None and (wbc > 12000 or wbc < 4000): sxs.append("WBC 이상")
    cond_sym = len(sxs) > 0

    flags.append("CXR 침윤: " + ("있음" if cond_cxr else "없음/미확인"))
    flags.append("호흡기 증상/징후: " + (", ".join(sxs) if sxs else "없음/미확인"))

    if cond_cxr and cond_sym:
        flags.append("⇒ CAP 기준 충족.")
    else:
        miss = []
        if not cond_cxr: miss.append("CXR infiltrate")
        if not cond_sym: miss.append("호흡기 증상(발열/빈호흡/저산소/WBC 이상)")
        flags.append("⇒ CAP 기준 일부 미충족: " + ", ".join(miss))
    flags.append("주의: 영상 없이 확진 불가. 폐부종·ARDS 등 감별.")
    return flags, used

# 9. CHF / Acute HF
def chf_guideline_evidence_for_row(row: pd.Series, colmap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    bnp = _get_float(row, colmap.get("BNP"))
    ntp = _get_float(row, colmap.get("NTproBNP"))
    cardm = _row_yesno(row, colmap.get("CXR_cardiomegaly"))
    cong = _row_yesno(row, colmap.get("congestion"))

    flags = []
    used = {"BNP": bnp, "NTproBNP": ntp, "CXR_cardiomegaly": cardm, "congestion": cong}

    cond_biom = ((bnp is not None and bnp > 100) or (ntp is not None and ntp > 300))
    cond_img  = ((cardm == 1) or (cong == 1))

    if bnp is not None: flags.append(f"BNP {'>' if bnp>100 else '≤'}100 (현재 {bnp:.1f})")
    else: flags.append("BNP 없음")
    if ntp is not None: flags.append(f"NTproBNP {'>' if ntp>300 else '≤'}300 (현재 {ntp:.1f})")
    else: flags.append("NTproBNP 없음")
    flags.append("CXR cardiomegaly: " + ("있음" if cardm==1 else ("없음" if cardm==0 else "미확인")))
    flags.append("울혈 소견: " + ("있음" if cong==1 else ("없음" if cong==0 else "미확인")))

    if cond_biom and cond_img:
        flags.append("⇒ CHF/급성심부전 기준 충족.")
    else:
        miss = []
        if not cond_biom: miss.append("BNP>100 또는 NTproBNP>300")
        if not cond_img:  miss.append("심비대 또는 울혈 소견")
        flags.append("⇒ CHF/급성심부전 기준 일부 미충족: " + ", ".join(miss))
    flags.append("주의: BNP는 비만/CKD/AF/ARNI 영향 가능. 영상 병합 필수.")
    return flags, used

# 10. Pulmonary edema
def pulm_edema_guideline_evidence_for_row(row: pd.Series, colmap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    spo2 = _get_float(row, colmap.get("SpO2"))
    rr = _get_float(row, colmap.get("RR"))
    bilat = _row_yesno(row, colmap.get("CXR_bilateral"))

    flags = []
    used = {"SpO2": spo2, "RR": rr, "CXR_bilateral": bilat}

    cond = ((spo2 is not None and spo2 < 90) and (rr is not None and rr > 25) and (bilat == 1))

    if spo2 is not None: flags.append(f"SpO₂ {'<' if spo2<90 else '≥'}90% (현재 {spo2:.1f})")
    else: flags.append("SpO₂ 없음")
    if rr is not None: flags.append(f"RR {'>' if rr>25 else '≤'}25/min (현재 {rr:.1f})")
    else: flags.append("RR 없음")
    flags.append("CXR 양측 침윤: " + ("있음" if bilat==1 else ("없음" if bilat==0 else "미확인")))

    flags.append("⇒ 폐부종 " + ("기준 충족." if cond else "기준 일부 미충족."))
    flags.append("주의: 심인성 vs 비심인성 감별(BNP·에코·병력).")
    return flags, used

# 11. MI (UDMI)
def mi_udmi_guideline_evidence_for_row(row: pd.Series, colmap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    trop = _get_float(row, colmap.get("Troponin"))
    url99 = _get_float(row, colmap.get("Troponin_URL99"))
    ischem = _row_yesno(row, colmap.get("ECG_ischemia"))

    flags = []
    used = {"Troponin": trop, "URL99": url99, "ECG_ischemia": ischem}

    cond_trop = False
    if trop is not None and url99 is not None and url99 > 0:
        cond_trop = trop > url99
        flags.append(f"Troponin {'>' if trop>url99 else '≤'}URL99 (현재 {trop:.3f} / 기준 {url99:.3f})")
    else:
        flags.append("Troponin URL99 부재 → 'URL99 초과' 판정 불가 (사내 ULN/플래그 컬럼이 있으면 연결 권장)")

    if ischem is not None:
        flags.append("허혈 증거(ECG): " + ("있음" if ischem==1 else "없음"))
    else:
        flags.append("허혈 증거(ECG) 정보 없음")

    if cond_trop and ischem==1:
        flags.append("⇒ UDMI 기준(99백분위수 초과 + 허혈 증거) 충족(상승/하강 패턴은 추가 데이터 필요).")
    else:
        miss = []
        if not cond_trop: miss.append("Troponin>URL99")
        if ischem != 1:  miss.append("허혈 증거")
        flags.append("⇒ UDMI 기준 일부 미충족: " + ", ".join(miss))
    flags.append("주의: Troponin 단독 상승=심근손상(injury). 상승/하강 패턴, 임상 맥락 필요.")
    return flags, used

if __name__ == '__main__':
    main()