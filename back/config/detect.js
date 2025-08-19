// worker.js (CommonJS로 통일)
const axios = require("axios");
const pool = require("./databaseSet");

// 윈도우 조건: lab 들어왔을 때 함께 보낼 최신 vital을 얼마까지 소급할지 (분)
const VITAL_LOOKBACK_MIN = 30;
const bus = require("./bus");

// === 유틸 ===
async function fetchBatch() {
  const [rows] = await pool.query(
    `SELECT * FROM inference_queue
     WHERE processed = 0
     ORDER BY id ASC
     LIMIT 100`
  );
  return rows;
}

async function markProcessed(ids) {
  if (!ids.length) return;
  await pool.query(
    `UPDATE inference_queue SET processed = 1 WHERE id IN (${ids
      .map(() => "?")
      .join(",")})`,
    ids
  );
}

// === 데이터 적재: PK/타임스탬프 컬럼명에 주의! ===

// patient_vital의 단일 레코드를 PK로 조회
async function getVitalByPk(vitalPk) {
  // PK 컬럼명이 'id'가 아닐 수 있음. 예: vital_id
  const [[row]] = await pool.query(
    `SELECT *
       FROM patient_vital
      WHERE vitalId = ?`, // <-- 여기를 네 실제 PK로 변경 (예: vital_id = ?)
    [vitalPk]
  );
  return row || null;
}

// patient_lab의 단일 레코드를 PK로 조회
async function getLabByPk(labPk) {
  // PK 컬럼명이 'id'가 아닐 수 있음. 예: lab_id
  const [[row]] = await pool.query(
    `SELECT *
       FROM patient_lab
      WHERE labId = ?`, // <-- 여기를 네 실제 PK로 변경 (예: lab_id = ?)
    [labPk]
  );
  return row || null;
}

// 특정 시각 전후 기준으로 같은 환자의 "가장 최신 vital" 1건 찾기
async function getLatestVitalForPatient(patientId, refTime /* Date or null */) {
  if (refTime) {
    // refTime 기준 lookback 윈도우 내에서 최신 vital 1건
    const [[row]] = await pool.query(
      `SELECT *
         FROM patient_vital
        WHERE patientId = ?
          AND measured_at >= DATE_SUB(?, INTERVAL ? MINUTE)   -- measured_at 컬럼명 확인
          AND measured_at <= ?
        ORDER BY measured_at DESC
        LIMIT 1`,
      [patientId, refTime, VITAL_LOOKBACK_MIN, refTime]
    );
    if (row) return row;
  }
  // 윈도우에서 못 찾으면 환자의 최신 vital 1건 (전체 중)
  const [[fallback]] = await pool.query(
    `SELECT *
       FROM patient_vital
      WHERE patientId = ?
      ORDER BY measured_at DESC
      LIMIT 1`,
    [patientId]
  );
  return fallback || null;
}

// === 페이로드 구성 ===

function buildVitalPayload(patientId, v) {
  if (!v) return null;
  return {
    patientId,
    temperature: v.temperature,
    heartrate: v.heartrate,
    resprate: v.resprate,
    o2sat: v.o2sat,
    sbp: v.sbp,
    dbp: v.dbp,
    map: v.map,
    // 원본 레코드 정보(옵션)
    source: "vital",
    vital_pk: v.vitalId, // <-- 실제 PK 컬럼명으로 교체
    measured_at: v.measured_at, // 컬럼명 확인
  };
}

function buildFullPayload(patientId, l, v /* vital can be null */) {
  if (!l) return null;
  const payload = {
    patientId,
    // lab
    lactate: l.lactate,
    glucose: l.glucose,
    pH: l.pH,
    bicarbonate: l.bicarbonate,
    aniongap: l.aniongap,
    troponin: l.troponin,
    wbc: l.wbc,
    creatinine: l.creatinine,
    potassium: l.potassium,
    bnp: l.bnp,
    uketone: l.uketone,
    hydroxybutyrate: l.hydroxybutyrate,
    // vital (있다면 포함)
    ...(v
      ? {
          temperature: v.temperature,
          heartrate: v.heartrate,
          resprate: v.resprate,
          o2sat: v.o2sat,
          sbp: v.sbp,
          dbp: v.dbp,
          map: v.map,
          vital_pk: v.vitalId, // <-- 실제 PK 컬럼명으로 교체
          vital_measured_at: v.measured_at,
        }
      : {}),
    // 메타
    source: "lab",
    lab_pk: l.labId, // <-- 실제 PK 컬럼명으로 교체
    lab_measured_at: l.measured_at, // 컬럼명 확인
    vital_attached: !!v,
  };
  return payload;
}

// === 라우팅 ===
// 환경변수 예시:
// FASTAPI_BASE=http://fastapi:8000
// VITAL_PATH=/predict/vital
// FULL_PATH=/predict/full

const FASTAPI_BASE = process.env.FASTAPI_BASE || "http://127.0.0.1:8000";
const VITAL_PATH = process.env.VITAL_PATH || "/predict/vital";
const FULL_PATH = process.env.FULL_PATH || "/predict/full";

async function callVitalAPI(payload) {
  const { data } = await axios.post(`${FASTAPI_BASE}${VITAL_PATH}`, payload, {
    timeout: 8000,
    headers: { "Content-Type": "application/json" },
  });
  return data;
}

async function callFullAPI(payload) {
  const { data } = await axios.post(`${FASTAPI_BASE}${FULL_PATH}`, payload, {
    timeout: 8000,
    headers: { "Content-Type": "application/json" },
  });
  return data;
}

// === 메인 루프 ===

async function tick() {
  const batch = await fetchBatch();
  const doneIds = [];

  for (const evt of batch) {
    try {
      if (evt.source_table === "patient_vital") {
        // 1차 예측: vital 단일
        const v = await getVitalByPk(evt.source_pk);
        const vitalPayload = buildVitalPayload(evt.patientId, v);
        if (!vitalPayload)
          throw new Error("Vital row not found or payload invalid");
        const result = await callVitalAPI(vitalPayload);

        bus.emit("vital", {
          stage: "vital",
          patientId: evt.patientId,
          source_table: evt.source_table,
          source_pk: evt.source_pk,
          result, // FastAPI 결과
          at: new Date().toISOString(),
        });

        await pool.query(
          `INSERT INTO inference_result
             (patientId, stage, source_table, source_pk, result_json, created_at)
           VALUES (?, 'vital', ?, ?, ?, NOW())`,
          [
            evt.patientId,
            evt.source_table,
            evt.source_pk,
            JSON.stringify(result),
          ]
        );
      } else if (evt.source_table === "patient_lab") {
        // 2차 예측: lab + (가능하면 최신 vital)
        const l = await getLabByPk(evt.source_pk);
        if (!l) throw new Error("Lab row not found");

        // lab 측정 시각 기준으로 최신 vital 찾기(윈도우 내 우선)
        const refTime = l.measured_at || null; // 컬럼명 확인
        const v = await getLatestVitalForPatient(evt.patientId, refTime);

        const fullPayload = buildFullPayload(evt.patientId, l, v);
        const result = await callFullAPI(fullPayload);

        bus.emit("full", {
          stage: "full",
          patientId: evt.patientId,
          source_table: evt.source_table,
          source_pk: evt.source_pk,
          result,
          at: new Date().toISOString(),
        });

        await pool.query(
          `INSERT INTO inference_result
             (patientId, stage, source_table, source_pk, result_json, created_at)
           VALUES (?, 'full', ?, ?, ?, NOW())`,
          [
            evt.patientId,
            evt.source_table,
            evt.source_pk,
            JSON.stringify(result),
          ]
        );
      } else {
        // 정의되지 않은 소스
        console.warn("Unknown source_table:", evt.source_table);
      }

      doneIds.push(evt.id);
    } catch (e) {
      // 실패 시 processed=0이라 다음 tick에 재시도됨
      console.error(
        "[inference error]",
        { queue_id: evt.id, src: evt.source_table },
        e.message
      );
      // 필요하면 오류 유형별 백오프/데드레터 큐 설계
    }
  }

  await markProcessed(doneIds);
}

// setInterval보다: 처리 시간이 주기를 초과해도 백투백로 돌아가게 루프로 돌릴 수도 있음
setInterval(tick, 700);

// // worker.js (간단 SSE 버전)
// const pool = require("./databaseSet");
// const bus = require("./bus");

// // === 유틸 ===
// async function fetchBatch() {
//   const [rows] = await pool.query(
//     `SELECT * FROM inference_queue
//      WHERE processed = 0
//      ORDER BY id ASC
//      LIMIT 100`
//   );
//   return rows;
// }

// async function markProcessed(ids) {
//   if (!ids.length) return;
//   await pool.query(
//     `UPDATE inference_queue SET processed = 1 WHERE id IN (${ids
//       .map(() => "?")
//       .join(",")})`,
//     ids
//   );
// }

// // === 메인 루프 ===
// async function tick() {
//   const batch = await fetchBatch();
//   const doneIds = [];

//   for (const evt of batch) {
//     try {
//       if (evt.source_table === "patient_vital") {
//         // ✔ vital 감지 시 프론트로 즉시 알림
//         bus.emit("vital", {
//           stage: "vital",
//           patientId: evt.patientId,
//           source_table: evt.source_table,
//           source_pk: evt.source_pk,
//           at: new Date().toISOString(),
//         });
//       } else if (evt.source_table === "patient_lab") {
//         // ✔ lab 감지 시 프론트로 즉시 알림
//         bus.emit("full", {
//           stage: "full",
//           patientId: evt.patientId,
//           source_table: evt.source_table,
//           source_pk: evt.source_pk,
//           at: new Date().toISOString(),
//         });
//       } else {
//         console.warn("Unknown source_table:", evt.source_table);
//       }

//       // 처리 완료 표시 (재전송 방지)
//       doneIds.push(evt.id);
//     } catch (e) {
//       console.error(
//         "[emit error]",
//         { queue_id: evt.id, src: evt.source_table },
//         e.message
//       );
//     }
//   }

//   await markProcessed(doneIds);
// }

// // 주기 실행
// setInterval(tick, 700);
