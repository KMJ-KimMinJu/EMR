// worker.js
const axios = require("axios");
const pool = require("./databaseSet");
const bus = require("./bus");

(async () => {
  const [[db]] = await pool.query("SELECT DATABASE() AS db");
  const [[sv]] = await pool.query("SELECT @@hostname AS host, @@port AS port");
  console.log("[DB INFO]", sv, db);

  const [q1] = await pool.query(
    "SELECT id, source_table, source_pk, patientId, processed FROM inference_queue ORDER BY id DESC LIMIT 10"
  );
  console.log("[SAMPLE ROWS (latest)]", q1);

  // ✅ 워커가 실제로 집어갈 대상
  const [q0] = await pool.query(
    "SELECT id, source_table, source_pk, patientId, processed, (source_pk IS NULL) AS is_null FROM inference_queue WHERE processed=0 ORDER BY id ASC LIMIT 20"
  );
  console.log("[CANDIDATES (processed=0)]", q0);
})();



// lab 들어왔을 때 함께 보낼 최신 vital의 lookback(분)
const VITAL_LOOKBACK_MIN = 30;

// FastAPI 엔드포인트
const FASTAPI_BASE = process.env.FASTAPI_BASE || "http://127.0.0.1:8000";
const VITAL_PATH = process.env.VITAL_PATH || "/predict/vital";
const FULL_PATH = process.env.FULL_PATH || "/predict/full";

// ========== 공용 유틸 ==========
async function fetchBatch() {
  const [rows] = await pool.query(
    `SELECT
        id,
        patientId,
        source_table,
        source_pk,
        processed
     FROM inference_queue
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

// ========== 스키마에 맞춘 조회 ==========
async function getVitalByPk(vitalPk) {
  const [[row]] = await pool.query(
    `SELECT *
       FROM patient_vital
      WHERE vitalId = ?`,
    [vitalPk]
  );
  return row || null;
}

async function getLabByPk(labPk) {
  const [[row]] = await pool.query(
    `SELECT *
       FROM patient_lab
      WHERE labId = ?`,
    [labPk]
  );
  return row || null;
}

// refTime(=lab.record_time) 근처의 최신 vital 1건 → 없으면 전체 최신
async function getLatestVitalForPatient(
  patientId,
  refTime /* Date|string|null */
) {
  if (refTime) {
    const [[row]] = await pool.query(
      `SELECT *
         FROM patient_vital
        WHERE patientId = ?
          AND record_time >= DATE_SUB(?, INTERVAL ? MINUTE)
          AND record_time <= ?
        ORDER BY record_time DESC
        LIMIT 1`,
      [patientId, refTime, VITAL_LOOKBACK_MIN, refTime]
    );
    if (row) return row;
  }
  const [[fallback]] = await pool.query(
    `SELECT *
       FROM patient_vital
      WHERE patientId = ?
      ORDER BY record_time DESC
      LIMIT 1`,
    [patientId]
  );
  return fallback || null;
}

// ========== FastAPI 호출 ==========
async function callVitalAPI(payload) {
  const { data } = await axios.post(`${FASTAPI_BASE}${VITAL_PATH}`, payload, {
    timeout: 8000,
    headers: { "Content-Type": "application/json" },
  });
  return data; // 기대: { success:true, predict:"고위험군"|"저위험군" }
}

async function callFullAPI(payload) {
  const { data } = await axios.post(`${FASTAPI_BASE}${FULL_PATH}`, payload, {
    timeout: 8000,
    headers: { "Content-Type": "application/json" },
  });
  return data; // 기대: { success:true, ICU_percent, ICU_hours, diseases:[...] }
}

// ========== 페이로드 구성(스키마 반영) ==========
// vital → FastAPI에 보낼 입력
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
    source: "vital",
    vital_pk: v.vitalId,
    measured_at: v.record_time, // DB 컬럼 record_time 사용
  };
}

// full → FastAPI에 보낼 입력 (lab + optional vital)
function buildFullPayload(patientId, l, v /* nullable */) {
  if (!l) return null;
  return {
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
    uketone: l.uketone, // TINYINT(1): 0/1로 전달
    hydroxybutyrate: l.hydroxybutyrate,
    // optional vital 동봉
    ...(v
      ? {
          temperature: v.temperature,
          heartrate: v.heartrate,
          resprate: v.resprate,
          o2sat: v.o2sat,
          sbp: v.sbp,
          dbp: v.dbp,
          map: v.map,
          vital_pk: v.vitalId,
          vital_measured_at: v.record_time,
          vital_attached: true,
        }
      : { vital_attached: false }),
    source: "lab",
    lab_pk: l.labId,
    lab_measured_at: l.record_time,
  };
}

// ========== 메인 루프 ==========
async function tick() {
  const batch = await fetchBatch();
  const doneIds = [];
  console.log("[tick RAW]", batch[0]);                 // 원시 레코드
  console.log("[tick KEYS]", Object.keys(batch[0]||{}));
  
  for (const evt of batch) {
    try {
      console.log("[tick]", {
        id: evt.id,
        src: evt.source_table,
        pk: evt.source_pk,
        patientId: evt.patientId,
      });

      if (evt.source_pk == null) {
        console.error(
          "[tick] missing source_pk; columns seen:",
          Object.keys(evt)
        );
        // 무한 재시도 막기 (임시로 버리거나, 별도 테이블로 이동)
        await markProcessed([evt.id]);
        continue;
      }

      if (evt.source_table === "patient_vital") {
        // 1차 예측
        const v = await getVitalByPk(evt.source_pk);
        const vitalPayload = buildVitalPayload(evt.patientId, v);
        console.log("[tick] vital row?", !!v, "pk", evt.source_pk);
        if (!vitalPayload)
          throw new Error("Vital row not found or payload invalid");

        const result = await callVitalAPI(vitalPayload); // { success, predict }
        console.log("[vital api result]", result);
        // 프론트가 바로 쓰기 좋게 루트로 펼쳐서 emit
        bus.emit("vital", {
          stage: "vital",
          patientId: evt.patientId,
          ...result, // success, predict
        });

        // (선택) 결과 저장 원하면 아래 주석 해제
        // await pool.query(
        //   `INSERT INTO inference_result
        //      (patientId, stage, source_table, source_pk, result_json, created_at)
        //    VALUES (?, 'vital', ?, ?, ?, NOW())`,
        //   [evt.patientId, evt.source_table, evt.source_pk, JSON.stringify(result)]
        // );
      } else if (evt.source_table === "patient_lab") {
        // 2차 예측
        const l = await getLabByPk(evt.source_pk);
        if (!l) throw new Error("Lab row not found");
        const refTime = l.record_time || null; // ⬅ record_time 사용
        const v = await getLatestVitalForPatient(evt.patientId, refTime);

        const fullPayload = buildFullPayload(evt.patientId, l, v);
        const result = await callFullAPI(fullPayload); // { success, ICU_percent, ICU_hours, diseases }

        bus.emit("full", {
          stage: "full",
          patientId: evt.patientId,
          ...result, // success, ICU_percent, ICU_hours, diseases
        });

        // await pool.query(
        //   `INSERT INTO inference_result
        //      (patientId, stage, source_table, source_pk, result_json, created_at)
        //    VALUES (?, 'full', ?, ?, ?, NOW())`,
        //   [evt.patientId, evt.source_table, evt.source_pk, JSON.stringify(result)]
        // );
      } else {
        console.warn("Unknown source_table:", evt.source_table);
      }

      doneIds.push(evt.id);
    } catch (e) {
      console.error(
        "[inference error]",
        { queue_id: evt.id, src: evt.source_table },
        e.message
      );
      // 실패 시 processed=0 → 다음 tick에 재시도
    }
  }

  await markProcessed(doneIds);
}

setInterval(tick, 700);
console.log("[worker] started");

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
