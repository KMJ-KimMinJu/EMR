const express = require("express");
const router = express.Router();
const bus = require("../config/bus");
const predictController = require("../controller/predictController");

router.get("/stream", (req, res) => {
  res.set({
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no", // nginx 앞단일 때 버퍼링 금지
    "Access-Control-Allow-Origin": "*", // 절대경로로 붙을 가능성 대비
  });

  res.flushHeaders();
  console.log("hello");

  // 재연결 지연(ms) 힌트 (선택)
  res.write(`retry: 5000\n\n`);

  const raw = req.query.patientId;

  // 환자 필터링(선택): /api/stream?patientId=12345
  const filterPatientId = req.query.patientId
    ? String(req.query.patientId)
    : null;

  const sendVital = (msg) => {
    const pid = String(parseInt(msg.patientId, 10));
    // console.log("[router] got vital for", pid, "filter=", filterPatientId);
    if (filterPatientId && pid !== filterPatientId) return;

    if (res.writableEnded) {
      console.log("[router] conn already closed");
      return;
    }

    const ok1 = res.write(`event: vital\n`);
    const ok2 = res.write(`data: ${JSON.stringify(msg)}\n\n`);
    console.log(
      "[router] wrote vital =>",
      ok1,
      ok2,
      "ended?",
      res.writableEnded
    );
  };

  const sendFull = (msg) => {
    if (filterPatientId && String(msg.patientId) !== filterPatientId) return;
    res.write(`event: full\n`);
    res.write(`data: ${JSON.stringify(msg)}\n\n`);
  };

  bus.on("vital", sendVital);
  bus.on("full", sendFull);

  // (선택) 커넥션 유지용 핑
  const ping = setInterval(() => {
    res.write(`event: ping\ndata: "keep-alive"\n\n`);
  }, 25000);

  req.on("close", () => {
    clearInterval(ping);
    bus.off("vital", sendVital);
    bus.off("full", sendFull);
  });
});

// //1차 예측 테스트
// router.get("/first", predictController.firstTest.bind(predictController));

// //2차 예측 테스트
// router.get("/second", predictController.secondTest.bind(predictController));

module.exports = router;
