const express = require("express");
const router = express.Router();
const bus = require("../config/bus");
const predictController = require("../controller/predictController");

router.get("/stream", (req, res) => {
  res.set({
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  res.flushHeaders();
  console.log("hello");

  // 재연결 지연(ms) 힌트 (선택)
  res.write(`retry: 5000\n\n`);

  // 환자 필터링(선택): /api/stream?patientId=12345
  const filterPatientId = req.query.patientId
    ? String(req.query.patientId)
    : null;

  const sendVital = (msg) => {
    if (filterPatientId && String(msg.patientId) !== filterPatientId) return;
    res.write(`event: vital\n`);
    res.write(`data: ${JSON.stringify(msg)}\n\n`);
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
