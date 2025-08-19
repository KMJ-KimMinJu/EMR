const express = require("express");
const router = express.Router();
const predictController = require("../controller/predictController");

//1차 예측 테스트
router.get("/first", predictController.firstTest.bind(predictController));

//2차 예측 테스트
router.get("/second", predictController.secondTest.bind(predictController));

module.exports = router;
