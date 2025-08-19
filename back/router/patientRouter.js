const express = require("express");
const router = express.Router();
const patientController = require("../controller/patientController");

//리스트 조회
router.get(
  "/patient/list",
  patientController.getPatientList.bind(patientController)
);

//환자 조회
router.get(
  "/patient/:patientId",
  patientController.getPatientDetail.bind(patientController)
);

//환자 등록
router.post(
  "/patient/",
  patientController.createPatient.bind(patientController)
);

//환자 정보 수정
router.put(
  "/patient/",
  patientController.updatePatient.bind(patientController)
);

module.exports = router;
