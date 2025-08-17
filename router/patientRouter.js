const express = require("express");
const router = express.Router();
const patientController = require("../service/patientController");

//리스트 조회
router.get("/list", patientController.getPatientList.bind(patientController));

//환자 조회
router.get(
  "/:patiendId",
  patientController.getPatientDetail.bind(patientController)
);

//환자 등록
router.post("/", patientController.createPatient.bind(patientController));

//환자 정보 수정
router.put("/", patientController.updatePatient.bind(patientController));
