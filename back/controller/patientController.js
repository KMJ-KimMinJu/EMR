// const patientService = require("../service/patientService");

class PatientController {
  async getPatientList(req, res, next) {
    try {
      // const patients = await patientService.getPatientList();
      const patients = [
        {
          patientId: 1,
          sex: "F",
          age: 23,
          name: "홍길동",
          birth: "20020129",
          address: "대충 어디 살고있음",
        },
        {
          patientId: 2,
          sex: "M",
          age: 30,
          name: "김철수",
          birth: "19950815",
          address: "알아서 뭐할라고",
        },
      ];
      res.status(200).json(patients);
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Internal Server Error" });
    }
  }

  async getPatientDetail(req, res, next) {
    try {
      const { patientId } = req.params;
      if (!patientId) {
        return res.status(400).json({ message: "patientId is required" });
      }

      // const patient = await patientService.getPatientDetail(patientId);
      const patient = {
        patientId,
        sex: "F",
        age: 23,
        phone: "010-1234-5678",
        name: "홍길동",
        birth: "20020129",
        address: "부산광역시 해운대구",
        registration_number: "020129-4",
      };

      if (!patient) {
        return res.status(404).json({ message: "Patient not found" });
      }

      res.status(200).json(patient);
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Internal Server Error" });
    }
  }

  async createPatient(req, res, next) {
    try {
      const { name, birth, age, sex, phone, address, registration_number } =
        req.body;

      if (
        !name ||
        !birth ||
        !age ||
        !sex ||
        !phone ||
        !address ||
        !registration_number
      ) {
        return res.status(400).json({ message: "Missing required fields" });
      }

      // const created = await patientService.createPatient(req.body);
      const created = {
        success: true,
        patientId: 9999,
      };

      res.status(201).json(created);
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Internal Server Error" });
    }
  }

  async updatePatient(req, res, next) {
    try {
      const { id } = req.body;

      if (!id) {
        return res.status(400).json({ message: "id is required" });
      }

      // const updated = await patientService.updatePatient(req.body);
      const updated = {
        success: true,
        patientId: id,
      };

      res.status(200).json(updated);
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Internal Server Error" });
    }
  }
}

module.exports = new PatientController();
