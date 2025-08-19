const patientService = require("../service/patientService");

class PatientController {
  async getPatientList(req, res, next) {
    try {
      const patients = await patientService.getPatientList();

      res.status(200).json(patients);
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Internal Server Error" });
    }
  }

  async getPatientDetail(req, res, next) {
    try {
      const { patientId } = req.params;

      console.log(req.params);
      if (!patientId) {
        return res.status(400).json({ message: "patientId is required" }); // 400
      }

      const patient = await patientService.getPatientDetail(patientId);
      console.log(patient);
      if (!patient) {
        return res.status(404).json({ message: "Patient not found" }); // 404
      }

      res.status(200).json(patient);
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Internal Server Error" }); // 500
    }
  }

  async createPatient(req, res, next) {
    try {
      const { name, birth, age, sex, phone, address, registration_number } =
        req.body;

      // 필수 값 체크 → 400
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

      const created = await patientService.createPatient(req.body);
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
        return res.status(400).json({ message: "id is required" }); // 400
      }

      const updated = await patientService.updatePatient(req.body);
      res.status(200).json(updated);
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Internal Server Error" }); // 500
    }
  }
}

module.exports = new PatientController();
