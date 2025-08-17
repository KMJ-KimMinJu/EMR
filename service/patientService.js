const patientRepository = require("../repository/patientRepository");

class PatientService {
  async getPatientList() {
    return await patientRepository.getPatientList();
  }

  async getPatientDetail(patientId) {
    return await patientRepository.getPatientDetail(patientId);
  }

  async createPatient(patientData) {
    return await patientRepository.createPatient(patientData);
  }

  async updatePatient(patientData) {
    return await patientRepository.updatePatient(patientData);
  }
}

module.exports = new PatientService();
