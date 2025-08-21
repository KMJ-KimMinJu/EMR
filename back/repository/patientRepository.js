const pool = require("../config/databaseSet");

class PatientRepository {
  async getPatientList() {
    const sql = `SELECT patientId, sex, name, age, DATE_FORMAT(birth,'%Y-%m-%d') as birth FROM patient`;

    try {
      const [result] = await pool.query(sql);

      return result;
    } catch (error) {
      console.log(error);
    }
  }

  async getPatientDetail(patientId) {
    const sql = `SELECT patientId, sex, age, phone, name, DATE_FORMAT(birth,'%Y-%m-%d') as birth, address, registration_number FROM patient WHERE patientId = ?`;

    try {
      const [result] = await pool.query(sql, patientId);

      return result[0];
    } catch (error) {
      console.log(error);
    }
  }

  async createPatient(patientData) {
    const sql = `
      INSERT INTO patient
        (name, birth, age, sex, phone, address, registration_number)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;

    try {
      const [result] = await pool.query(sql, [
        patientData.name,
        patientData.birth,
        patientData.age,
        patientData.sex,
        patientData.phone,
        patientData.address,
        patientData.registration_number,
      ]);

      return { success: true, insertedId: result.insertId };
    } catch (error) {
      console.log(error);
    }
  }

  async updatePatient(patientData) {
    const sql = `
      UPDATE patient
      SET
        name = ?,
        birth = ?,
        age = ?,
        sex = ?,
        phone = ?,
        address = ?,
        registration_number = ?
      WHERE id = ?
    `;

    try {
      const [result] = await pool.query(sql, [
        patientData.name,
        // YYYYMMDD → YYYY-MM-DD 변환
        patientData.birth,
        patientData.age,
        patientData.sex,
        patientData.phone,
        patientData.address,
        patientData.registration_number,
        patientData.id, // 업데이트 기준 (PK)
      ]);

      return { success: true, affectedRows: result.affectedRows };
    } catch (error) {
      console.error(error);
      throw error;
    }
  }
}

module.exports = new PatientRepository();
