const pool = require("../config/databaseSet");

class PatientRepository {
  async getPatientList() {
    sql = `SELECT * FROM PATIENT`;

    try {
      const [result] = await pool.query(sql);

      return { success: true, patientList: result };
    } catch (error) {
      console.log(error);
    }
  }

  async getPatientDetail(patientId) {
    sql = `SELECT * FROM PATIENT WHERE patientId = ?`;

    try {
      const [result] = await pool.query(sql, patientId);
      return { success: true, patientDetail: result };
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
        // YYYYMMDD → DATE 로 바꾸기
        `${patientData.birth.toString().slice(0, 4)}-${patientData.birth
          .toString()
          .slice(4, 6)}-${patientData.birth.toString().slice(6, 8)}`,
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
        `${patientData.birth.toString().slice(0, 4)}-${patientData.birth
          .toString()
          .slice(4, 6)}-${patientData.birth.toString().slice(6, 8)}`,
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
