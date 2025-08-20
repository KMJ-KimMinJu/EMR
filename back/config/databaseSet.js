const mysql = require("mysql2/promise");
require("dotenv").config();

const pool = mysql.createPool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  user: process.env.DB_USER,
  password: process.env.DB_PASS,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  connectTimeout: 10000, // 10초로 설정
});

console.log(process.env.DB_HOST, process.env.DB_PORT, process.env.DB_USER, process.env.DB_PASS, process.env.DB_NAME);

pool
  .getConnection()
  .then((conn) => {
    console.log("Connected to the database");
    conn.release();
  })
  .catch((err) => {
    console.error("Error connecting to the database:", err);
  });

module.exports = pool;
