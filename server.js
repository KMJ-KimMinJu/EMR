const express = require("express");
const cors = require("cors");
const http = require("http");
const app = express();
const pool = require("./config/databaseSet");

app.use(cors);
app.use(express.json());

const server = http.createServer(app);

app.use("/api/patient");

server.listen(3000, () => {
  console.log("서버 실행 중, 포트는 3000");
});
