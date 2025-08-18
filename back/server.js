const express = require("express");
const cors = require("cors");
const http = require("http");
const app = express();
const patientRouter = require("./router/patientRouter");
require("dotenv").config();

app.use(cors());
app.use(express.json());

const server = http.createServer(app);

app.use("/api", patientRouter);

server.listen(process.env.PORT, () => {
  console.log(`서버 실행 중, 포트는 ${process.env.PORT}`);
});
