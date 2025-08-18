const express = require("express");
const cors = require("cors");
const http = require("http");
const app = express();
const patientRouter = require("./router/patientRouter");

app.use(cors);
app.use(express.json());

const server = http.createServer(app);

app.use("/api", patientRouter);

server.listen(3000, () => {
  console.log("서버 실행 중, 포트는 3000");
});
