const EventEmitter = require("events");
const bus = new EventEmitter();

// 필요하면 리스너 최대치 늘리기 (대량 접속 대비)
bus.setMaxListeners(0); // 연결 많이 붙어도 경고 안 나게

module.exports = bus;
