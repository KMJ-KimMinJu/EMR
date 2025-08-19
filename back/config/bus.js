const EventEmitter = require("events");
const bus = new EventEmitter();

// 필요하면 리스너 최대치 늘리기 (대량 접속 대비)
// bus.setMaxListeners(1000);

module.exports = bus;
