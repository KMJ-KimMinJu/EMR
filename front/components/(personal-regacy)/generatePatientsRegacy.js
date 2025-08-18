const sampleData = {
  진료구분: ["초진", "재진", "응급"],
  예약: ["예약완료", "예약대기", "예약취소"],
  내원일자: ["2025-08-01", "2025-08-18", "2025-09-02"],
  센터: ["심장센터", "신경센터", "정형외과센터"],
  진료과: ["내과", "외과", "소아과", "산부인과"],
  주치의: ["김철수", "이영희", "박민수", "최하늘"],
  병실: ["101호", "202호", "VIP실", "일반실", "통원"],
  예약구분: ["온라인", "전화", "현장"],
  보험유형: ["국민건강", "실손보험", "민간보험", "비보험"],
  진료여부: ["진료완료", "진료대기", "진료중"],
  수납여부: ["수납완료", "미수납", "보험청구"],
  진단코드: ["A10", "B20", "C30", "D40"],
  진단명: [
    "고혈압",
    "당뇨병",
    "감기",
    "폐렴",
    "골절",
    "뇌진탕",
    "저혈압",
    "찰과상",
    "타박상",
  ],
};

// 1~3개 랜덤 선택
function getRandomValues(arr) {
  const count = Math.floor(Math.random()) + 1; // 1~3개
  const shuffled = [...arr].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, count);
}

export function generatePatientRow() {
  const row = {};
  Object.keys(sampleData).forEach((key) => {
    row[key] = getRandomValues(sampleData[key]);
  });
  return row;
}

// 여러 행 생성
export function generatePatientTable(rows = 5) {
  return Array.from({ length: rows }, () => generatePatientRow());
}
