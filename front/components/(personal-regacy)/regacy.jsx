"use client";
import { useState, useEffect } from "react";
import tableStyles from "./PatientTable.module.css";
import styles from "./regacy.module.css";

export default function Regacy() {
  const [data, setData] = useState([]);
  const [toggle1, setToggle1] = useState(true);
  const [toggle2, setToggle2] = useState(true);

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
      "화상",
      "두통",
      "치통",
      "시력저하",
    ],
  };

  // ✅ 1~3개 랜덤 선택 (배열 길이 초과 방지)
  function getRandomValues(arr) {
    const count = Math.min(arr.length, Math.floor(Math.random() * 3) + 1); // 1~3
    const shuffled = [...arr].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  }

  function generatePatientRow() {
    const row = {};
    for (const key of Object.keys(sampleData)) {
      row[key] = getRandomValues(sampleData[key]);
    }
    return row;
  }

  function generatePatientTable(rows = 5) {
    return Array.from({ length: rows }, () => generatePatientRow());
  }

  useEffect(() => {
    const newData = generatePatientTable(5);
    setData(newData);
    // console.log("generated:", newData);
  }, []);

  // 헤더는 sampleData의 키에서 직접 뽑아 일치 보장
  const headers = Object.keys(sampleData);

  const renderTable = () => (
    <div className={tableStyles.tableWrapper}>
      <table className={tableStyles.table}>
        <thead>
          <tr>
            {headers.map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              {headers.map((h) => (
                <td key={h}>
                  {Array.isArray(row[h]) ? row[h].join(", ") : ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <div className={styles.wrap}>
      <div className={styles.block}>
        <div className={styles.tabbar}>
          <button
            className={`${styles.tab} ${toggle1 ? styles.active : ""}`}
            onClick={() => setToggle1(true)}
          >
            수진이력
          </button>
          <button
            className={`${styles.tab} ${!toggle1 ? styles.active : ""}`}
            onClick={() => setToggle1(false)}
          >
            수술이력
          </button>
        </div>
        <div className={styles.body}>{renderTable()}</div>
      </div>

      <div className={styles.block}>
        <div className={styles.tabbar}>
          <button
            className={`${styles.tab} ${toggle2 ? styles.active : ""}`}
            onClick={() => setToggle2(true)}
          >
            처방이력
          </button>
          <button
            className={`${styles.tab} ${!toggle2 ? styles.active : ""}`}
            onClick={() => setToggle2(false)}
          >
            전실이력
          </button>
        </div>
        <div className={styles.body}>{renderTable()}</div>
      </div>
    </div>
  );
}
