"use client";
import { useState, useEffect } from "react";
import { generatePatientTable } from "./generatePatientsRegacy";
import tableStyles from "./PatientTable.module.css"; // 표 전용 CSS (기존 그대로)
import styles from "./regacy.module.css"; // 새로 만든 외곽/탭 CSS

export default function Regacy() {
  const [data, setData] = useState([]);
  const [toggle1, setToggle1] = useState(true);
  const [toggle2, setToggle2] = useState(true);

  useEffect(() => {
    setData(generatePatientTable(5));
  }, []);

  const headers = [
    "진료구분",
    "예약",
    "내원일자",
    "센터",
    "진료과",
    "주치의",
    "병실",
    "예약구분",
    "보험유형",
    "진료여부",
    "수납여부",
    "진단코드",
    "진단명",
  ];

  // 표 렌더 함수 (중복 제거용)
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
                <td key={h}>{row[h].join(", ")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <div className={styles.wrap}>
      {/* 블록 1: 수진/수술 이력 */}
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

      {/* 블록 2: 처방/전실 이력 */}
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
