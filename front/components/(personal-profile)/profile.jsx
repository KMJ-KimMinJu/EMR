// components/(personal-profile)/profile.jsx
"use client";
import { useEffect, useState } from "react";
import styles from "./profile.module.css";

export default function Profile({ patientId }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // 선택 전이면 비움
    if (!patientId) {
      setDetail(null);
      return;
    }
    let alive = true;
    setLoading(true);
    (async () => {
      try {
        const res = await fetch(
          `http://localhost:4000/api/patient/${patientId}`,
          {
            cache: "force-cache",
          }
        );
        console.log(1);
        const json = await res.json();
        if (alive) setDetail(json);
      } catch (e) {
        console.error(e);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [patientId]);

  return (
    <div className={styles.con}>
      <div className={styles.title}>환자 진료정보 조회</div>
      <div className={styles.profile}>
        {!patientId ? (
          <div className={styles.span3}>좌측 리스트에서 환자를 선택하세요.</div>
        ) : loading ? (
          <div className={styles.span3}>불러오는 중…</div>
        ) : detail ? (
          <>
            <div className={styles.wrap}>
              <div>등록번호</div>
              <div>{detail.patientId}</div>
            </div>
            <div className={styles.wrap}>
              <div>성별</div>
              <div>{detail.sex}</div>
            </div>
            <div className={styles.wrap}>
              <div>나이</div>
              <div>{detail.age}</div>
            </div>
            <div className={styles.wrap}>
              <div>연락처</div>
              <div>{detail.phone}</div>
            </div>
            <div className={styles.wrap}>
              <div>이름</div>
              <div>{detail.name}</div>
            </div>
            <div className={styles.wrap}>
              <div>생년월일</div>
              <div>{detail.birth}</div>
            </div>
            <div className={`${styles.wrap} ${styles.span2}`}>
              <div>주소</div>
              <div>{detail.address}</div>
            </div>
            <div className={styles.wrap}>
              <div>주민번호</div>
              <div>{detail.registration_number}******</div>
            </div>
          </>
        ) : (
          <div className={styles.span3}>데이터가 없습니다.</div>
        )}
      </div>
    </div>
  );
}
