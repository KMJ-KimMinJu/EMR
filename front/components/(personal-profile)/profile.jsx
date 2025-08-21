// components/(personal-profile)/profile.jsx
"use client";
import { useEffect, useState, useCallback } from "react";
import styles from "./profile.module.css";
import { APIADDRESS } from "../../app/apiAddress";
import Modal from "../(modal)/modal";
export default function Profile({ patientId }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [firstPredict, setFirstPredict] = useState(null);
  const [secondPredict, setSecondPredict] = useState(null);
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
        const res = await fetch(`api/patient/${patientId}`, {
          cache: "no-store",
        });
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
  // 2) SSE 붙이기 (vital/full 콘솔 출력)
  useEffect(() => {
    if (!patientId) return;

    // Express가 4000에서 돌고 있으면 절대경로, Next에서 프록시하면 "/api/stream"로 변경
    const es = new EventSource(
      `/api/prediction/stream?patientId=${encodeURIComponent(patientId)}`
    );

    const onVital = (e) => {
      try {
        const msg = JSON.parse(e.data);
        console.log("[SSE][VITAL]", msg);
        setFirstPredict(msg.predict);
      } catch (err) {
        console.warn("VITAL parse error:", err, e.data);
      }
    };

    const onFull = (e) => {
      try {
        const msg = JSON.parse(e.data);
        console.log("[SSE][FULL ]", msg);
        setSecondPredict(msg);
        console.log(msg);
      } catch (err) {
        console.warn("FULL parse error:", err, e.data);
      }
    };

    es.addEventListener("vital", onVital);
    es.addEventListener("full", onFull);
    es.onerror = (err) => console.warn("[SSE] error:", err);

    return () => {
      es.removeEventListener("vital", onVital);
      es.removeEventListener("full", onFull);
      es.close();
    };
  }, [patientId]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    if (open) window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);
  const handleOpen = useCallback(() => setOpen(true), []);
  const handleClose = useCallback(() => setOpen(false), []);
  const handleSuccess = useCallback(() => {
    setOpen(false);
  }, []);
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
            <div className={`${styles.predictMsgWrap} ${styles.spanFull}`}>
              <div
                className={
                  firstPredict === "고위험군"
                    ? styles.firstPredictMsgRed
                    : styles.firstPredictMsg
                }
              >
                {firstPredict}
              </div>
              {secondPredict && (
                <>
                  <Modal
                    open={open}
                    onClose={handleClose}
                    scope="container"
                    fit="fill"
                    portalTo="#main-modal-root" // ✅ 메인 기준으로 덮음
                    className={styles.modal} // ✅ 이제 제대로 전달됨
                  >
                    <h1 className={styles.predictData}>
                      {secondPredict.ICU_hours}시간이내에 이송될 확률{" "}
                      {secondPredict.ICU_percent}%
                    </h1>
                    <table className={styles.predictTable}>
                      <thead>
                        <tr>
                          <th>질병명</th>
                          <th>확률</th>
                          <th>근거</th>
                        </tr>
                      </thead>
                      <tbody>
                        {secondPredict?.diseases?.map((d, idx) => (
                          <tr key={idx}>
                            <td className={styles.diseaseCell}>{d.name}</td>
                            <td className={styles.percentCell}>
                              <div className={styles.percentWrap}>
                                <div className={styles.percentTrack}>
                                  <div
                                    className={styles.percentFill}
                                    style={{ "--p": `${d.percent ?? 0}%` }}
                                  />
                                </div>
                                <span className={styles.percentText}>
                                  {d.percent ?? 0}%
                                </span>
                              </div>
                            </td>
                            <td
                              className={styles.basisCell}
                              title={d.basis} /* 전체 내용 툴팁으로 */
                            >
                              {d.basis}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Modal>
                  <button
                    className={styles.predictButton}
                    onClick={() => handleOpen()}
                  >
                    {secondPredict.ICU_hours}시간이내에 이송될 확률{" "}
                    {secondPredict.ICU_percent}%
                  </button>
                </>
              )}
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
