import styles from "./profile.module.css";

export default function Profile() {
  return (
    <div className={styles.con}>
      <div className={styles.title}>환자 진료정보 조회</div>
      <div className={styles.profile}>
        <div className={styles.wrap}>
          <div>등록번호</div>
          <div>1&382</div>
        </div>
        <div className={styles.wrap}>
          <div>성별</div>
          <div>M</div>
        </div>
        <div className={styles.wrap}>
          <div>나이</div>
          <div>18</div>
        </div>
        <div className={styles.wrap}>
          <div>연락처</div>
          <div>01055553333</div>
        </div>
        <div className={styles.wrap}>
          <div>이름</div>
          <div>박밤이</div>
        </div>
        <div className={styles.wrap}>
          <div>생년월일</div>
          <div>20221013</div>
        </div>
        <div className={styles.wrap}>
          <div>주소</div>
          <div>하단 귀여운곳</div>
        </div>
        <div className={styles.wrap}>
          <div>주민번호</div>
          <div>221013-4</div>
        </div>
      </div>
    </div>
  );
}
