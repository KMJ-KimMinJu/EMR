import styles from "./patients.module.css";

export default async function Patients() {
  return (
    <div className={styles.con}>
      <div>환자 리스트</div>
      <div className={styles.listdiv}>
        <ul>
          <li>
            <div>
              <div>
                <span>환자이름</span>
                <span>나이/성별</span>
              </div>
              <div>생년월일</div>
            </div>
            <div>예측버튼</div>
          </li>
          <li>
            <div>
              <div>
                <span>환자이름</span>
                <span>나이/성별</span>
              </div>
              <div>생년월일</div>
            </div>
            <div>예측버튼</div>
          </li>
        </ul>
      </div>
    </div>
  );
}
