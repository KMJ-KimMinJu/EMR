import styles from "./regist-button.module.css";

export default function RegistButton({ onClick }) {
  return (
    <button onClick={onClick} className={styles.button}>
      환자등록
    </button>
  );
}
