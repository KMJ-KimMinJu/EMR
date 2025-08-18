import Patients from "../components/(all-patientslist)/patients";
import Profile from "../components/(personal-profile)/profile";
import styles from "./home.module.css";
import Regacy from "../components/(personal-regacy)/regacy";
export default function test() {
  return (
    <div className={styles.wrap}>
      <Patients className={styles.side} />
      <div className={styles.main}>
        <Profile />
        <Regacy />
        <div></div>
      </div>
    </div>
  );
}
