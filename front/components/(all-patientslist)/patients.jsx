import styles from "./patients.module.css";

export async function getPatientAll() {
  // await new Promise((resolve)=>setTimeout(resolve,1000))
  const response = await fetch("http://localhost:4000/api/patient/list", {
    cache: "force-cache",
  });
  const json = await response.json();
  console.log("[NETWORK FETCH]");
  return json;
}
export default async function Patients() {
  const patients = await getPatientAll();
  return (
    <div className={styles.con}>
      <div className={styles.listdiv}>
        <ul>
          {patients.map((patient) => {
            <li>
              <div>
                <div>
                  <span>{patient.name}</span>
                  <span>
                    {patient.age}/{patient.sex}
                  </span>
                </div>
                <div>{patient.birth}</div>
              </div>
              <div>예측버튼</div>
            </li>;
          })}
        </ul>
      </div>
    </div>
  );
}
