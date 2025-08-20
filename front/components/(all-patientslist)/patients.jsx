// components/(all-patientslist)/patients.jsx
"use client";
import { useEffect, useState } from "react";
import { APIADDRESS } from "../../app/apiAddress";
import styles from "./patients.module.css";
export async function getProfile(patientId) {
  const res = await fetch(`/api/patient/list`, {
    cache: "no-store",
  });
  const json = await res.json();
  return json;
}
export default function Patients({ onSelect, refresh = 0 }) {
  const [patients, setPatients] = useState([]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch(`/api/patient/list`, {
          cache: "no-store",
        });
        const json = await res.json();
        if (alive) setPatients(json);
      } catch (e) {
        console.error(e);
      }
    })();
    return () => {
      alive = false;
    };
  }, [refresh]);

  return (
    <div className={styles.con}>
      <div className={styles.listdiv}>
        <ul>
          {patients.map((p) => (
            <li key={p.patientId} onClick={() => onSelect?.(p.patientId)}>
              <div>
                <div>
                  <span>{p.name}</span>
                  <span>
                    {p.age}/{p.sex}
                  </span>
                </div>
                <div>{p.birth}</div>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
