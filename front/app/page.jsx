"use client";

import { useState, useCallback, useEffect } from "react";
import Patients from "../components/(all-patientslist)/patients";
import Profile from "../components/(personal-profile)/profile";
import Regacy from "../components/(personal-regacy)/regacy";
import RegistForm from "../components/(patient-regist)/patient-regist-form";
import RegistButton from "../components/(all-patientslist)/patients-regist-button";
import Modal from "../components/(modal)/modal";
import styles from "./home.module.css";
import "./global.css";
export default function Test() {
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
    <div className={styles.wrap}>
      <div className={styles.side}>
        <div className={styles.sideHeader}>
          <div>환자 리스트</div>
          <RegistButton onClick={handleOpen} className={styles.button} />
        </div>
        <Patients />
      </div>

      <div className={styles.main}>
        <Profile />
        <Regacy />
        <Modal open={open} onClose={handleClose}>
          <RegistForm onSuccess={handleSuccess} onCancel={handleClose} />
        </Modal>
      </div>
    </div>
  );
}
