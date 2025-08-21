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
  // const [open, setOpen] = useState(false);
  const [post, setPost] = useState(false);
  const [refresh, setRefresh] = useState(0);

  const [selectedPatientId, setSelectedPatientId] = useState(10001);

  const handlePost = useCallback(() => setPost(true), []);
  const handlePostSuccess = useCallback(() => {
    setPost(false);
    setRefresh((n) => n + 1); // ✅ 신호만 보냄
  }, []);
  const handlePostClose = useCallback(() => setPost(false), []);

  return (
    <div className={styles.wrap}>
      <div className={styles.side}>
        <div className={styles.sideHeader}>
          <div></div>
          <div className={styles.patientListTitle}>환자 리스트</div>
          <RegistButton onClick={handlePost} className={styles.button} />
        </div>
        <Patients onSelect={setSelectedPatientId} refresh={refresh} />
      </div>
      <div
        className={`${styles.main} ${styles.container} ${styles.modalRoot} `}
      >
        {post == false ? <Profile patientId={selectedPatientId} /> : ""}
        {!post && <Regacy />}
        {/* <Modal open={open} onClose={handleClose}></Modal> */}
        {post == true ? (
          <RegistForm
            onSuccess={handlePostSuccess}
            onCancel={handlePostClose}
          />
        ) : (
          ""
        )}
        <div id="main-modal-root" />
      </div>
    </div>
  );
}
