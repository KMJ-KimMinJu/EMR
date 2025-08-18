"use client";
import styles from "./modal.module.css";

export default function Modal({ open, onClose, children }) {
  if (!open) return null;

  const handleOverlayClick = () => onClose?.();
  const stop = (e) => e.stopPropagation();

  return (
    <div className={styles.overlay} onClick={handleOverlayClick}>
      <div className={styles.dialog} onClick={stop}>
        <button className={styles.close} onClick={onClose} aria-label="닫기">
          ×
        </button>
        {children}
      </div>
    </div>
  );
}
