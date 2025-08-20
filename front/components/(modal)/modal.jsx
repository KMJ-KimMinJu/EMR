// components/(modal)/modal.jsx
"use client";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import styles from "./modal.module.css";

export default function Modal({
  open,
  onClose,
  children,
  scope = "container", // "viewport" | "container"
  fit = "fill", // "content"  | "fill"
  portalTo = "#main-modal-root",
  className, // ✅ 전달받기
}) {
  const [mount, setMount] = useState(null);
  useEffect(() => {
    if (!open) return;
    const el = portalTo ? document.querySelector(portalTo) : null;
    setMount(el || null);
  }, [open, portalTo]);

  if (!open) return null;

  const overlayCls =
    scope === "container" ? styles.overlayInContainer : styles.overlayFixed;
  const dialogCls = fit === "fill" ? styles.dialogFill : styles.dialog;

  const node = (
    <div className={`${overlayCls} ${className || ""}`} onClick={onClose}>
      <div className={dialogCls} onClick={(e) => e.stopPropagation()}>
        <button className={styles.close} onClick={onClose} aria-label="닫기">
          ×
        </button>
        {children}
      </div>
    </div>
  );
  return mount ? createPortal(node, mount) : node;
}
