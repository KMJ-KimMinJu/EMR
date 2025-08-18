"use client";
import { useForm } from "react-hook-form";
import { useState } from "react";
import styles from "./regist-form.module.css";
export default function RegistForm({ onSuccess, onCancel }) {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset,
  } = useForm({
    defaultValues: {
      name: "",
      birth: "",
      age: "",
      sex: "",
      phone: "",
      address: "",
      registeration_number: "",
    },
    mode: "onSubmit",
  });

  const [serverMsg, setServerMsg] = useState(null);

  const onSubmit = async (values) => {
    setServerMsg(null);
    try {
      const res = await fetch("/api/seven", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setServerMsg(json.message || "요청 실패");
        return;
      }
      // 성공 처리
      reset();
      onSuccess?.(); // 모달 닫히게
    } catch {
      setServerMsg("네트워크 오류가 발생했습니다.");
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className={styles.form}>
      <h3 className={styles.title}>환자 등록</h3>

      {/* 1) 이름 */}
      <div className={styles.field}>
        <label className={styles.label} htmlFor="name">
          이름
        </label>
        <input
          id="name"
          className={styles.input}
          {...register("name", { required: "이름은 필수입니다." })}
        />
        {errors.name && <p className={styles.error}>{errors.name.message}</p>}
      </div>

      {/* 2) 생년월일 */}
      <div className={styles.field}>
        <label className={styles.label} htmlFor="birth">
          생년월일
        </label>
        <input
          id="birth"
          type="date"
          className={styles.input}
          {...register("birth", { required: "생년월일은 필수입니다." })}
        />
        {errors.birth && <p className={styles.error}>{errors.birth.message}</p>}
      </div>

      {/* 3) 나이 */}
      <div className={styles.field}>
        <label className={styles.label} htmlFor="age">
          나이
        </label>
        <input
          id="age"
          type="number"
          className={styles.input}
          {...register("age", {
            required: "나이는 필수입니다.",
            valueAsNumber: true,
            min: { value: 0, message: "0 이상이어야 합니다." },
          })}
        />
        {errors.age && <p className={styles.error}>{errors.age.message}</p>}
      </div>

      {/* 4) 성별 */}
      <div className={styles.field}>
        <label className={styles.label} htmlFor="sex">
          성별
        </label>
        <select
          id="sex"
          className={styles.select}
          {...register("sex", { required: "성별을 선택하세요." })}
          defaultValue=""
        >
          <option value="" disabled>
            선택
          </option>
          <option value="M">남</option>
          <option value="F">여</option>
        </select>
        {errors.sex && <p className={styles.error}>{errors.sex.message}</p>}
      </div>

      {/* 5) 연락처 */}
      <div className={styles.field}>
        <label className={styles.label} htmlFor="phone">
          연락처
        </label>
        <input
          id="phone"
          className={styles.input}
          {...register("phone", {
            required: "연락처는 필수입니다.",
            pattern: { value: /^\d{10,11}$/, message: "숫자만 10~11자리" },
          })}
        />
        {errors.phone && <p className={styles.error}>{errors.phone.message}</p>}
      </div>

      {/* 6) 주소 */}
      <div className={styles.field}>
        <label className={styles.label} htmlFor="address">
          주소
        </label>
        <input
          id="address"
          className={styles.input}
          {...register("address", { required: "주소는 필수입니다." })}
        />
        {errors.address && (
          <p className={styles.error}>{errors.address.message}</p>
        )}
      </div>

      {/* 7) 등록번호 */}
      <div className={styles.field}>
        <label className={styles.label} htmlFor="registeration_number">
          등록번호
        </label>
        <input
          id="registeration_number"
          className={styles.input}
          {...register("registeration_number", {
            required: "등록번호는 필수입니다.",
          })}
        />
        {errors.registeration_number && (
          <p className={styles.error}>{errors.registeration_number.message}</p>
        )}
      </div>

      {serverMsg && <p className={styles.serverMsg}>{serverMsg}</p>}

      <div className={styles.actions}>
        <button
          type="button"
          onClick={onCancel}
          className={`${styles.button} ${styles.cancel}`}
        >
          취소
        </button>
        <button
          type="submit"
          disabled={isSubmitting}
          className={`${styles.button} ${styles.submit}`}
        >
          {isSubmitting ? "전송중..." : "저장"}
        </button>
      </div>
    </form>
  );
}
