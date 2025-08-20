"use client";
import { useForm } from "react-hook-form";
import { useState } from "react";
import styles from "./regist-form.module.css";
import { APIADDRESS } from "../../app/apiAddress";
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
      const res = await fetch(`http://${APIADDRESS}:4000/api/patient`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setServerMsg(json.message || "요청 실패");
        return;
      }
      reset();
      onSuccess?.();
    } catch {
      setServerMsg("네트워크 오류가 발생했습니다.");
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className={styles.con}>
      <div className={styles.title}>
        <div>환자 등록</div>
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
      </div>

      <div className={styles.grid}>
        {/* 이름 */}
        <div className={styles.cell}>
          <div className={styles.label}>이름</div>
          <div className={styles.control}>
            <input
              className={styles.input}
              {...register("name", { required: "이름은 필수입니다." })}
              placeholder="홍길동"
            />
            {errors.name && (
              <p className={styles.error}>{errors.name.message}</p>
            )}
          </div>
        </div>

        {/* 생년월일 */}
        <div className={styles.cell}>
          <div className={styles.label}>생년월일</div>
          <div className={styles.control}>
            <input
              type="date"
              className={styles.input}
              {...register("birth", { required: "생년월일은 필수입니다." })}
            />
            {errors.birth && (
              <p className={styles.error}>{errors.birth.message}</p>
            )}
          </div>
        </div>

        {/* 나이 */}
        <div className={styles.cell}>
          <div className={styles.label}>나이</div>
          <div className={styles.control}>
            <input
              type="number"
              className={styles.input}
              {...register("age", {
                required: "나이는 필수입니다.",
                valueAsNumber: true,
                min: { value: 0, message: "0 이상이어야 합니다." },
              })}
              placeholder="30"
            />
            {errors.age && <p className={styles.error}>{errors.age.message}</p>}
          </div>
        </div>

        {/* 성별 */}
        <div className={styles.cell}>
          <div className={styles.label}>성별</div>
          <div className={styles.control}>
            <select
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
        </div>

        {/* 연락처 */}
        <div className={styles.cell}>
          <div className={styles.label}>연락처</div>
          <div className={styles.control}>
            <input
              className={styles.input}
              {...register("phone", {
                required: "연락처는 필수입니다.",
                pattern: { value: /^\d{10,11}$/, message: "숫자만 10~11자리" },
              })}
              placeholder="01012345678"
            />
            {errors.phone && (
              <p className={styles.error}>{errors.phone.message}</p>
            )}
          </div>
        </div>

        {/* 주소 – 넓게 쓰고 싶으면 span2/3 적용 가능 */}
        <div className={`${styles.cell} ${styles.span2}`}>
          <div className={styles.label}>주소</div>
          <div className={styles.control}>
            <input
              className={styles.input}
              {...register("address", { required: "주소는 필수입니다." })}
              placeholder="서울시 강남구 ..."
            />
            {errors.address && (
              <p className={styles.error}>{errors.address.message}</p>
            )}
          </div>
        </div>

        {/* 등록번호 */}
        <div className={styles.cell}>
          <div className={styles.label}>등록번호</div>
          <div className={styles.control}>
            <input
              className={styles.input}
              {...register("registeration_number", {
                required: "등록번호는 필수입니다.",
              })}
              placeholder="예: 123456-7890123"
            />
            {errors.registeration_number && (
              <p className={styles.error}>
                {errors.registeration_number.message}
              </p>
            )}
          </div>
        </div>
      </div>

      {serverMsg && <p className={styles.serverMsg}>{serverMsg}</p>}
    </form>
  );
}
