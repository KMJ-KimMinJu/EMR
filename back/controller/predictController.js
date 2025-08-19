class predictController {
  constructor() {
    // 필수 필드 정의
    this.REQUIRED_FIELDS = [
      "temperature",
      "heartrate",
      "resprate",
      "o2sat",
      "sbp",
      "dbp",
      "map",
    ];

    this.REQUIRED_LAB_FIELDS = [
      "lactate",
      "glucose",
      "pH",
      "bicarbonate",
      "aniongap",
      "troponin",
      "wbc",
      "creatinine",
      "potassium",
      "bnp",
      "uketone",
      "hydroxybutyrate",
    ];
  }
  async firstTest(req, res, next) {
    try {
      const vital = req.body?.vital;
      if (!vital || typeof vital !== "object" || Array.isArray(vital)) {
        return res.status(400).json({
          success: false,
          error: "VITAL_OBJECT_REQUIRED",
          detail: "`vital` 객체가 필요합니다.",
        });
      }

      const missing = this.REQUIRED_FIELDS.filter((f) => !(f in vital));
      if (missing.length > 0) {
        return res.status(400).json({
          success: false,
          error: "MISSING_FIELDS",
          detail: `다음 필드가 누락됨: ${missing.join(", ")}`,
        });
      }
      res.status(200).json({ success: true, predict: "고위험군" });
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Internal Server Error" });
    }
  }

  async secondTest(req, res, next) {
    try {
      const lab = req.body?.lab;
      if (!lab || typeof lab !== "object" || Array.isArray(lab)) {
        return res.status(400).json({
          success: false,
          error: "LAB_OBJECT_REQUIRED",
          detail: "`lab` 객체가 필요합니다.",
        });
      }

      const missing = this.REQUIRED_LAB_FIELDS.filter((f) => !(f in lab));
      if (missing.length > 0) {
        return res.status(400).json({
          success: false,
          error: "MISSING_FIELDS",
          detail: `누락된 필드: ${missing.join(", ")}`,
        });
      }

      res.status(200).json({
        success: true,
        ICU_percent: 70,
        ICU_hours: 3,
        diseases: [
          {
            name: "sepsis",
            percent: 92,
            basis: "sbp 250 및 뭐시기",
          },
          {
            name: "aki",
            percent: 75,
            basis: "뭐시기뭐사기 하니까 영상 검사 필요",
          },
        ],
      });
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Internal Server Error" });
    }
  }
}

module.exports = new predictController();
