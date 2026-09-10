# CLAUDE.md — PTT_ai_mini

> ภาษาโปรเจกต์: **ไทย** (คอมเมนต์ในโค้ด, ข้อความ `print`, และเอกสาร เขียนภาษาไทย)
> ไฟล์นี้คือบริบทเดียวที่ต้องอ่านก่อนเริ่มงานในโปรเจกต์นี้

---

## โปรเจกต์นี้คืออะไร

`PTT_ai_mini` = **Sandbox / ชุด mini-labs สำหรับศึกษา** ระบบ **PTT Smart AI Platform**
(ระบบตรวจจับอุปกรณ์ + วิเคราะห์ความปลอดภัยในโรงแยกก๊าซธรรมชาติ ปตท. ด้วย YOLO12, 26 คลาส)

- **ไม่ใช่ระบบโปรดักชัน** — เป็นโค้ดตัวอย่างแบบ self-contained ให้วิศวกร AI/MLOps คนใหม่
  เข้าใจ pipeline ทั้งหมดได้ใน 1–2 วัน ก่อนไปแตะ repo จริง
- repo จริงอยู่คนละที่: `/home/luke/ai_training/PTT_smart_ai_platform/` — **โปรเจกต์นี้ไม่แก้ไฟล์ใน repo นั้น**
- remote: `https://github.com/Overdamp/active_ai_guide`

### ทิศทางจากหัวหน้างาน (กรอบของทุกงานในโปรเจกต์นี้)
> โฟกัส 100% ที่ **เครื่องมือ AI Pipeline**: Data Filtering → Annotation/CVAT → Retraining → Diagnostics & Action Directives
> **ห้ามเสียเวลากับ** Fullstack Web UI, Frontend, หรือ DB CRUD ทั่วไป (มีทีมอื่นดูแล)

---

## เสาหลัก 4 ต้น และไฟล์ที่เกี่ยวข้อง

| เสา | โฟลเดอร์ | ไฟล์ | สาระสำคัญ |
| :-- | :-- | :-- | :-- |
| **1. Data Quality Filtering** | `filter_module/` | `lab_a_blur_glare.py` | Laplacian Variance < 120 = เบลอ; HSV mask (V≥240, S≤40) > 15% = แสงสะท้อนโลหะ |
| **1. Dedup / Uniqueness** | `filter_module/` | `lab_b_dedup_uniqueness.py` | MobileNetV3 embedding 576 มิติ + cosine sim ≥ 0.90 = ภาพซ้ำ |
| **1+2. Hard Sample Mining** | `filter_module/` | `lab_c_hard_samples.py` | รัน YOLO จริงเทียบ GT หา False Negatives (IoU<0.5) + ambiguous (conf 0.25–0.65) → `cvat_tasks_manifest.json` |
| **2. Review State** | `filter_module/` | `lab01_review_state.py` | SQLite state machine: `pending → approved / rejected` |
| **3+4. Retrain & Diagnostics** | `filter_module/` | `lab_d_retrain_and_diagnostics.py` | Leaky Split Guard (บล็อกภาพ test set) + Rules Engine → `diagnostic_action_report.json` (DIR-001 SAHI, DIR-002 Copy-Paste) |
| **ทั้งระบบ** | `fiftyone_module/` | `lab_fiftyone_curation.py` | ผสาน FiftyOne built-in + custom OpenCV fields |
| **4 (ยังว่าง)** | `diagnostic_module/` | — | ที่สำหรับพัฒนาต่อของเสาที่ 4 |

เอกสารประกอบ:
- `docs/about_project.md` — คู่มือ handover ฉบับเต็ม
- `docs/implementation_checklist.md` — checklist เตรียม implement ทั้ง 4 เสา
- `fiftyone_module/fiftyone_complete_guide_th.md` — FiftyOne built-in vs custom

---

## Environment & การรัน

```bash
conda activate ai_training          # Python 3.12, PyTorch 2.5+CUDA, Ultralytics, OpenCV, FiftyOne 1.21
python filter_module/lab_a_blur_glare.py
python filter_module/lab_b_dedup_uniqueness.py
python filter_module/lab_c_hard_samples.py
python filter_module/lab_d_retrain_and_diagnostics.py
python filter_module/lab01_review_state.py
python fiftyone_module/lab_fiftyone_curation.py
```

### External assets ที่ labs พึ่งพา (อยู่นอก repo นี้ — ห้าม hardcode ซ้ำ ให้ import จากที่เดียว)
- โมเดล: `/home/luke/ai_training/PTT_smart_ai_platform/models/PTT_YOLO12n_v11i_Baseline_v1.0.0_best.pt`
- ชุดข้อมูล: `/home/luke/ai_training/PTT_smart_ai_platform/datasets/overall-ptt-object-detection.v11i.yolov11/{train,valid,test}/{images,labels}`
- Lab C/D จะรันไม่ได้ถ้าไม่มีไฟล์ 2 อย่างนี้ → รายงานว่าขาด asset ไม่ใช่ bug ในโค้ด

---

## กติกาการเขียนโค้ด

- คอมเมนต์ + ข้อความ `print` = **ภาษาไทย**; ชื่อ identifier = อังกฤษ
- ทุกฟังก์ชันมี **type hints + docstring** (มาตรฐานในเอกสาร handover)
- **Threshold ทุกตัวต้องเป็นตัวแปร config ไว้บนหัวไฟล์** ห้ามฝังกลางฟังก์ชัน
  (`BLUR_THRESHOLD=120`, `GLARE_RATIO_THRESHOLD=0.15`, `SIMILARITY_THRESHOLD=0.90`, `MIN_SAMPLES_TO_TRIGGER`)
- แต่ละ lab ต้องรันเดี่ยวจบในตัว (self-contained) ไม่ import ข้าม lab
- helper ที่ซ้ำ (`calculate_iou`) — คัดลอกในแต่ละ lab โดยตั้งใจ เพื่อให้อ่านจบในไฟล์เดียว
- **กฎเหล็ก MLOps:** ห้ามนำภาพจาก `test/` split เข้ากระบวนการเทรนเด็ดขาด (Leaky Split Guard)
- ผลลัพธ์ที่ generate (`*.json`, `*_visualized.jpg`) เก็บไว้เป็นตัวอย่างอ้างอิงได้ ไม่ต้องลบ

## สิ่งที่ไม่ต้องทำ

- ไม่สร้าง Web UI / Frontend / REST API / DB CRUD
- ไม่แก้อะไรใน `/home/luke/ai_training/PTT_smart_ai_platform/`
- `.claude/worktrees/` = git worktree ของ repo อื่น (`PTT_smart_ai_platform`) — เพิกเฉย, ถูก gitignore ไว้แล้ว
- `.agents/skills/` = ชุด skill ส่วนตัวของผู้ใช้ (ไม่เกี่ยวโปรเจกต์) — ถูก gitignore ไว้แล้ว
