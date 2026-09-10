# 🧪 filter_module — Mini-Labs เสาที่ 1–4

ชุดสคริปต์ทดลองแบบ **self-contained** (รันเดี่ยวจบในไฟล์เดียว ไม่ import ข้ามกัน)
สาธิต pipeline ของ PTT Smart AI Platform ตั้งแต่คัดกรองภาพ → ส่ง CVAT → เทรนซ้ำ → ออกคำแนะนำ

---

## ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | เสา | ทำอะไร | ใช้โมเดล YOLO? |
| :-- | :--: | :-- | :--: |
| `lab_a_blur_glare.py` | 1 | Laplacian Variance (เบลอ) + HSV mask V≥240·S≤40 (แสงสะท้อนโลหะ) — คัด 3 เคส | – |
| `lab_b_dedup_uniqueness.py` | 1 | MobileNetV3 embedding 576 มิติ → cosine similarity ≥ 0.90 (ภาพซ้ำ) + uniqueness ranking (25 ภาพ) | – |
| `lab_c_hard_samples.py` | 1 & 2 | รัน YOLO จริงเทียบ GT → False Negatives (IoU<0.5) + ambiguous (conf 0.25–0.65) → `cvat_tasks_manifest.json` + ภาพเปรียบเทียบ | ✅ |
| `lab_d_retrain_and_diagnostics.py` | 3 & 4 | Leaky Split Guard (บล็อกภาพ `test/` split) + retrain config + Rules Engine → `diagnostic_action_report.json` | ✅ |
| `lab01_review_state.py` | 2 | SQLAlchemy/SQLite `hard_sample_reviews`: `pending → approved/rejected` → query เฉพาะ approved | – |

ผลลัพธ์ตัวอย่างที่ commit ไว้: `cvat_tasks_manifest.json`, `diagnostic_action_report.json`, `lab_c_hard_sample_visualized.jpg`
(รันใหม่จะเขียนทับ + เขียนสำเนาที่ root ของ repo ด้วย ซึ่ง gitignore ไว้)

---

## วิธีรัน

```bash
conda activate ai_training        # Python 3.12 + PyTorch + Ultralytics + OpenCV
python filter_module/lab_a_blur_glare.py
python filter_module/lab_b_dedup_uniqueness.py
python filter_module/lab_c_hard_samples.py
python filter_module/lab_d_retrain_and_diagnostics.py
python filter_module/lab01_review_state.py     # สร้าง mini_ptt.db (gitignored)
```

### ชุดข้อมูล / โมเดล (อ่านจาก env — มีค่า default)

| env | default | หมายเหตุ |
| :-- | :-- | :-- |
| `PTT_DATASET_DIR` | `<repo>/datasets/active_learning_split/seed_dataset` | โครง YOLO `train/valid/test` + `data.yaml`; เปลี่ยนเป็น `.../pool_dataset` ได้ |
| `PTT_MODEL_PATH` | `/home/luke/ai_training/PTT_smart_ai_platform/models/PTT_YOLO12n_v11i_Baseline_v1.0.0_best.pt` | ใช้ใน lab_c/lab_d |

**ตั้งค่า symlink ครั้งเดียว** (ภาพใน `active_learning_split` เป็น symlink สัมพัทธ์):
```bash
ln -sfn /home/luke/ai_training/PTT_smart_ai_platform/datasets/overall-ptt-object-detection.v11i.yolov11 \
        /home/luke/ai_training/PTT_ai_mini/datasets/overall-ptt-object-detection.v11i.yolov11
# ตรวจ: ควรได้ 773
find /home/luke/ai_training/PTT_ai_mini/datasets/active_learning_split/seed_dataset/valid/images -type l -xtype f | wc -l
```

ถ้า symlink dangling หรือไม่มีโมเดล → lab_c/lab_d จะ error เรื่องหาไฟล์ไม่เจอ (= ขาด asset ไม่ใช่ bug ในโค้ด)

---

## กติกาของโค้ดในโฟลเดอร์นี้ (ดู `CLAUDE.md`)

- คอมเมนต์ + `print` ภาษาไทย, มี type hints + docstring
- threshold ทุกตัวเป็น config บนหัวไฟล์ (`BLUR_THRESHOLD=120`, `GLARE_RATIO_THRESHOLD=0.15`, `SIMILARITY_THRESHOLD=0.90`, `MIN_SAMPLES_TO_TRIGGER`)
- helper ที่ซ้ำ (`calculate_iou`) — คัดลอกในแต่ละ lab โดยตั้งใจ ให้อ่านจบในไฟล์เดียว
- **กฎเหล็ก:** ห้ามนำภาพจาก `test/` split เข้ากระบวนการเทรน (Leaky Split Guard ใน lab_d)

## เอาไป implement ต่อ

ดู `docs/implementation_checklist.md` §2 (เสาที่ 1) และ §4–5 (เสาที่ 3–4) — แต่ละ lab แมปกับโมดูลจริงใน `src/ptt_diagnostics/` ตามที่ระบุในเช็กลิสต์
