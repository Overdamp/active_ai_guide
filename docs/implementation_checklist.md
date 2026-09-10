# ✅ Implementation Checklist — PTT Smart AI Platform

> ใช้เตรียมตัวก่อนลงมือ implement เสาหลักทั้ง 4 ใน repo จริง `PTT_smart_ai_platform/`
> อ้างอิงจาก `docs/about_project.md` และ mini-labs ใน `PTT_ai_mini/`
> อัปเดตล่าสุด: 2026-09-10 (rev 2 — dataset ในตัว repo + docker FiftyOne lab)

---

## 0. เตรียมสภาพแวดล้อม (Environment Setup)

- [ ] `conda activate ai_training` และยืนยัน Python 3.12
- [ ] รัน Smoke Test (จาก `about_project.md` §5.2) — เช็ก `torch.cuda.is_available()`, FiftyOne 1.21, OpenCV, Ultralytics
- [ ] **ชุดข้อมูลอยู่ในตัว repo แล้ว:** `datasets/active_learning_split/{seed,pool}_dataset/{train,valid,test}/{images,labels}` + `data.yaml` (26 คลาส)
- [ ] **ตั้งค่า symlink ครั้งเดียว** (ภาพใน active_learning_split เป็น symlink):
      `ln -sfn /home/luke/ai_training/PTT_smart_ai_platform/datasets/overall-ptt-object-detection.v11i.yolov11 datasets/overall-ptt-object-detection.v11i.yolov11`
      แล้วตรวจ: `find datasets/active_learning_split/seed_dataset/valid/images -type l -xtype f | wc -l` → 773
- [ ] labs อ่าน path จาก env — ตั้งได้ถ้าต้องการ override:
      `PTT_DATASET_DIR` (default `.../active_learning_split/seed_dataset`), `PTT_MODEL_PATH` (default โมเดลใน `PTT_smart_ai_platform/models/`)
- [ ] ตรวจว่ามี GPU ว่างพอ (nvidia-smi) ก่อนเริ่มงานเทรน
- [ ] มี CVAT server พร้อมใช้ + credentials (`url`, user, pass) สำหรับ `dataset.annotate()`
- [ ] ตั้ง git branch ใหม่ต่อ 1 เสา (ห้าม commit ตรง main)

---

## 1. ทำความเข้าใจ Baseline (รัน Mini-Labs ให้ผ่านครบ)

- [ ] `python filter_module/lab_a_blur_glare.py` — เข้าใจสูตร Laplacian Variance + HSV glare mask
- [ ] `python filter_module/lab_b_dedup_uniqueness.py` — เข้าใจ MobileNetV3 embedding + cosine similarity + uniqueness ranking
- [ ] `python filter_module/lab_c_hard_samples.py` — ตรวจว่าสร้าง `cvat_tasks_manifest.json` + `lab_c_hard_sample_visualized.jpg` ได้
- [ ] `python filter_module/lab_d_retrain_and_diagnostics.py` — ตรวจว่าสร้าง `diagnostic_action_report.json` + Leaky Guard บล็อกภาพ test ได้
- [ ] `python filter_module/lab01_review_state.py` — เข้าใจ state machine (`pending → approved/rejected`)
- [ ] `python fiftyone_module/lab_fiftyone_curation.py` — เข้าใจการผสาน FiftyOne built-in + custom fields
- [ ] อ่าน `fiftyone_module/fiftyone_complete_guide_th.md` — จำได้ว่าอะไร built-in / อะไรต้องเขียนเอง
- [ ] **(ทางเลือก) docker FiftyOne lab** — `cd docker && cp .env.example .env && docker compose up -d --build`
      แล้ว `docker/scripts/run-tests.sh` (12 ไฟล์เทส ครอบคลุมเครื่องมือ FiftyOne ทีละตัว) — ดู `docker/README.md`
- [ ] จดค่า baseline metrics ปัจจุบัน (mAP50, mAP50-95, per-class recall) จาก `model.val()` ไว้เทียบผลหลัง retrain

---

## 2. เสาที่ 1 — Data Quality Filtering & Hard Sample Mining

### 2.1 Optical Quality Filter (`data_filter/filter_stage1_blur_glare.py`)
- [ ] ย้าย logic จาก `lab_a` เข้าโมดูลจริง + เพิ่ม batch loop ทั้งโฟลเดอร์
- [ ] แยก threshold เป็น config (`blur_threshold=120`, `glare_ratio=0.15`) — อย่า hardcode
- [ ] เพิ่ม logging ต่อภาพ (blur_score, glare_ratio, verdict) เป็น CSV/JSON
- [ ] กำหนด behavior เมื่อ `cv2.imread` คืน `None` (ไฟล์เสีย/ไม่ใช่ภาพ)
- [ ] เขียน unit test ด้วยภาพคมชัด 1 + ภาพเบลอจริง 1 (path จาก `lab_a`)

### 2.2 Confidence / False Negative Mining (`data_filter/filter_stage2_confidence.py`)
- [ ] ย้าย `calculate_iou` + FN detection จาก `lab_c` เข้าโมดูล (แชร์ helper กับ `lab_d`)
- [ ] รองรับ config: `conf_threshold=0.25`, `iou_match=0.5`, `ambiguous=[0.25, 0.65]`
- [ ] จัดการเคส GT ว่าง / prediction ว่าง โดยไม่ crash
- [ ] batch inference (ไม่ใช่ทีละภาพ) เพื่อความเร็วบน GPU
- [ ] ส่งออก manifest schema เดียวกับ `cvat_tasks_manifest.json` (`pre_annotations` ต้องอยู่ในรูป CVAT รับได้)

### 2.3 Diversity & Deduplication (`data_filter/filter_stage5_diversity.py`)
- [ ] ตัดสินใจ: ใช้ `fob.compute_similarity` (built-in) หรือ MobileNetV3 offline แบบ `lab_b`
- [ ] เพิ่ม logic ลบ/ย้ายไฟล์ซ้ำจริงบนดิสก์ (เก็บภาพแรก ทิ้งภาพหลัง) + log ว่าลบอะไร
- [ ] `fob.compute_uniqueness` / core-set selection สำหรับเลือกภาพหลากหลายส่งเทรน
- [ ] กำหนด `similarity_threshold=0.90` เป็น config

### 2.4 FiftyOne Enricher (`data_filter/fiftyone_enricher.py`)
- [ ] เขียนฟังก์ชันฝัง `blur_score`, `glare_ratio`, `fn_count`, `ambiguous_count` ลง sample fields
- [ ] ใช้ `dataset.set_values()` แบบ bulk (ไม่ loop เซฟทีละ sample)
- [ ] เพิ่ม View helper: `get_hard_sample_view(dataset)` คืน samples ที่ควรส่ง CVAT

### 2.5 Review State Machine (`data_filter/hard_sample_review.py`)
- [ ] สร้างตาราง `hard_sample_reviews` จริง (ดู schema จาก `lab01`) — ใช้ DB ของโปรเจกต์ ไม่ใช่ SQLite ชั่วคราว
- [ ] สถานะ: `pending / approved / rejected` + `reviewed_by` + `reviewed_at`
- [ ] ฟังก์ชัน query "ภาพ approved ที่ยังไม่ถูกส่ง CVAT"
- [ ] กันการเพิ่มภาพซ้ำเข้าคิว (unique constraint บน `image_path` / hash)

---

## 3. เสาที่ 2 — Annotation Workflow & CVAT Integration

- [ ] ทดสอบ `dataset.annotate(anno_key, backend="cvat", ...)` ยิงเข้า CVAT server จริง (End-to-End)
- [ ] แมป `pre_annotations` จาก manifest → FiftyOne `Detections` label ให้ถูก format
- [ ] ยืนยันคลาส 26 ตัวใน CVAT project ตรง index กับ `model.names`
- [ ] ทดสอบดึงผลกลับ `dataset.load_annotations(anno_key)` → อัปเดต ground truth
- [ ] เชื่อมผลที่ดึงกลับ → เปลี่ยนสถานะใน `hard_sample_reviews` เป็น `approved`
- [ ] กำหนด auto-assignment rule / batch naming convention (`active_learning_batch_XX`)
- [ ] จัดการ error: CVAT timeout, task สร้างไม่สำเร็จ, network หลุด

---

## 4. เสาที่ 3 — Continuous Retraining Pipeline

### 4.1 Leaky Split Guard (กฎเหล็ก — ทำก่อน)
- [ ] ย้าย logic จาก `lab_d` เข้า `mlops_integration/` — เช็กทั้ง **basename และ file hash** เทียบ `test/` + `valid/`
- [ ] บล็อก + log + แยกภาพ leak ออกจาก buffer (ห้ามผ่านเข้าเทรนเด็ดขาด)
- [ ] เขียน test: จงใจใส่ภาพ test 1 ภาพ → ต้องถูกปฏิเสธ
- [ ] เพิ่มเช็ก hash (ไม่ใช่แค่ชื่อไฟล์) เพราะ Roboflow rename ไฟล์ได้

### 4.2 Trigger Condition (`mlops_integration/pipeline_trigger.py`)
- [ ] เงื่อนไข: จำนวน `approved` hard samples ≥ `MIN_SAMPLES_TO_TRIGGER` (ปัจจุบัน 3 — ปรับให้สมจริง)
- [ ] สร้าง retrain config (จาก `lab_d`): `lr0=0.001`, `copy_paste=0.3`, `mosaic=1.0`, `epochs`, `batch`, `imgsz=640`
- [ ] รัน subprocess `train_yolo.py` + จับ exit code + stream log
- [ ] ตั้งชื่อ run version อัตโนมัติ (`yolo12n_active_learned_vN`)
- [ ] merge ภาพ approved ใหม่เข้า train split อย่างปลอดภัย (สำเนา ไม่ใช่ย้าย)

### 4.3 Copy-Paste Augmentor (`mlops_integration/copy_paste_augmentor.py`)
- [ ] ตัด crop ของคลาสหายาก/สับสนสูง จากภาพที่ผ่าน CVAT
- [ ] แปะสุ่มบนพื้นหลังท่อเหล็กอื่น + อัปเดต label ให้ตรง
- [ ] กำหนดคลาสเป้าหมายจาก output เสาที่ 4 (`target_class`)

### 4.4 หลังเทรน
- [ ] รัน `model.val()` บน test set → เทียบ mAP/recall กับ baseline (§1)
- [ ] Gate: ถ้า metric แย่ลง → ไม่ promote โมเดลใหม่ + แจ้งเตือน
- [ ] เก็บ weights + metrics + config เป็น 1 artifact ต่อ version

---

## 5. เสาที่ 4 — Model Diagnostics & Action Directives

> ⚠️ `diagnostic_module/` ในโฟลเดอร์ mini ยังว่าง — เริ่มจากพอร์ต `lab_d` เข้ามาเป็นโครง

### 5.1 Per-Class Failure Profiler (`active_diagnostics/diagnostics_hub.py`)
- [ ] รวบรวมสัญญาณจากทุกโมดูล: FN counter, ambiguous counter, blur/glare สะสม
- [ ] คำนวณ confusion matrix เต็ม (ไม่ใช่แค่ FN) — ระบุคู่คลาสที่สับสนกัน
- [ ] per-class report: recall, precision, FN rate, จำนวน sample ในเทรน
- [ ] รันบน valid set เต็ม (ไม่ใช่แค่ 10 ภาพแบบ lab)

### 5.2 Directive Rules Engine (`active_diagnostics/directive_rules_engine.py`)
- [ ] ย้าย rules จาก `lab_d` + แยกเกณฑ์เป็น config (`small_dense_missed > 10` ฯลฯ)
- [ ] **DIR-001 (SAHI)**: trigger เมื่อคลาสเล็ก (`small-valve`, `manual-valve-stem`, `flange`, `valve-body`) FN สูง → แนะ slice 512×512 overlap 0.2
- [ ] **DIR-002 (Copy-Paste)**: trigger จากคลาส ambiguous สูงสุด → ป้อนกลับเข้าเสาที่ 3
- [ ] **DIR-003 (Sensor Maintenance)**: trigger เมื่อ blur สะสมข้ามหลายภาพ/กล้องเดียวกัน → แจ้งทีมซ่อมบำรุง
- [ ] เพิ่ม field: `directive_id`, `category`, `severity`, `diagnosis`, `action_steps`, `suggested_parameters`
- [ ] ส่งออก `diagnostic_action_report.json` (schema เดียวกับ mini-lab) + push ขึ้น dashboard

### 5.3 SAHI Inference Path
- [ ] เพิ่มโหมด inference แบบ slicing (512×512, overlap 20%) + merge ด้วย NMS/Soft-NMS
- [ ] วัดจริงว่า recall คลาสเล็กเพิ่มขึ้นตามที่ DIR-001 อ้าง (15–25%)

---

## 6. Config, Threshold Tuning & Integration

- [ ] รวม threshold ทั้งหมดไว้ใน `platform_core/config_manager.py` หรือ YAML เดียว
- [ ] ทบทวน `blur_threshold=120` กับภาพแดดเที่ยงจริงของโรงแยกก๊าซ (จาก Next Action Items)
- [ ] ทบทวน `glare_ratio=0.15` กับผิวท่อสแตนเลสจริง
- [ ] ทบทวน `similarity_threshold=0.90` และ `MIN_SAMPLES_TO_TRIGGER`
- [ ] เชื่อม 4 เสาเป็น pipeline เดียว (filter → enrich → CVAT → review → retrain → diagnose → loop)
- [ ] เขียน entrypoint / CLI รันทั้ง pipeline แบบ dry-run ได้
- [ ] เขียน integration test แบบ end-to-end บนชุดภาพเล็ก (~20 ภาพ)

---

## 7. Definition of Done

- [ ] ทั้ง 4 เสามีโมดูลจริงใน `src/ptt_diagnostics/` + test ผ่าน
- [ ] Leaky Split Guard มี test พิสูจน์ว่าบล็อกภาพ test set ได้ 100%
- [ ] retrain 1 รอบเต็ม แล้ว metric บน test set ไม่แย่ลงกว่า baseline
- [ ] CVAT round-trip (ส่งขึ้น → วาด → ดึงกลับ) ทำงานจริง
- [ ] `diagnostic_action_report.json` ออกมาแสดงบน dashboard ได้
- [ ] ทุก threshold เป็น config, มี docstring + type hints ครบ (ตามมาตรฐานในเอกสาร handover)
