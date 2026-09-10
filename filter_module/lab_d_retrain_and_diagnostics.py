"""
Lab D (เสาที่ 3 + เสาที่ 4): กลไกการเทรนซ้ำ (Retraining) & ระบบวิเคราะห์ออกคำแนะนำ (Diagnostics)
-------------------------------------------------------------------------------------------------
ทดสอบบนชุดข้อมูลและโมเดลจริงของ ปตท. (PTT_YOLO12n_Baseline)

ครอบคลุม 2 เสาหลักสุดท้ายที่หัวหน้าต้องการ:
  1. เสาที่ 3: Continuous Retraining Pipeline
     - Automated Trigger Condition: เช็คเงื่อนไขว่าควรเริ่มเทรนซ้ำหรือยัง (Hard Sample Count & Threshold)
     - Leaky Split Guard: ตรวจจับและบล็อก "ข้อมูลรั่วไหล" (ห้ามนำภาพจาก Test Set ไปเทรนเด็ดขาด!)
     - Retrain Config Generator: สร้างพารามิเตอร์คำสั่งเทรน Ultralytics YOLO ที่เหมาะสม
  2. เสาที่ 4: Model Diagnostics & Action Directive Cards
     - Failure Profiler: สถิติคลาสที่ AI พลาดบ่อยที่สุด (False Negatives & Ambiguous Classes)
     - Rules Engine: แปลงผลการวิเคราะห์ทางเทคนิค ออกมาเป็น "การ์ดคำสั่งปฏิบัติการ (Action Cards)"
       เช่น SAHI สำหรับวาล์วจิ๋ว, Copy-Paste Augmentation สำหรับคลาสหายาก, ปรับ Soft-NMS
"""

import os
import glob
import json
import torch
from collections import Counter
from ultralytics import YOLO

# ==============================================================================
# 1. โหลดโมเดลและข้อมูลทดสอบจริง
# ==============================================================================
# ที่อยู่ชุดข้อมูล/โมเดล (แก้ผ่าน env PTT_DATASET_DIR / PTT_MODEL_PATH ได้)
MODEL_PATH = os.environ.get(
    "PTT_MODEL_PATH",
    "/home/luke/ai_training/PTT_smart_ai_platform/models/PTT_YOLO12n_v11i_Baseline_v1.0.0_best.pt",
)
BASE_DATASET = os.environ.get(
    "PTT_DATASET_DIR",
    "/home/luke/ai_training/PTT_ai_mini/datasets/active_learning_split/seed_dataset",
)
valid_img_dir = os.path.join(BASE_DATASET, "valid", "images")
valid_lbl_dir = os.path.join(BASE_DATASET, "valid", "labels")
test_img_dir = os.path.join(BASE_DATASET, "test", "images")

print("⏳ กำลังโหลดโมเดล PTT YOLO12 Baseline...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
print(f"✅ โหลดโมเดลสำเร็จบน {device.upper()} (26 คลาสอุปกรณ์ ปตท.)\n")

# ==============================================================================
# เสาที่ 3: CONTINUOUS RETRAINING PIPELINE & LEAKY SPLIT GUARD
# ==============================================================================
print("="*80)
print("🚀 เสาที่ 3: ระบบตรวจสอบเงื่อนไขการเทรนซ้ำ & ป้องกันข้อมูลรั่ว (Retrain & Safety Guard)")
print("="*80)

# จำลอง Buffer ภาพที่ผ่านการรีวิวจาก CVAT (10 ภาพที่ได้จาก Lab C + 1 ภาพทดสอบพิเศษ)
# โดยแอบใส่ภาพจาก 'test' เข้ามา 1 ภาพเพื่อทดสอบระบบป้องกันข้อมูลรั่วไหล!
sample_tasks = [
    {"image_name": "009717_jpg.rf.b00138baf471bc45befbabddcd78f3d0.jpg", "split": "valid"},
    {"image_name": "010434_jpg.rf.9a50040fb0fd7f30e781228f1c12ef05.jpg", "split": "valid"},
    {"image_name": "010449_jpg.rf.39e9d0eae05eb702a9ff8826f89f0b5a.jpg", "split": "valid"},
    # ภาพตัวอย่างที่แอบหลุดมาจาก Test Split (จำลอง Human Error):
    {"image_name": "Part6-Sin_003152_jpg.rf.69c5b1f214e30093f3381e886e62c164.jpg", "split": "test"}
]

# กฎเหล็ก MLOps: ตรวจสอบ Leaky Split Guard
test_image_basenames = set(os.path.basename(p) for p in glob.glob(os.path.join(test_img_dir, "*.jpg")))

safe_training_buffer = []
leaked_samples = []

for task in sample_tasks:
    fname = task["image_name"]
    if fname in test_image_basenames:
        leaked_samples.append(fname)
    else:
        safe_training_buffer.append(fname)

print(f"📦 จำนวนภาพในบัฟเฟอร์พร้อมเทรน : {len(sample_tasks)} ภาพ")
if leaked_samples:
    print(f"🛑 [LEAKY SPLIT GUARD ALERT]: ตรวจพบข้อมูลจาก Test Set แอบแฝงเข้ามา {len(leaked_samples)} ภาพ!")
    for leak in leaked_samples:
        print(f"   ❌ ปฏิเสธภาพ: {leak[:40]}... (ห้ามนำ Test Set มาเทรนเด็ดขาด ไม่งั้นข้อสอบรั่ว!)")
    print("   ➔ ระบบได้คัดแยกภาพเหล่านี้ทิ้งจาก Retraining Buffer เรียบร้อยแล้ว ✅\n")

# ตรวจสอบว่าจำนวน Hard Samples ถึงเกณฑ์เริ่มเทรนอัตโนมัติหรือไม่
MIN_SAMPLES_TO_TRIGGER = 3
if len(safe_training_buffer) >= MIN_SAMPLES_TO_TRIGGER:
    print(f"🎯 [TRIGGER APPROVED]: ตัวอย่างเคสยากที่ปลอดภัยมี {len(safe_training_buffer)} ภาพ (ครบเกณฑ์ขั้นต่ำ {MIN_SAMPLES_TO_TRIGGER})")
    recommended_training_config = {
        "model": MODEL_PATH,
        "data": os.path.join(BASE_DATASET, "data.yaml"),
        "epochs": 30,
        "batch": 16,
        "imgsz": 640,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "lr0": 0.001,           # Low learning rate for fine-tuning
        "copy_paste": 0.3,      # เพิ่มการสังเคราะห์ชิ้นส่วนคลาสที่หายาก
        "mosaic": 1.0,
        "project": "runs/ptt_retrain",
        "name": "yolo12n_active_learned_v2"
    }
    print("📋 [RECOMMENDED RETRAIN CONFIG]:")
    print(json.dumps(recommended_training_config, indent=2))
else:
    print(f"⏳ [TRIGGER HOLD]: จำนวนภาพยังไม่ครบเกณฑ์ขั้นต่ำ ({len(safe_training_buffer)}/{MIN_SAMPLES_TO_TRIGGER}) สะสมต่อใน Buffer")

# ==============================================================================
# เสาที่ 4: MODEL DIAGNOSTICS & ACTION DIRECTIVE CARDS
# ==============================================================================
print("\n" + "="*80)
print("🩺 เสาที่ 4: การวิเคราะห์เชิงลึก & สร้างการ์ดคำสั่งปฏิบัติการ (Action Directives)")
print("="*80)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

# วิเคราะห์หาจุดอ่อนรายคลาส (Per-Class Diagnostics) จากภาพจริง 10 ภาพ
img_paths = sorted(glob.glob(os.path.join(valid_img_dir, "*.jpg")))[:10]
missed_class_counter = Counter()
ambiguous_class_counter = Counter()

for img_p in img_paths:
    fname = os.path.basename(img_p)
    lbl_p = os.path.join(valid_lbl_dir, fname.rsplit(".", 1)[0] + ".txt")
    if not os.path.exists(lbl_p):
        continue
    results = model(img_p, conf=0.25, device=device, verbose=False)[0]
    h, w = results.orig_shape

    gt_boxes = []
    with open(lbl_p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                gt_boxes.append((cid, [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]))

    preds = []
    for b in results.boxes:
        cid = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        preds.append((cid, conf, xyxy))
        if 0.25 <= conf <= 0.65:
            ambiguous_class_counter[model.names[cid]] += 1

    for g_cid, g_box in gt_boxes:
        max_iou = max([calculate_iou(g_box, p[2]) for p in preds], default=0.0)
        if max_iou < 0.5:
            missed_class_counter[model.names[g_cid]] += 1

print("📊 สถิติคลาสอุปกรณ์ที่ AI มีปัญหามากที่สุดจากชุดทดสอบ:")
print("   • Top 3 คลาสที่ AI 'มองข้ามมากที่สุด (False Negatives)':")
for cls_name, cnt in missed_class_counter.most_common(3):
    print(f"     ❌ {cls_name:<20}: พลาด {cnt} จุด")

print("\n   • Top 3 คลาสที่ AI 'เกิดความลังเลสูง (Confidence 0.25 - 0.65)':")
for cls_name, cnt in ambiguous_class_counter.most_common(3):
    print(f"     ⚠️ {cls_name:<20}: ลังเล {cnt} กล่อง")

# ==============================================================================
# สร้าง ACTION DIRECTIVE CARDS (ถอดแบบจาก DirectiveRulesEngine ในโปรดักชัน)
# ==============================================================================
action_cards = []

# กฎที่ 1: ตรวจพบชิ้นส่วนขนาดเล็กและเบียดเสียด (small-valve, manual-valve-stem, flange)
small_dense_classes = ["small-valve", "manual-valve-stem", "flange", "valve-body"]
small_dense_missed = sum(missed_class_counter[c] for c in small_dense_classes)

if small_dense_missed > 10:
    action_cards.append({
        "directive_id": "DIR-001",
        "category": "INFERENCE_POSTPROC",
        "severity": "CRITICAL",
        "title": "เปิดใช้งาน SAHI (Slicing Aided Hyper Inference) สำหรับวาล์วขนาดเล็ก",
        "diagnosis": f"พบอุปกรณ์ขนาดเล็กและหนาแน่น ({', '.join(small_dense_classes)}) หลุดรอดไปถึง {small_dense_missed} จุด เพราะภาพถูกย่อเหลือ 640x640 จนวาล์วกลายเป็นพิกเซลขนาดจิ๋ว",
        "action_steps": [
            "เปิดโหมด SAHI Slicing ในขั้นตอน Inference เพื่อตัดภาพย่อยขนาด 512x512 แบบ Overlap 20%",
            "นำผลลัพธ์มา Merge ด้วย NMS / Soft-NMS เพื่อเก็บรายละเอียดวาล์วจิ๋วระยะไกล",
            "คาดการณ์: ช่วยดึง Recall ของ small-valve และ valve-stem กลับมาได้ 15-25%"
        ],
        "suggested_parameters": {
            "slice_height": 512,
            "slice_width": 512,
            "overlap_height_ratio": 0.2,
            "overlap_width_ratio": 0.2
        }
    })

# กฎที่ 2: ตรวจพบคลาสที่มีความไม่มั่นใจสูง (Ambiguity)
top_ambiguous_class = ambiguous_class_counter.most_common(1)[0][0] if ambiguous_class_counter else "None"
action_cards.append({
    "directive_id": "DIR-002",
    "category": "DATA_AUGMENTATION",
    "severity": "WARNING",
    "title": f"เพิ่มการทำ Copy-Paste Augmentation สำหรับคลาส {top_ambiguous_class}",
    "diagnosis": f"คลาส {top_ambiguous_class} มีค่า Confidence ก้ำกึ่งสูงที่สุด ({ambiguous_class_counter[top_ambiguous_class]} กล่อง) แสดงว่าโมเดลยังสับสนกับ Texture พื้นหลัง",
    "action_steps": [
        f"ตัดชิ้นส่วนของ {top_ambiguous_class} จากภาพที่ผ่าน CVAT แล้วนำไปแปะสุ่ม (Copy-Paste) บนพื้นหลังท่อเหล็กอื่นๆ",
        "ตั้งค่า copy_paste=0.3 ใน hyperparameter การเทรนรอบถัดไป",
        "เน้นส่งภาพกลุ่มนี้ให้คนรีวิวใน CVAT เพิ่มเติม"
    ],
    "suggested_parameters": {
        "copy_paste_prob": 0.3,
        "target_class": top_ambiguous_class
    }
})

print("\n" + "="*80)
print(f"📑 สรุปการ์ดคำสั่งปฏิบัติการสำหรับวิศวกร (สร้างสำเร็จ {len(action_cards)} คำสั่ง):")
print("="*80)
for card in action_cards:
    print(f"\n🏷️ [{card['directive_id']}] {card['title']} (ระดับ: {card['severity']})")
    print(f"   🔍 ปัญหาที่ตรวจพบ: {card['diagnosis']}")
    print(f"   🛠️ แนวทางแก้ไข:")
    for step in card["action_steps"]:
        print(f"      • {step}")
    print(f"   ⚙️ พารามิเตอร์แนะนำ: {json.dumps(card['suggested_parameters'])}")

# บันทึกเป็นรายงานระดับผู้บริหาร (Executive Report)
report_path = "/home/luke/ai_training/PTT_ai_mini/diagnostic_action_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({
        "status": "NEEDS_ATTENTION",
        "retrain_ready": True,
        "safe_buffer_count": len(safe_training_buffer),
        "leaks_blocked": len(leaked_samples),
        "action_directives": action_cards
    }, f, indent=2, ensure_ascii=False)

print(f"\n💾 บันทึกรายงานสรุป Action Directives เรียบร้อยที่: {report_path}")
