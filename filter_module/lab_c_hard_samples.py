"""
Lab C (เสาที่ 1 ขั้นสูง + เสาที่ 2): การหาภาพเคสยาก (Hard Sample Mining) จากผลทำนาย YOLO จริง
-----------------------------------------------------------------------------------------
ทดสอบกับโมเดลจริง (PTT_YOLO12n_Baseline) และภาพถ่ายจริงในโรงงาน ปตท.

หัวใจสำคัญ:
  ในระบบงานจริง เราจะไม่ส่งภาพทุกภาพไปให้คนวาดกล่องใน CVAT เพราะเปลืองแรงและเสียเวลามาก
  เราจะส่งเฉพาะ "ภาพเคสยาก (Hard Samples)" ที่ AI ทำงานผิดพลาดเท่านั้น ได้แก่:
    1. False Negatives (FN): ของจริงมีอยู่ในภาพ แต่ AI มองข้าม ตรวจไม่เจอเลย! (วิกฤตที่สุด)
    2. Ambiguous Confidence: AI ตรวจเจอ แต่ความมั่นใจต่ำ (0.25 <= conf <= 0.65) ลังเล
    3. ส่งออกรายการเคสยากเป็น CVAT Task Manifest (JSON) ให้ทีม Labeler นำไปเปิดใน CVAT
"""

import os
import glob
import json
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ที่อยู่ชุดข้อมูล/โมเดล (แก้ผ่าน env PTT_DATASET_DIR / PTT_MODEL_PATH ได้)
DATASET_DIR = os.environ.get(
    "PTT_DATASET_DIR",
    "/home/luke/ai_training/PTT_ai_mini/datasets/active_learning_split/seed_dataset",
)

# 1. โหลดโมเดลจริง PTT YOLO12 Baseline
MODEL_PATH = os.environ.get(
    "PTT_MODEL_PATH",
    "/home/luke/ai_training/PTT_ai_mini/models/PTT_YOLO12n_v11i_Baseline_v1.0.0_best.pt",
)
print(f"⏳ กำลังโหลดโมเดล: {os.path.basename(MODEL_PATH)}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
print(f"✅ โหลดโมเดลสำเร็จบนอุปกรณ์: {device.upper()} (รองรับ {len(model.names)} คลาสอุปกรณ์ ปตท.)\n")

# 2. เตรียมชุดภาพและ Label จาก Validation Set ของ ปตท.
BASE_DIR = os.path.join(DATASET_DIR, "valid")
img_paths = sorted(glob.glob(os.path.join(BASE_DIR, "images", "*.jpg")))[:10]
label_dir = os.path.join(BASE_DIR, "labels")

def calculate_iou(boxA, boxB):
    """คำนวณ Intersection over Union (IoU) ระหว่าง 2 กล่อง [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

# 3. รันโมเดลทำนายและวิเคราะห์ Hard Samples
print("="*85)
print(f"{'ชื่อไฟล์ภาพ':<32} | {'ของจริง(GT)':<11} | {'AIเจอ':<7} | {'หลุด(FN)':<9} | {'ลังเล(0.25-0.65)':<16} | {'สถานะ'}")
print("="*85)

cvat_tasks = []
demo_viz_data = None

for img_p in img_paths:
    fname = os.path.basename(img_p)
    lbl_p = os.path.join(label_dir, fname.rsplit(".", 1)[0] + ".txt")
    if not os.path.exists(lbl_p):
        continue

    # รัน YOLO ทำนาย
    results = model(img_p, conf=0.25, device=device, verbose=False)[0]
    h, w = results.orig_shape

    # อ่าน Ground Truth จากไฟล์ Label จริง
    gt_boxes = []
    with open(lbl_p, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gt_boxes.append({"class_id": cls_id, "name": model.names[cls_id], "box": [x1, y1, x2, y2]})

    # ดึงผลทำนายของ AI
    preds = []
    for b in results.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        preds.append({"class_id": cls_id, "name": model.names[cls_id], "conf": conf, "box": xyxy})

    # ค้นหา False Negatives (ของจริงที่ไม่มีกล่องทำนายทับ IoU >= 0.5)
    missed_gt = []
    for gt in gt_boxes:
        max_iou = max([calculate_iou(gt["box"], p["box"]) for p in preds], default=0.0)
        if max_iou < 0.5:
            missed_gt.append(gt)

    # ค้นหากล่องที่ AI ลังเล (Ambiguous: 0.25 <= conf <= 0.65)
    ambiguous_preds = [p for p in preds if 0.25 <= p["conf"] <= 0.65]

    # ประเมินสถานะของภาพ
    is_hard_sample = False
    reasons = []
    if len(missed_gt) > 0:
        is_hard_sample = True
        reasons.append(f"หลุดรอด {len(missed_gt)} ชิ้น")
    if len(ambiguous_preds) > 0:
        is_hard_sample = True
        reasons.append(f"ลังเล {len(ambiguous_preds)} กล่อง")

    if is_hard_sample:
        status = "🚨 ส่งขึ้น CVAT ด่วน!"
        cvat_tasks.append({
            "image_name": fname,
            "image_path": img_p,
            "reasons": reasons,
            "missed_objects_count": len(missed_gt),
            "ambiguous_count": len(ambiguous_preds),
            "pre_annotations": preds  # ส่งกล่องที่ AI พอรู้ไปเป็นไกด์ใน CVAT
        })
    else:
        status = "✅ แม่นยำ ข้ามได้"

    short_name = fname[:30] + ".."
    miss_str = f"{len(missed_gt):2d} ชิ้น" if missed_gt else " - "
    amb_str = f"{len(ambiguous_preds):2d} กล่อง" if ambiguous_preds else " - "
    print(f"{short_name:<32} | {len(gt_boxes):^11} | {len(preds):^7} | {miss_str:^9} | {amb_str:^16} | {status}")

    # บันทึกข้อมูลภาพแรกเพื่อนำไปวาดรูปตัวอย่าง
    if demo_viz_data is None and len(missed_gt) > 0:
        demo_viz_data = (img_p, gt_boxes, preds, missed_gt)

print("="*85)

# 4. บันทึกผลลัพธ์เป็น CVAT Task Manifest
manifest_path = "/home/luke/ai_training/PTT_ai_mini/cvat_tasks_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump({"total_hard_samples": len(cvat_tasks), "tasks": cvat_tasks}, f, indent=2, ensure_ascii=False)
print(f"\n📦 สร้างไฟล์ CVAT Manifest เรียบร้อย: {manifest_path}")
print(f"   • ตรวจพบ Hard Samples ที่ต้องส่งคนตรวจ : {len(cvat_tasks)} จาก {len(img_paths)} ภาพ")

# 5. วาดภาพตัวอย่างความผิดพลาด (Visual Inspection)
if demo_viz_data:
    v_img_p, v_gt, v_preds, v_missed = demo_viz_data
    canvas = cv2.imread(v_img_p)
    
    # วาดกล่องที่ AI ทายแม่น (สีน้ำเงิน/ส้ม)
    for p in v_preds:
        if p["conf"] > 0.65:
            x1, y1, x2, y2 = map(int, p["box"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(canvas, f"{p['name']} ({p['conf']:.2f})", (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

    # วาดกล่องที่ AI พลาดมองไม่เห็น False Negative (สีแดงหนา!)
    for m in v_missed:
        x1, y1, x2, y2 = map(int, m["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(canvas, f"MISSED: {m['name']}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    viz_out = "/home/luke/ai_training/PTT_ai_mini/lab_c_hard_sample_visualized.jpg"
    cv2.imwrite(viz_out, canvas)
    print(f"🎨 บันทึกภาพจำลองความผิดพลาดที่: {viz_out}")
    print("   • กล่องสีแดง = วัตถุของจริงที่ AI ตรวจไม่เจอ (False Negative) ต้องให้คนใน CVAT วาด!")
    print("   • กล่องสีส้ม/ฟ้า = กล่องที่ AI ทายถูกและมั่นใจ (> 0.65)")
