"""
Lab FiftyOne Curation: การผสานฟังก์ชันสำเร็จรูปของ FiftyOne ร่วมกับ Custom Logic
-----------------------------------------------------------------------------
ทดสอบบนภาพถ่ายจริงของ ปตท.

วัตถุประสงค์:
  1. แสดงการใช้งาน Built-in Method ของ FiftyOne (เช่น compute_metadata)
  2. แสดงการเขียน Custom Enrichment (คำนวณ Blur, Glare, False Negatives) แล้วบันทึกลง FiftyOne Fields
  3. แสดงการ Query ผ่าน FiftyOne View Expression เพื่อคัดกรอง Hard Samples เตรียมส่งขึ้น CVAT
"""

import os
import glob
import cv2
import fiftyone as fo
from fiftyone import ViewField as F

# 1. ปิด Progress Bar เพื่อความสะอาดของ Log
fo.config.show_progress_bars = False

# ที่อยู่ชุดข้อมูลของโปรเจกต์นี้ (แก้ผ่าน env PTT_DATASET_DIR ได้)
DATASET_DIR = os.environ.get(
    "PTT_DATASET_DIR",
    "/home/luke/ai_training/PTT_ai_mini/datasets/active_learning_split/seed_dataset",
)

# 2. เตรียมภาพถ่ายจริง 5 ภาพจากชุดข้อมูล ปตท.
DATASET_NAME = "ptt_mini_curation_demo"
if fo.dataset_exists(DATASET_NAME):
    fo.delete_dataset(DATASET_NAME)

dataset = fo.Dataset(DATASET_NAME)

img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "valid/images/*.jpg")))[:5]
print(f"📁 โหลดภาพถ่ายจริง {len(img_paths)} ภาพเข้า FiftyOne Dataset...")

# เพิ่มภาพลง FiftyOne
samples = [fo.Sample(filepath=p) for p in img_paths]
dataset.add_samples(samples)

# 3. [BUILT-IN METHOD]: คำนวณ Metadata ภาพอัตโนมัติ (Resolution, Size, MIME)
print("\n--- ⚡ [1. BUILT-IN METHOD]: FiftyOne compute_metadata() ---")
dataset.compute_metadata()
for sample in dataset:
    print(f"   • {os.path.basename(sample.filepath)[:30]}... | ขนาด: {sample.metadata.width}x{sample.metadata.height} | MIME: {sample.metadata.mime_type}")

# 4. [CUSTOM IMPLEMENTATION]: คำนวณความเบลอ (Blur) และแสงสะท้อน (Glare)
print("\n--- 🛠️ [2. CUSTOM LOGIC]: OpenCV Blur Variance & Glare Masking ---")
blur_scores = []
glare_ratios = []

for sample in dataset.select_fields("filepath"):
    img = cv2.imread(sample.filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # คำนวณ Blur Score ด้วย Variance of Laplacian
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_scores.append(blur)
    
    # คำนวณ Glare Ratio (พิกเซลสว่างจัด V > 240 และ S < 40)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    glare_mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 240)
    glare_ratio = float(glare_mask.mean())
    glare_ratios.append(glare_ratio)

# บันทึกค่าที่คำนวณได้ลง FiftyOne Schema Fields
dataset.set_values("blur_score", blur_scores)
dataset.set_values("glare_ratio", glare_ratios)
print(f"✅ บันทึก custom fields ('blur_score', 'glare_ratio') ลง FiftyOne สำเร็จ!")

# 5. [CUSTOM + BUILT-IN]: กรองภาพด้วย FiftyOne View Expressions
print("\n--- 🎯 [3. FIFTYONE VIEW EXPRESSION]: คัดกรองภาพคุณภาพดีสำหรับส่งต่อ ---")
# กรองเฉพาะภาพที่ความคมชัดผ่านเกณฑ์ (> 100) และแสงสะท้อนไม่เกิน 15%
clean_view = dataset.match((F("blur_score") > 100.0) & (F("glare_ratio") < 0.15))
print(f"   • จำนวนภาพทั้งหมด : {len(dataset)} ภาพ")
print(f"   • จำนวนภาพที่ผ่านเกณฑ์ (Clean View) : {len(clean_view)} ภาพ")

for sample in clean_view:
    print(f"     ✅ ผ่าน: {os.path.basename(sample.filepath)[:30]}... | Blur={sample.blur_score:.1f} | Glare={sample.glare_ratio*100:.2f}%")

print("\n" + "="*70)
print("💡 สรุปการเชื่อมโยง:")
print("   • FiftyOne เก่งเรื่องการจัดการ Dataset Schema, View Query, และส่งต่อขึ้น CVAT (dataset.annotate)")
print("   • วิศวกร MLOps เขียน Custom OpenCV / PyTorch เพื่อสกัดฟีเจอร์เชิงลึก แล้วโยนค่ากลับเข้า FiftyOne")
print("="*70)
