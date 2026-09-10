"""
Lab B (เสาที่ 1 ต่อ): ตรวจจับภาพซ้ำจริง (Deduplication) & ค้นหาภาพแปลกใหม่ (Uniqueness)
----------------------------------------------------------------------------------------
ทดสอบกับภาพถ่ายจริง 25 ภาพจากชุดข้อมูลของ ปตท. (ไม่มีการจำลองภาพหรือ Crop ใดๆ ทั้งสิ้น!)

กระบวนการ:
  1. Image Embedding: สกัดเวกเตอร์ 576 มิติจากภาพจริง 25 ภาพด้วย MobileNetV3 (Offline)
  2. Pairwise Cosine Similarity: คำนวณความคล้ายคลึงระหว่างทุกคู่ภาพ (25 x 25 = 300 คู่)
  3. Real Near-Duplicate Detection: หาคู่ภาพที่ช่างหน้างานถ่ายรัวซ้ำมุมเดิม (เช่น เฟรม 003201 กับ 003202)
  4. Real Uniqueness Ranking: จัดลำดับภาพที่แปลกใหม่ แตกต่างจากภาพอื่นๆ ในโรงงานมากที่สุด
"""

import cv2
import numpy as np
import os
import glob
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ที่อยู่ชุดข้อมูลของโปรเจกต์นี้ (แก้ผ่าน env PTT_DATASET_DIR ได้)
DATASET_DIR = os.environ.get(
    "PTT_DATASET_DIR",
    "/home/luke/ai_training/PTT_ai_mini/datasets/active_learning_split/seed_dataset",
)

# 1. โหลดโมเดลสกัดเวกเตอร์ MobileNetV3 (Offline จากแคชในเครื่อง)
print("⏳ กำลังโหลดโมเดล MobileNetV3 (Offline)...")
model = models.mobilenet_v3_small(weights="DEFAULT")
model.classifier = torch.nn.Identity()  # ดึงเวกเตอร์ขนาด 576 มิติ
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. กวาดภาพถ่ายจริง 25 ภาพจากชุดข้อมูล ปตท.
image_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "**", "*.jpg"), recursive=True))[:25]
print(f"📁 พบภาพถ่ายจริงจากชุดข้อมูล ปตท. ทั้งหมด: {len(image_paths)} ภาพ\n")

print("--- 🔍 สเต็ปที่ 1: สกัดเวกเตอร์จากภาพถ่ายจริง (Feature Embeddings) ---")
vectors = []
valid_names = []

for idx, p in enumerate(image_paths, 1):
    img = cv2.imread(p)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0)
    with torch.no_grad():
        v = model(tensor).squeeze().numpy()
    v = v / np.linalg.norm(v)  # Normalized Vector
    vectors.append(v)
    valid_names.append(os.path.basename(p))
    print(f"   [{idx:02d}/25] สกัดเวกเตอร์สำเร็จ: {os.path.basename(p)[:45]}...")

# ==============================================================================
# สเต็ปที่ 2: ตรวจจับภาพซ้ำจริงในชุดข้อมูล (Near-Duplicate Detection)
# ==============================================================================
SIMILARITY_THRESHOLD = 0.90  # ถ้าความคล้ายคลึงกันเกิน 90% = ภาพซ้ำช็อตเดียวกัน

print("\n--- ✂️ สเต็ปที่ 2: ตรวจหาภาพถ่ายรัวซ้ำมุมเดิมในชุดข้อมูลจริง ---")
n = len(vectors)
is_duplicate = [False] * n
duplicate_pairs = []

for i in range(n):
    if is_duplicate[i]:
        continue
    for j in range(i + 1, n):
        if is_duplicate[j]:
            continue
        # คำนวณ Cosine Similarity (Dot Product ของ normalized vectors)
        sim = float(np.dot(vectors[i], vectors[j]))
        if sim >= SIMILARITY_THRESHOLD:
            duplicate_pairs.append((sim, valid_names[i], valid_names[j]))
            is_duplicate[j] = True  # ทำเครื่องหมายตัดภาพ j ทิ้ง

if duplicate_pairs:
    for sim, keeper, dropped in duplicate_pairs:
        print(f"🚨 ตรวจพบภาพซ้ำจริงหน้างาน! (ความคล้ายคลึง: {sim * 100:.2f}%)")
        print(f"   • ภาพหลักที่เก็บไว้ (Keeper)  : {keeper}")
        print(f"   • ภาพซ้ำที่คัดทิ้ง (Duplicate) : {dropped}")
        print("   ➔ [ผลลัพธ์]: ตัดภาพซ้ำทิ้ง ไม่ส่งไปให้คนวาดกล่องใน CVAT ซ้ำซาก\n")
else:
    print("ไม่พบคู่ภาพที่มีความคล้ายเกิน 90% ในกลุ่ม 25 ภาพนี้")

# ==============================================================================
# สเต็ปที่ 3: จัดลำดับภาพตามความแปลกใหม่จริง (Uniqueness Ranking)
# ==============================================================================
print("--- 🌟 สเต็ปที่ 3: คำนวณความแปลกใหม่ของภาพจริง (Uniqueness Ranking) ---")
kept_indices = [i for i in range(n) if not is_duplicate[i]]

uniqueness_scores = []
for i in kept_indices:
    # ยิ่งเวกเตอร์ของภาพนี้อยู่ห่างจากภาพอื่นๆ ค่าเฉลี่ยระยะห่างยิ่งสูง = ยิ่งแปลกใหม่
    distances = [1.0 - float(np.dot(vectors[i], vectors[j])) for j in kept_indices if i != j]
    avg_dist = float(np.mean(distances)) if distances else 0.0
    uniqueness_scores.append((valid_names[i], avg_dist))

uniqueness_scores.sort(key=lambda x: x[1], reverse=True)

print("🏆 Top 3 ภาพที่ 'แปลกใหม่และไม่เหมือนใครที่สุด' (ควรส่งให้คนตรวจและเทรนก่อน):")
for rank, (name, score) in enumerate(uniqueness_scores[:3], 1):
    print(f"   อันดับ {rank}: [คะแนนความแปลก {score:.4f}] ➔ {name}")

print("\n📉 ภาพที่ 'หน้าตาจำเจ ซ้ำซากที่สุดในกลุ่ม' (คล้ายกับภาพส่วนใหญ่ในโรงงาน):")
for rank, (name, score) in enumerate(uniqueness_scores[-2:], 1):
    print(f"   อันดับท้าย: [คะแนนความแปลก {score:.4f}] ➔ {name}")

print("\n" + "="*70)
print(f"🎉 สรุปผล Lab B จากภาพถ่ายจริง ปตท.:")
print(f"   • จำนวนภาพที่ตรวจทั้งหมด : {n} ภาพ")
print(f"   • ตรวจพบภาพซ้ำจริงทิ้งไป  : {sum(is_duplicate)} ภาพ")
print(f"   • เหลือภาพตัวแทนกลุ่ม     : {len(kept_indices)} ภาพ")
