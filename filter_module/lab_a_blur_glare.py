"""
Lab A (เสาที่ 1): การคัดกรองภาพเชิงกายภาพ (Optical Quality Filter)
-------------------------------------------------------------------
หัวใจสำคัญ:
ก่อนส่งภาพไปให้ AI ตรวจ หรือส่งให้คนวาดกล่องใน CVAT
เราต้องคัดภาพที่ "ใช้งานไม่ได้" ทิ้งก่อน 2 อาการ:
  1. ภาพเบลอจากการสั่นของกล้อง (Motion Blur) -> ใช้ Laplacian Variance
  2. แสงสะท้อนจ้าบนผิวท่อโลหะ (Metal Glare) -> ใช้ HSV Saturation/Value Masking
"""

import cv2
import numpy as np
import os

# เกณฑ์มาตรฐานของ ปตท. (Thresholds อ้างอิงจาก filter_stage1_blur_glare.py)
BLUR_THRESHOLD = 120.0       # ถ้าคะแนน < 120.0 = ภาพเบลอเกินไป
GLARE_RATIO_THRESHOLD = 0.15 # ถ้าแสงสะท้อนเกิน 15% ของภาพ = ภาพจ้าเกินไป

def check_image_quality(image_bgr, label="Image"):
    """ฟังก์ชันตรวจวัดคุณภาพภาพตามสูตรของ ปตท."""
    # 1. ตรวจสอบความเบลอ (Laplacian Variance)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())
    is_blurry = blur_score < BLUR_THRESHOLD

    # 2. ตรวจสอบแสงสะท้อนโลหะ (HSV Glare Ratio)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]  # Value = ความสว่าง (0-255)
    s_channel = hsv[:, :, 1]  # Saturation = ความสดของสี (0-255)

    # แสงสะท้อนจ้าบนโลหะ: สว่างจ้า (V >= 240) และ ขาวซีดไม่มีสี (S <= 40)
    glare_mask = (v_channel >= 240) & (s_channel <= 40)
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    glare_ratio = float(np.count_nonzero(glare_mask) / total_pixels)
    is_glared = glare_ratio > GLARE_RATIO_THRESHOLD

    # สรุปผลการตัดสินใจ
    passed = (not is_blurry) and (not is_glared)
    
    print(f"\n📸 [{label}]")
    print(f"   • ค่าความคมชัด (Blur Score) : {blur_score:7.2f} " + ("❌ (เบลอ! ต่ำกว่า 120)" if is_blurry else "✅ (คมชัด ผ่านเกณฑ์)"))
    print(f"   • แสงสะท้อนจ้า (Glare Ratio) : {glare_ratio * 100:6.2f}% " + ("❌ (แสงจ้าเกิน 15%!)" if is_glared else "✅ (แสงปกติ ผ่านเกณฑ์)"))
    
    if passed:
        print("   🎯 ผลการคัดกรอง: ✅ [ผ่าน] นำภาพนี้ไปให้ AI ตรวจต่อได้")
    else:
        reasons = []
        if is_blurry: reasons.append("ภาพเบลอจากการสั่น")
        if is_glared: reasons.append("แสงสะท้อนโลหะจ้าเกินไป")
        print(f"   🎯 ผลการคัดกรอง: ❌ [คัดทิ้ง] ไม่ส่งต่อ (สาเหตุ: {', '.join(reasons)})")

# ==============================================================================
# รันการทดสอบ 3 กรณีศึกษาจริง
# ==============================================================================

# 1. ภาพจริงที่คมชัดสูง (จากชุดข้อมูล ปตท.)
sharp_path = "/home/luke/ai_training/PTT_smart_ai_platform/datasets/overall-ptt-object-detection.v11i.yolov11/valid/images/Part7-Drift_018267_jpg.rf.31f902b44e079fda71110507d397f908.jpg"
# 2. ภาพจริงที่กล้องสั่น/เบลอจริง (จากชุดข้อมูล ปตท.)
blurry_path = "/home/luke/ai_training/PTT_smart_ai_platform/datasets/overall-ptt-object-detection.v11i.yolov11/valid/images/Part8-Nep_031295_jpg.rf.c0bfedf4fe42d92841e3be55d6c5cf0e.jpg"

if os.path.exists(sharp_path) and os.path.exists(blurry_path):
    # เคส 1: ภาพคมชัด
    sharp_img = cv2.imread(sharp_path)
    check_image_quality(sharp_img, label="เคสที่ 1: ภาพถ่ายหน้างานที่คมชัดปกติ")

    # เคส 2: ภาพเบลอจริงจากหน้างาน
    real_blurry_img = cv2.imread(blurry_path)
    check_image_quality(real_blurry_img, label="เคสที่ 2: ภาพถ่ายจริงที่มีอาการเบลอ (กล้องสั่น)")

    # เคส 3: ภาพคมชัด แต่มีแสงสปอตไลท์สะท้อนผิวท่อจนขาวโพลน (Simulated Metal Glare)
    glared_img = sharp_img.copy()
    h, w, _ = glared_img.shape
    # จำลองแสงสะท้อนจ้า 25% ของพื้นที่ภาพ
    glared_img[int(h*0.2):int(h*0.7), int(w*0.2):int(w*0.7)] = (255, 255, 255)
    check_image_quality(glared_img, label="เคสที่ 3: ภาพที่โดนแสงแฟลช/แสงแดดสะท้อนท่อเหล็ก (Metal Glare)")

print("\n" + "="*60)
print("🎉 สรุปผล Lab A: เราสามารถคัดทิ้งภาพเสียได้ตั้งแต่ต้นทาง ก่อนส่งเข้า Pipeline!")
