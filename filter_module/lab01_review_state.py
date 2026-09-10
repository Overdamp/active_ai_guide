"""
Lab 01: จำลองระบบ Review State (หัวใจของ Day 2)
-----------------------------------------------
วัตถุประสงค์: 
จำลองดูว่าตาราง `hard_sample_reviews` ในฐานข้อมูล
ช่วยจัดการสถานะของภาพจาก Step 2 (คัดกรอง) ส่งต่อไป Step 3 (CVAT) ได้อย่างไร
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

# 1. สร้างฐานข้อมูลจำลอง SQLite (เป็นไฟล์ mini_ptt.db ไม่ต้องลงโปรแกรมเพิ่ม)
engine = create_engine("sqlite:///mini_ptt.db", echo=False)

class Base(DeclarativeBase):
    pass

# 2. จำลองตาราง HardSampleReview (เหมือนในโปรเจกต์จริง!)
class HardSampleReview(Base):
    __tablename__ = "hard_sample_reviews"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_path: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(default="pending")  # pending / approved / rejected
    reviewed_by: Mapped[Optional[str]] = mapped_column(nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

# สร้างตารางในฐานข้อมูล
Base.metadata.create_all(engine)

# ==============================================================================
# จำลองการทำงานจริง 3 สเต็ป (Step 2 ➔ Step 3)
# ==============================================================================

with Session(engine) as session:
    print("--- 🚀 สเต็ปที่ 1: AI คัดกรองภาพยากมาได้ 3 ภาพ (สถานะเป็น pending) ---")
    samples = [
        HardSampleReview(image_path="pipes/pipe_valve_01.jpg"),
        HardSampleReview(image_path="tanks/tank_rust_02.jpg"),
        HardSampleReview(image_path="gauges/gauge_blurry_03.jpg"),
    ]
    session.add_all(samples)
    session.commit()
    print("✅ บันทึกภาพยาก 3 ภาพเข้าสู่คิวรอตรวจเรียบร้อย!\n")

    print("--- 👤 สเต็ปที่ 2: วิศวกร (somchai) เปิดหน้าเว็บมากดตรวจภาพ ---")
    # สมมุติว่าคุณสมชายตรวจภาพ: อนุมัติ 2 ภาพแรก และปฏิเสธภาพที่ 3
    img1 = session.get(HardSampleReview, 1)
    img1.status = "approved"
    img1.reviewed_by = "somchai_ptt"
    img1.reviewed_at = datetime.now()

    img2 = session.get(HardSampleReview, 2)
    img2.status = "approved"
    img2.reviewed_by = "somchai_ptt"
    img2.reviewed_at = datetime.now()

    img3 = session.get(HardSampleReview, 3)
    img3.status = "rejected"
    img3.reviewed_by = "somchai_ptt"
    img3.reviewed_at = datetime.now()

    session.commit()
    print("✅ คุณสมชายตรวจภาพเสร็จแล้ว (อนุมัติ 2, ปฏิเสธ 1)\n")

    print("--- 🤖 สเต็ปที่ 3: ระบบส่งงานเข้า CVAT (ดึงเฉพาะภาพที่ approved) ---")
    # Worker ทำการค้นหา: "ขอเฉพาะภาพที่มีสถานะ approved เท่านั้น!"
    query = select(HardSampleReview).where(HardSampleReview.status == "approved")
    approved_images = session.scalars(query).all()

    print(f"📦 ภาพที่พร้อมส่งไปวาดกล่องใน CVAT มีทั้งหมด {len(approved_images)} ภาพ:")
    for img in approved_images:
        print(f"   ➔ ส่งภาพ: {img.image_path} (อนุมัติโดย: {img.reviewed_by} เมื่อ {img.reviewed_at.strftime('%H:%M:%S')})")

print("\n🎉 จบการทำงานของ Lab 01 เรียบร้อย!")
