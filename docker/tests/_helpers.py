"""
เครื่องมือช่วยสร้างข้อมูลทดสอบสำหรับชุดเทส FiftyOne ของ PTT_ai_mini
---------------------------------------------------------------------
แนวคิด: เทสทุกไฟล์ต้องรันได้แบบ offline และ deterministic
  - ถ้ามี dataset จริงของ ปตท. mount เข้ามา  -> ใช้ภาพจริง + label .txt (YOLO)
  - ถ้าไม่มี                                  -> สร้างภาพสังเคราะห์ด้วย OpenCV (ยังทดสอบ API ได้)
  - embedding ใช้สูตรถูก ๆ จากพิกเซล (8x8 grayscale = 64 มิติ) ไม่ต้องพึ่ง torch/โมเดล zoo
    ภาพที่เหมือนกันเป๊ะ -> เวกเตอร์เท่ากัน -> ใช้ทดสอบ near-duplicate / similarity ได้จริง
"""
from __future__ import annotations

import glob
import hashlib
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# --- path ของ dataset ในโปรเจกต์นี้ (แก้ผ่าน env PTT_DATASET_DIR ได้) -------
# ค่าเริ่มต้น = active_learning_split/seed_dataset ของ PTT_ai_mini
# (ใน container docker-compose ตั้ง env ให้ชี้มาที่ /workspace/... อยู่แล้ว)
PTT_DATASET_DIR = Path(os.environ.get(
    "PTT_DATASET_DIR",
    "/home/luke/ai_training/PTT_ai_mini/datasets/active_learning_split/seed_dataset",
))


# =========================================================================
# 1. ภาพจริงจากชุดข้อมูล ปตท.
# =========================================================================
def list_ptt_images(split: str = "valid", limit: int = 12) -> list[Path]:
    """คืน path ภาพ .jpg จาก split ที่ระบุ (เรียงชื่อ) — คืน [] ถ้าไม่มี dataset mount"""
    img_dir = PTT_DATASET_DIR / split / "images"
    if not img_dir.is_dir():
        return []
    return [Path(p) for p in sorted(glob.glob(str(img_dir / "*.jpg")))[:limit]]


def ptt_dataset_available(split: str = "valid") -> bool:
    return bool(list_ptt_images(split, limit=1))


def read_class_names() -> list[str]:
    """อ่านรายชื่อคลาสจาก data.yaml (26 คลาสอุปกรณ์ ปตท.) — fallback เป็น class_0..25"""
    yaml_path = PTT_DATASET_DIR / "data.yaml"
    if yaml_path.is_file():
        txt = yaml_path.read_text(encoding="utf-8")
        # parse แบบง่าย ๆ ไม่พึ่ง pyyaml
        if "names:" in txt:
            after = txt.split("names:", 1)[1]
            # รูปแบบ list: [a, b, c]  หรือ  - a\n  - b
            if "[" in after.split("\n", 1)[0]:
                inside = after.split("[", 1)[1].split("]", 1)[0]
                names = [s.strip().strip("'\"") for s in inside.split(",") if s.strip()]
                if names:
                    return names
            names = []
            for line in after.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    names.append(line[2:].strip().strip("'\""))
                elif names:
                    break
            if names:
                return names
    return [f"class_{i}" for i in range(26)]


def yolo_txt_to_detections(label_path: Path, names: list[str]):
    """แปลงไฟล์ label YOLO (cx cy w h normalized) -> fiftyone.Detections
    FiftyOne ใช้ bounding_box = [top-left-x, top-left-y, width, height] แบบ normalized 0..1
    """
    import fiftyone as fo

    dets = []
    if not label_path.is_file():
        return fo.Detections(detections=dets)
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        tlx, tly = cx - w / 2.0, cy - h / 2.0
        label = names[cid] if 0 <= cid < len(names) else f"class_{cid}"
        dets.append(
            fo.Detection(
                label=label,
                bounding_box=[
                    max(0.0, tlx), max(0.0, tly),
                    min(1.0, w), min(1.0, h),
                ],
            )
        )
    return fo.Detections(detections=dets)


# =========================================================================
# 2. ภาพสังเคราะห์ (ใช้เมื่อไม่มี dataset จริง)
# =========================================================================
def make_synthetic_images(dst_dir: Path, n: int = 12) -> list[Path]:
    """สร้างภาพ n ภาพแบบกำหนดผลได้:
      - ภาพ index 3 กับ 4 เหมือนกันเป๊ะ (ใช้ทดสอบ exact-duplicate / near-duplicate)
      - ภาพ index 5 เบลอ (Gaussian blur แรง ๆ) -> blur_score ต่ำ
      - ภาพ index 6 มีสี่เหลี่ยมขาวโพลนกลางภาพ -> glare_ratio สูง
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(1234)
    paths: list[Path] = []
    base_dup = None
    for i in range(n):
        img = rng.randint(40, 210, size=(480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (60 + i * 5, 80), (260 + i * 5, 300), (30, 120, 200), -1)
        cv2.circle(img, (450, 240), 40 + i * 3, (200, 200, 40), -1)
        cv2.putText(img, f"synthetic-{i:02d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)

        if i == 3:
            base_dup = img.copy()
        if i == 4 and base_dup is not None:
            img = base_dup.copy()               # เหมือน index 3 เป๊ะ
        if i == 5:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=9)   # เบลอ
        if i == 6:
            img[120:360, 200:440] = (255, 255, 255)         # แสงสะท้อนจ้า

        p = dst_dir / f"synthetic_{i:02d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


# =========================================================================
# 3. embedding ราคาถูก (ไม่ใช้โมเดล) — ใช้ทดสอบ similarity / uniqueness / visualization
# =========================================================================
def cheap_embedding(image_path: str | Path, dim_side: int = 8) -> np.ndarray:
    """ย่อภาพเป็น grayscale dim_side x dim_side แล้ว flatten + normalize -> เวกเตอร์ (dim_side**2,)
    เพียงพอต่อการทดสอบ API และให้พฤติกรรมถูกต้อง (ภาพเหมือนกัน = เวกเตอร์เท่ากัน)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    small = cv2.resize(img, (dim_side, dim_side), interpolation=cv2.INTER_AREA)
    v = small.astype(np.float32).flatten()
    v = v - v.mean()
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v


def embeddings_for(paths) -> np.ndarray:
    return np.stack([cheap_embedding(p) for p in paths]).astype(np.float32)


# =========================================================================
# 4. สังเคราะห์ prediction (สำหรับ evaluate_detections / compute_mistakenness)
# =========================================================================
def perturb_detections(gt, seed: int, names: list[str]):
    """สร้าง 'predictions' จาก ground_truth โดย:
      - 80% ของกล่องจริง: เก็บไว้ + ขยับกล่องเล็กน้อย + ใส่ confidence 0.30-0.98
      - 20% ของกล่องจริง: ตัดทิ้ง (จำลอง False Negative)
      - เพิ่มกล่องมั่ว 1 กล่อง/ภาพ (จำลอง False Positive) confidence 0.26-0.55
    """
    import fiftyone as fo

    rnd = random.Random(seed)
    out = []
    for d in gt.detections:
        if rnd.random() < 0.20:
            continue
        x, y, w, h = d.bounding_box
        jitter = lambda val: float(min(1.0, max(0.0, val + rnd.uniform(-0.02, 0.02))))
        out.append(
            fo.Detection(
                label=d.label,
                bounding_box=[jitter(x), jitter(y), jitter(w), jitter(h)],
                confidence=round(rnd.uniform(0.30, 0.98), 3),
            )
        )
    # false positive
    out.append(
        fo.Detection(
            label=rnd.choice(names),
            bounding_box=[rnd.uniform(0.1, 0.7), rnd.uniform(0.1, 0.7), 0.08, 0.08],
            confidence=round(rnd.uniform(0.26, 0.55), 3),
        )
    )
    return fo.Detections(detections=out)


def synth_classification(names: list[str], seed: int):
    """สร้าง fiftyone.Classification พร้อม logits (จำเป็นสำหรับ compute_hardness)
    seed คู่  -> logits เกือบเท่ากันทุกคลาส (การกระจายแบน = entropy สูง = hardness สูง)
    seed คี่  -> logits มียอดเดียวชัดมาก (เกือบ one-hot = entropy ต่ำ = hardness ต่ำ)
    """
    import fiftyone as fo

    rnd = np.random.RandomState(seed)
    k = len(names)
    if seed % 2 == 1:                                   # มั่นใจ
        logits = np.full(k, -8.0, dtype=np.float32)
        logits[rnd.randint(k)] = 12.0
    else:                                               # ลังเล / ยาก
        logits = rnd.normal(0.0, 0.5, size=k).astype(np.float32)

    label_idx = int(np.argmax(logits))
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return fo.Classification(
        label=names[label_idx],
        confidence=float(probs[label_idx]),
        logits=logits,
    )


# =========================================================================
# 5. เบ็ดเตล็ด
# =========================================================================
def duplicate_file(src: Path, dst_dir: Path) -> Path:
    """คัดลอกไฟล์ภาพแบบ byte-for-byte ไปที่ชื่อใหม่ (ทดสอบ exact-duplicate)"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (src.stem + "__COPY" + src.suffix)
    shutil.copyfile(src, dst)
    return dst


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
