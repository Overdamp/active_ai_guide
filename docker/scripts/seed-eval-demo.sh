#!/usr/bin/env bash
# สร้าง dataset "ptt_eval_demo" (persistent) สำหรับเรียนรู้ panel "Model Evaluation" ใน FiftyOne App
#   - ภาพจริง + ground_truth จริง (จาก label YOLO ของ ปตท.)
#   - predictions: ตัวอย่างสังเคราะห์จาก GT (ของจริงให้แทนด้วยผล model.predict — ดูหมายเหตุท้ายสคริปต์)
#   - รัน dataset.evaluate_detections(eval_key="eval_v1")  -> เปิดใน App ได้ทันที
# ใช้: docker/scripts/seed-eval-demo.sh [จำนวนภาพ]   (ค่าเริ่มต้น 40)
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose up -d

N="${1:-40}"
docker compose exec -T -e SEED_N="$N" -w /workspace/docker fiftyone python - <<'PY'
import os, sys; sys.path.insert(0, "tests")
from pathlib import Path
import fiftyone as fo
import _helpers as H

fo.config.show_progress_bars = False
NAME = "ptt_eval_demo"
N = int(os.environ.get("SEED_N", "40"))
names = H.read_class_names()

pairs = []
for img in H.list_ptt_images("valid", limit=N):
    lbl = H.PTT_DATASET_DIR / "valid" / "labels" / (img.stem + ".txt")
    if lbl.is_file():
        pairs.append((img, lbl))
if len(pairs) < 4:
    raise SystemExit("ไม่พบ dataset จริงของ ปตท. — ตรวจ PTT_PLATFORM_DIR ใน .env แล้ว up -d ใหม่")

ds = fo.Dataset(NAME, persistent=True, overwrite=True)
samples = []
for img, lbl in pairs:
    s = fo.Sample(filepath=str(img))
    s["ground_truth"] = H.yolo_txt_to_detections(lbl, names)   # <-- label จริง
    samples.append(s)
ds.add_samples(samples)

# predictions (สังเคราะห์จาก GT เพื่อสาธิต UI) — ของจริงแทนด้วย model.predict()
ds.set_values(
    "predictions",
    [H.perturb_detections(s["ground_truth"], seed=i + 7, names=names) for i, s in enumerate(ds)],
)

# ===== Model Evaluation =====
res = ds.evaluate_detections(
    "predictions", gt_field="ground_truth",
    eval_key="eval_v1", method="coco", compute_mAP=True,
)
print(f"\ndataset '{NAME}': {len(ds)} ภาพ | GT boxes={ds.count('ground_truth.detections')} "
      f"| pred boxes={ds.count('predictions.detections')}")
print(f"mAP@[.5:.95] = {res.mAP():.4f}")
print(f"TP={ds.sum('eval_v1_tp')}  FP={ds.sum('eval_v1_fp')}  FN={ds.sum('eval_v1_fn')}")
print("\nเปิดดู:  docker/scripts/launch-app.sh " + NAME)
print("ใน App:  แถบแท็บเหนือ grid -> ' + ' -> Model Evaluation -> เลือก 'eval_v1'")
PY
