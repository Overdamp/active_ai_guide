#!/usr/bin/env bash
# สร้าง dataset ตัวอย่างแบบ persistent ชื่อ "ptt_fiftyone_demo"
# พร้อม field ที่คำนวณจากเครื่องมือ FiftyOne หลายตัว เพื่อเปิดดูใน App ได้ทันที
#   - compute_metadata / blur+glare (OpenCV) / embeddings
#   - compute_uniqueness / compute_similarity / compute_visualization(pca)
# ใช้: docker/scripts/seed-demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # -> docker/
docker compose up -d

docker compose exec -T -w /workspace/docker fiftyone python - <<'PY'
import sys; sys.path.insert(0, "tests")
from pathlib import Path
import cv2, numpy as np
import fiftyone as fo, fiftyone.brain as fob
import _helpers as H

fo.config.show_progress_bars = False
NAME = "ptt_fiftyone_demo"
ds = fo.Dataset(NAME, persistent=True, overwrite=True)

paths = H.list_ptt_images(split="valid", limit=25)
kind = "ptt"
if not paths:
    paths = H.make_synthetic_images(Path("/workspace/docker/.demo_images"), n=25)
    kind = "synthetic"
ds.add_samples([fo.Sample(filepath=str(p)) for p in paths])
ds.info["kind"] = kind
ds.save()

# 1) built-in metadata
ds.compute_metadata()

# 2) custom OpenCV: blur + glare
blur, glare = [], []
for s in ds.select_fields("filepath"):
    img = cv2.imread(s.filepath)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    glare.append(float(((hsv[:, :, 2] >= 240) & (hsv[:, :, 1] <= 40)).mean()))
ds.set_values("blur_score", blur)
ds.set_values("glare_ratio", glare)

# 3) embeddings (offline) -> brain runs
embs = H.embeddings_for([s.filepath for s in ds])
ds.set_values("embedding", [e.tolist() for e in embs])
fob.compute_uniqueness(ds, embeddings="embedding")
fob.compute_similarity(ds, embeddings="embedding", brain_key="img_sim")
fob.compute_visualization(ds, embeddings="embedding", method="pca",
                          brain_key="img_viz", num_dims=2)

print(f"สร้าง dataset '{NAME}' ({kind}) : {len(ds)} ภาพ")
print("fields :", [f for f in ds.get_field_schema() if f not in ('id','filepath','tags','metadata','created_at','last_modified_at')])
print("brain  :", ds.list_brain_runs())
print("เปิดดู :  docker/scripts/launch-app.sh", NAME)
PY
