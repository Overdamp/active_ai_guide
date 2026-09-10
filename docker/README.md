# 🐳 FiftyOne Lab Stack — คู่มือใช้งานตามลำดับ

> โฟลเดอร์นี้: `/home/luke/ai_training/PTT_ai_mini/docker/`
> เป้าหมาย: รัน `fiftyone_module/` (lab + เอกสาร) และชุดเทสที่ครอบคลุม **เครื่องมือ FiftyOne ทีละตัว**
> ในสภาพแวดล้อมที่ทำซ้ำได้ ด้วย Docker Compose (ไม่ต้องลง conda / MongoDB บนเครื่อง)

อ่านไล่จากบนลงล่าง ทำตามทีละขั้น (ขั้นที่ 1 → 8)

---

## 0. ภาพรวมและสถาปัตยกรรม

FiftyOne **ต้องมี MongoDB เสมอ** เพื่อเก็บ metadata ของ Dataset / Sample / Label
(ตัวภาพจริงอยู่บนดิสก์ตามเดิม MongoDB เก็บแค่ "ทะเบียน" และผลการวิเคราะห์)

```
┌──────────────────────────────┐        ┌───────────────────────────────┐
│  service: mongo (mongo:7.0)  │◄──────►│  service: fiftyone            │
│  - เก็บ dataset/sample/label │  27017 │  - python 3.12 + fiftyone 1.21│
│  - volume: mongo_data        │        │  - opencv, sklearn, pytest    │
└──────────────────────────────┘        │  - รัน lab + tests            │
                                        │  - FiftyOne App :5151 ─► host │
                                        └───────────────┬───────────────┘
                                       bind mounts:     │
   ..(PTT_ai_mini) ─► /workspace + /home/luke/ai_training/PTT_ai_mini (rw)
   PTT_smart_ai_platform ─► path เดิม (ro)  ◄────────────┘  (ต้นทางภาพ + โมเดล YOLO)
```

**ชุดข้อมูลของโปรเจกต์นี้:** `datasets/active_learning_split/{seed,pool}_dataset`
(โครงสร้าง YOLO ปกติ `train/valid/test` × `images/labels`) โดย **ภาพเก็บเป็น symlink**
ที่ชี้ไปยัง `datasets/overall-ptt-object-detection.v11i.yolov11/` ซึ่งเป็น symlink อีกทีไปที่
repo `PTT_smart_ai_platform` — labs/tests อ่าน path จาก env **`PTT_DATASET_DIR`**
(ค่าเริ่มต้น = `.../active_learning_split/seed_dataset`)

### โครงสร้างไฟล์

```
docker/
├── docker-compose.yml        # 2 services: mongo + fiftyone
├── Dockerfile                # python:3.12-slim + fiftyone 1.21 + opencv + sklearn + pytest
├── requirements.txt
├── .env.example              # -> คัดลอกเป็น .env
├── pytest.ini                # markers: needs_dataset / needs_cvat / slow
├── README.md                 # ไฟล์นี้
├── scripts/
│   ├── run-tests.sh          # รัน pytest ใน container
│   ├── run-lab.sh            # รัน fiftyone_module/<lab>.py ใน container
│   ├── seed-demo.sh          # สร้าง dataset 'ptt_fiftyone_demo' (persistent) + brain runs
│   └── launch-app.sh         # เปิด FiftyOne App :5151
└── tests/
    ├── conftest.py           # fixtures: เช็ก mongo, สร้าง/ลบ dataset, เตรียม GT/predictions
    ├── _helpers.py           # โหลดภาพจริง/สังเคราะห์, แปลง YOLO->Detections, embedding, synth preds
    └── test_00..test_11 *.py # เครื่องมือ FiftyOne ทีละตัว (ดูตารางข้อ 7)
```

- **ไม่มี** torch / YOLO ใน image นี้ (lab fiftyone + เทสใช้แค่ opencv + numpy + sklearn) → build เร็ว ขนาดเล็ก
- เทสออกแบบให้รัน **offline + ผลกำหนดได้ (deterministic)**:
  - มี dataset จริง mount → ใช้ภาพจริง + label YOLO
  - ไม่มี → สร้างภาพสังเคราะห์ด้วย OpenCV (เทสที่ต้องใช้ label จริงจะขึ้น `SKIPPED`)
  - embedding ใช้สูตรถูก ๆ จากพิกเซล ไม่โหลดโมเดล zoo

---

## 1. สิ่งที่ต้องมีก่อน (Prerequisites)

| ต้องมี | ตรวจด้วย |
| :-- | :-- |
| Docker Engine ≥ 24 + Docker Compose v2 | `docker version` / `docker compose version` |
| พื้นที่ดิสก์ ~2–3 GB (image + mongo volume) | `df -h .` |
| พอร์ต `5151` ว่างบนเครื่อง host | `ss -ltnp | grep 5151` (ควรไม่มีผลลัพธ์) |
| ชุดข้อมูล `datasets/active_learning_split/` — symlink ต้อง resolve ได้ | ดู "ตั้งค่าชุดข้อมูล" ด้านล่าง |
| (ต้นทาง symlink) repo `PTT_smart_ai_platform` พร้อม `datasets/` + `models/` | `ls /home/luke/ai_training/PTT_smart_ai_platform/datasets` |

### ตั้งค่าชุดข้อมูล (ทำครั้งเดียว)

ภาพใน `datasets/active_learning_split/` เป็น symlink สัมพัทธ์ที่ต้องการ sibling ชื่อ
`overall-ptt-object-detection.v11i.yolov11` — สร้างให้ชี้ไป repo ต้นทาง:

```bash
ln -sfn /home/luke/ai_training/PTT_smart_ai_platform/datasets/overall-ptt-object-detection.v11i.yolov11 \
        /home/luke/ai_training/PTT_ai_mini/datasets/overall-ptt-object-detection.v11i.yolov11

# ตรวจว่า resolve ได้
find /home/luke/ai_training/PTT_ai_mini/datasets/active_learning_split/seed_dataset/valid/images \
     -type l -xtype f | wc -l          # ควรได้ 773
```

> ถ้า **ไม่มี** `PTT_smart_ai_platform` บนเครื่องนี้ → symlink จะ dangling, เทสที่ต้องใช้ label
> จริงจะขึ้น `SKIPPED` (ดูขั้นที่ 2 — คอมเมนต์ volume ข้อ 2 ทิ้ง)

---

## 2. ขั้นที่ 1 — ตั้งค่า `.env`

```bash
cd /home/luke/ai_training/PTT_ai_mini/docker
cp .env.example .env
```

แก้ `.env` ตามเครื่อง:

```ini
PTT_PLATFORM_DIR=/home/luke/ai_training/PTT_smart_ai_platform   # ต้นทางภาพ (ที่ symlink ชี้ไป) + โมเดล YOLO
FIFTYONE_APP_PORT=5151                                          # พอร์ต host ของ FiftyOne App
```

> ชุดข้อมูลที่ labs/tests ใช้ = `datasets/active_learning_split/seed_dataset` (ตั้งใน compose ผ่าน
> `PTT_DATASET_DIR=/workspace/datasets/active_learning_split/seed_dataset` แล้ว)
> อยากใช้ `pool_dataset` แทน: แก้ค่านั้นใน `docker-compose.yml` หรือ `export PTT_DATASET_DIR=...` ตอนรัน

**กรณีไม่มี repo `PTT_smart_ai_platform`:** เปิด `docker-compose.yml` แล้วคอมเมนต์บรรทัด bind mount ข้อ (2):

```yaml
      # - ${PTT_PLATFORM_DIR:-...}:${PTT_PLATFORM_DIR:-...}:ro
```

---

## 3. ขั้นที่ 2 — Build และ Start

```bash
docker compose up -d --build
```

ครั้งแรกจะ:
1. ดึง `mongo:7.0`
2. build image `ptt-fiftyone-lab:latest` (ลง fiftyone + opencv + sklearn + pytest)
   — ระหว่าง build มีขั้น `import` ตรวจว่าติดตั้งสำเร็จ (fail fast ถ้าพัง)
3. สตาร์ท `mongo` → รอ healthcheck ผ่าน → สตาร์ท `fiftyone` (ค้างไว้ด้วย `sleep infinity`)

---

## 4. ขั้นที่ 3 — ตรวจสุขภาพ Stack

```bash
docker compose ps
```
ต้องเห็น `ptt-fo-mongo` เป็น `healthy` และ `ptt-fo-app` เป็น `running`

```bash
# fiftyone import ได้ + ต่อ mongo ได้
docker compose exec fiftyone python -c "import fiftyone as fo; print(fo.__version__); print(fo.list_datasets())"
```
คาดหวัง: เลขเวอร์ชัน `1.21.x` แล้วตามด้วย `[]` (ยังไม่มี dataset) — **ไม่มี error เรื่องเชื่อมต่อ**

```bash
# dataset จริงถูก mount ไหม
docker compose exec fiftyone bash -lc 'ls "$PTT_DATASET_DIR/valid/images" | head -3'   # ควรเห็นไฟล์ .jpg
docker compose exec fiftyone bash -lc 'python -c "import cv2,glob,os; p=sorted(glob.glob(os.environ[\"PTT_DATASET_DIR\"]+\"/valid/images/*.jpg\"))[0]; print(p, cv2.imread(p) is not None)"'
```

---

## 5. ขั้นที่ 4 — รัน lab เดิมของ `fiftyone_module`

```bash
docker/scripts/run-lab.sh
# = docker compose exec -w /workspace fiftyone python fiftyone_module/lab_fiftyone_curation.py
```

สคริปต์นี้จะ: สร้าง dataset `ptt_mini_curation_demo` → `compute_metadata()` →
คำนวณ blur/glare ด้วย OpenCV → บันทึกเป็น custom fields → กรอง clean view

รัน lab อื่นก็ได้ (ถ้าไฟล์อยู่ใน `fiftyone_module/`):
```bash
docker/scripts/run-lab.sh <ชื่อไฟล์.py>
```

---

## 6. ขั้นที่ 5 — เปิด FiftyOne App (เว็บ)

> ⚠️ dataset ที่สร้างจาก lab/เทส เป็นแบบ **ไม่ persistent** → หายเมื่อสคริปต์จบ
> เปิด App ให้มีของดู ต้อง seed dataset แบบ persistent ก่อน:

```bash
docker/scripts/seed-demo.sh          # สร้าง 'ptt_fiftyone_demo' (persistent) + brain runs หลายตัว
docker/scripts/launch-app.sh                       # list dataset ที่มี
docker/scripts/launch-app.sh ptt_fiftyone_demo
```
รอ ~15–20 วินาที แล้วเปิดเบราว์เซอร์ → **http://localhost:5151**

ใน App จะเห็น: grid ภาพ, ตัวกรองซ้ายมือ (เลือก field เช่น `blur_score`, `uniqueness`),
แท็บ **Embeddings** (จาก `compute_visualization`), เลือกภาพเพื่อดู label / detection

หยุด App: `Ctrl+C` — ถ้าไม่หยุด ใช้ `docker compose restart fiftyone`
(image เป็น slim ไม่มี `pkill`; การ restart service คือวิธีปิด App ที่ชัวร์สุด)

---

## 7. ขั้นที่ 6 — รันชุดเทส (เครื่องมือ FiftyOne ทีละตัว)

```bash
docker/scripts/run-tests.sh                # ทั้งหมด
docker/scripts/run-tests.sh -m "not slow"  # ข้ามเทสช้า (UMAP)
docker/scripts/run-tests.sh -m "not needs_dataset"   # เฉพาะที่ไม่ต้องใช้ label จริง
docker/scripts/run-tests.sh tests/test_04_similarity_near_duplicates.py -v
```
(เบื้องหลัง = `docker compose exec -w /workspace/docker fiftyone pytest ...`)

### เทสแต่ละไฟล์ — ครอบคลุมอะไร

| ไฟล์ | เครื่องมือ FiftyOne ที่พิสูจน์ | ต้องมี label จริง? |
| :-- | :-- | :--: |
| `test_00_connection.py` | เชื่อม MongoDB, `fo.list_datasets`, สร้าง/ลบ dataset | – |
| `test_01_dataset_and_metadata.py` | `fo.Dataset` / `fo.Sample` / `add_samples` / **`compute_metadata()`** / `persistent` / `clone` / `add_sample_field` | – |
| `test_02_custom_fields_blur_glare.py` | `set_values` + **`ViewField` / `match` / `bounds`** — สะท้อน `lab_fiftyone_curation.py` (blur = Variance of Laplacian, glare = HSV mask) | – |
| `test_03_exact_duplicates.py` | **`fob.compute_exact_duplicates()`** (hash เนื้อไฟล์) + logic เก็บตัวแรก/ลบที่เหลือ | – |
| `test_04_similarity_near_duplicates.py` | **`fob.compute_similarity()`**, `sort_by_similarity()`, `index.find_duplicates()` / `duplicates_view()` | – |
| `test_05_uniqueness.py` | **`fob.compute_uniqueness()`** + `sort_by` — ภาพซ้ำ ⇒ uniqueness ต่ำ | – |
| `test_06_hardness.py` | **`fob.compute_hardness()`** (ต้องมี `Classification` + `logits`) — ภาพ ambiguous ⇒ hardness สูง | – |
| `test_07_mistakenness.py` | **`fob.compute_mistakenness()`** — `mistakenness`, `possible_missing`, `possible_spurious` | ✅ |
| `test_08_evaluate_detections.py` | **`dataset.evaluate_detections(method="coco")`** — `mAP()`, `report()`, `eval_tp/fp/fn`, กรอง FN ด้วย `filter_labels` | ✅ |
| `test_09_visualization.py` | **`fob.compute_visualization()`** — `method="pca"` (เสมอ), `method="umap"` (marker `slow`, ต้องลง `umap-learn`) | – |
| `test_10_views_and_aggregations.py` | `match` / `sort_by` / `limit` / `skip` / `exists` / `match_tags` / `filter_labels` + `count_values` / `distinct` / `bounds` / `mean` | บางเทส ✅ |
| `test_11_export_import_and_cvat.py` | `dataset.export()` (**YOLOv5 / COCO**) + `Dataset.from_dir()` round-trip; `dataset.annotate(backend="cvat")` (marker `needs_cvat`) | ✅ (export), CVAT ต้องมี server |

> เทส `needs_dataset` = ต้องอ่าน label จาก `$PTT_DATASET_DIR/valid/labels` ได้ (symlink resolve ได้) ไม่งั้น `SKIPPED` (ไม่ใช่ FAILED)
> เทส `needs_cvat` = ต้องตั้ง env `CVAT_URL`, `CVAT_USERNAME`, `CVAT_PASSWORD`

---

## 8. ตารางสรุป: FiftyOne มีให้ vs ต้องเขียนเอง

(ขยายจาก `fiftyone_module/README.md` — ผูกกับเทสที่พิสูจน์)

| ความสามารถ | FiftyOne built-in | ต้องเขียนเอง | เทส |
| :-- | :-- | :-- | :-- |
| Metadata ภาพ (w/h/mime/size) | `compute_metadata()` | – | 01 |
| คุณภาพภาพ (เบลอ/แสงสะท้อน) | ❌ ไม่มี | Variance of Laplacian + HSV glare mask (OpenCV) แล้ว `set_values()` | 02 |
| ภาพซ้ำเป๊ะ | `fob.compute_exact_duplicates()` | logic ลบไฟล์จริงบนดิสก์ตามกติกาธุรกิจ | 03 |
| ภาพเกือบซ้ำ / ความคล้าย | `fob.compute_similarity()` + `find_duplicates()` | ตัว extractor ของ embeddings (โมเดล) | 04 |
| Diversity / ความแปลกใหม่ | `fob.compute_uniqueness()` | อัลกอริทึม core-set / k-medoids ถ้าต้องการ | 05 |
| ความยาก / outlier ต่อโมเดล | `fob.compute_hardness()` | ผลิต `logits` จากโมเดลมาป้อน | 06 |
| Label คนวาดผิด | `fob.compute_mistakenness()` | (ถ้าใช้ Cleanlab backend ต้องส่ง prob matrix ออกไปคำนวณข้างนอก) | 07 |
| ประเมิน detection (mAP, TP/FP/FN) | `dataset.evaluate_detections()` | – (ใช้แทนโค้ด IoU มือ ๆ ใน lab_c/lab_d) | 08 |
| ลดมิติเพื่อ plot | `fob.compute_visualization()` (pca/umap/tsne) | – | 09 |
| Query / กรอง Hard Sample | `ViewField` / `match` / `filter_labels` / aggregations | เกณฑ์ threshold เชิงโดเมน | 02, 08, 10 |
| ส่งขึ้น/ดึงกลับจาก CVAT | `dataset.annotate()` / `load_annotations()` | auto-assignment / trigger อัตโนมัติ | 11 |
| Uncertainty ขั้นสูง (entropy, margin) | ❌ (มีแค่ query confidence) | คำนวณ Shannon entropy / prediction margin เอง | (ดู `fiftyone_module/fiftyone_complete_guide_th.md` §3.2) |

---

## 9. เดินทีละเครื่องมือ (รันเทสเฉพาะตัว + ดูผลใน App)

รูปแบบ: รันเทส → เปิด App ดู field ที่มันเติมให้

```bash
# ตัวอย่าง: uniqueness
docker compose exec -w /workspace/docker fiftyone python - <<'PY'
import fiftyone as fo, fiftyone.brain as fob, glob
import sys; sys.path.insert(0, "tests")
import _helpers as H
ds = fo.Dataset("demo_uniqueness", persistent=True, overwrite=True)
paths = H.list_ptt_images(limit=15) or H.make_synthetic_images(__import__("pathlib").Path("/tmp/si"), 15)
ds.add_samples([fo.Sample(filepath=str(p)) for p in paths])
ds.set_values("embedding", [e.tolist() for e in H.embeddings_for([s.filepath for s in ds])])
fob.compute_uniqueness(ds, embeddings="embedding")
print(ds.bounds("uniqueness"))
PY

docker/scripts/launch-app.sh demo_uniqueness   # เปิด http://localhost:5151 แล้ว sort ด้วย uniqueness
```

เปลี่ยน `compute_uniqueness` เป็น `compute_hardness`, `compute_visualization`, ฯลฯ ได้ตามต้องการ
(โค้ดต้นแบบอยู่ในไฟล์เทสที่ตรงกัน)

---

## 10. เชื่อมต่อ CVAT (ของจริง)

1. มี CVAT server รันอยู่ (คนละ stack — ดู pillar 2 ของโปรเจกต์)
2. ตั้ง env ก่อนรันเทส:
   ```bash
   docker compose exec \
     -e CVAT_URL=http://<host>:8080 \
     -e CVAT_USERNAME=<user> -e CVAT_PASSWORD=<pass> \
     -w /workspace/docker fiftyone \
     pytest tests/test_11_export_import_and_cvat.py -m needs_cvat -v
   ```
3. Flow จริง: `view.annotate(key, backend="cvat", label_field="ground_truth", url=...)`
   → คนแก้กล่องใน CVAT → `dataset.load_annotations(key)` ดึงกลับ → อัปเดตสถานะใน review state (lab01)

---

## 11. Troubleshooting

| อาการ | สาเหตุ / วิธีแก้ |
| :-- | :-- |
| `pytest` หยุดทันทีพร้อม "เชื่อมต่อ FiftyOne/MongoDB ไม่ได้" | `mongo` ยังไม่ healthy → `docker compose logs mongo`; รอ ~20s แล้วลองใหม่ |
| `ImportError: libGL.so.1` | ใช้ image นี้ (มี `libgl1` แล้ว) อย่ารันบน python host; ถ้าแก้ Dockerfile ให้คง `opencv-python-headless` |
| FiftyOne App เปิดใน browser ไม่ขึ้น | (1) รอ 15–20s ตอนเปิดครั้งแรก (2) ต้อง `--address 0.0.0.0` — สคริปต์ทำให้แล้ว (3) `DatasetNotFoundError` = dataset ไม่ persistent, รัน `seed-demo.sh` ก่อน (4) เช็ก `FIFTYONE_APP_PORT` ไม่ชนพอร์ตอื่น |
| ปิด App ไม่ได้ (`pkill: not found`) | image เป็น slim — ใช้ `docker compose restart fiftyone` |
| เทส `test_07/08/11` ขึ้น `SKIPPED` | symlink ของ dataset ยัง dangling — สร้าง sibling `overall-ptt-object-detection.v11i.yolov11` (ดู "ตั้งค่าชุดข้อมูล") + `PTT_PLATFORM_DIR` ใน `.env` ถูกต้อง แล้ว `docker compose up -d` ใหม่ |
| `compute_visualization(method="umap")` fail | ยังไม่ลง `umap-learn` → ปลดคอมเมนต์ใน `requirements.txt` แล้ว `docker compose build` ใหม่ (หรือรัน `-m "not slow"`) |
| Mongo volume ข้อมูลค้าง/พัง | `docker compose down -v` แล้วเริ่มใหม่ (ลบ `mongo_data` + `fiftyone_home`) |
| Permission denied ตอนเขียนไฟล์ใน `/workspace` | container รันเป็น root; ไฟล์ที่สร้างจะเป็น root บน host — `sudo chown -R $USER .` ถ้าจำเป็น |

---

## 12. Cleanup

```bash
docker compose down          # หยุด container (เก็บ volume/ข้อมูล FiftyOne)
docker compose down -v       # หยุด + ลบ volume ทั้งหมด (mongo_data, fiftyone_home) — เริ่มศูนย์
docker image rm ptt-fiftyone-lab:latest   # ลบ image ถ้าต้องการ
```

---

## 13. นำไปใช้ต่อในระบบจริง

- image นี้ตั้งใจให้เบา (ไม่มี torch) — ในโปรดักชันเพิ่ม `torch`/`ultralytics` แล้วสลับ `cheap_embedding`
  เป็น MobileNetV3 / CLIP (แบบ `filter_module/lab_b_dedup_uniqueness.py`)
- pattern "custom OpenCV field → `set_values` → `ViewField` filter" ใน `test_02` คือรูปแบบเดียวกับ
  `fiftyone_enricher.py` ที่ระบุใน `docs/implementation_checklist.md` §2.4
- `evaluate_detections` (`test_08`) ใช้แทนโค้ดคำนวณ IoU มือ ๆ ใน `lab_c` / `lab_d` ได้เลย
