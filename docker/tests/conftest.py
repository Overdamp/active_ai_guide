"""
Fixtures กลางของชุดเทส FiftyOne
--------------------------------
- ตรวจการเชื่อมต่อ MongoDB ก่อนเริ่ม (ถ้าต่อไม่ได้ = หยุดทั้งชุด พร้อมข้อความชัดเจน)
- สร้าง Dataset ชั่วคราวชื่อไม่ซ้ำ แล้วลบให้อัตโนมัติหลังจบเทส
- เตรียมชุดภาพ (จริงหรือสังเคราะห์) + ground_truth + predictions
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

import _helpers as H


# ---------------------------------------------------------------------------
# 0) เช็ก MongoDB / FiftyOne พร้อมใช้งาน (ทำครั้งเดียวต่อ session)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _fiftyone_ready():
    uri = os.environ.get("FIFTYONE_DATABASE_URI", "(ไม่ได้ตั้ง — จะใช้ mongo ฝังในตัว)")
    try:
        import fiftyone as fo

        fo.config.show_progress_bars = False
        fo.list_datasets()  # บังคับให้เชื่อม DB จริง
    except Exception as exc:  # noqa: BLE001
        pytest.exit(
            "เชื่อมต่อ FiftyOne/MongoDB ไม่ได้\n"
            f"  FIFTYONE_DATABASE_URI = {uri}\n"
            f"  error: {exc}\n"
            "  ตรวจว่า service 'mongo' ขึ้นแล้ว:  docker compose ps",
            returncode=3,
        )
    return uri


@pytest.fixture(scope="session")
def fo_mod(_fiftyone_ready):
    import fiftyone as fo

    return fo


@pytest.fixture(scope="session")
def brain_mod(_fiftyone_ready):
    import fiftyone.brain as fob

    return fob


@pytest.fixture(scope="session")
def class_names() -> list[str]:
    return H.read_class_names()


# ---------------------------------------------------------------------------
# 1) แหล่งภาพ: จริง (ถ้ามี mount) หรือสังเคราะห์
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def image_source(tmp_path_factory):
    """คืน dict: {paths: [...], kind: 'ptt'|'synthetic'}"""
    real = H.list_ptt_images(split="valid", limit=12)
    if real:
        return {"paths": real, "kind": "ptt"}
    synth_dir = tmp_path_factory.mktemp("synthetic_images")
    return {"paths": H.make_synthetic_images(Path(synth_dir), n=12), "kind": "synthetic"}


@pytest.fixture
def unique_name() -> str:
    return f"pttlab_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def dataset(fo_mod, unique_name, image_source):
    """Dataset เปล่า + ภาพ (ยังไม่มี label) — ลบอัตโนมัติหลังเทส"""
    ds = fo_mod.Dataset(unique_name)
    ds.add_samples([fo_mod.Sample(filepath=str(p)) for p in image_source["paths"]])
    ds.info["kind"] = image_source["kind"]
    ds.save()
    yield ds
    try:
        fo_mod.delete_dataset(unique_name)
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def dataset_with_gt(fo_mod, unique_name, class_names):
    """Dataset + ground_truth (Detections) จาก label YOLO จริง
    ต้องมี dataset จริง mount เข้ามา ไม่งั้น skip
    """
    pairs = []
    for img in H.list_ptt_images(split="valid", limit=12):
        lbl = H.PTT_DATASET_DIR / "valid" / "labels" / (img.stem + ".txt")
        if lbl.is_file():
            pairs.append((img, lbl))
    if len(pairs) < 4:
        pytest.skip("ต้องมี dataset จริงของ ปตท. (ภาพ + label .txt) — ดู README ส่วน mount")

    ds = fo_mod.Dataset(unique_name)
    samples = []
    for img, lbl in pairs:
        s = fo_mod.Sample(filepath=str(img))
        s["ground_truth"] = H.yolo_txt_to_detections(lbl, class_names)
        samples.append(s)
    ds.add_samples(samples)
    ds.save()
    yield ds
    try:
        fo_mod.delete_dataset(unique_name)
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def dataset_with_preds(fo_mod, dataset_with_gt, class_names):
    """เพิ่มฟิลด์ predictions (สังเคราะห์จาก ground_truth) ให้ dataset_with_gt"""
    ds = dataset_with_gt
    preds = [
        H.perturb_detections(s["ground_truth"], seed=i + 100, names=class_names)
        for i, s in enumerate(ds)
    ]
    ds.set_values("predictions", preds)
    return ds
