"""
เทส 11 — ส่งออก/นำเข้า + เชื่อมต่อ CVAT
เครื่องมือ FiftyOne (built-in):
  dataset.export(dataset_type=fo.types.*)   : YOLOv5 / COCO / CVATImage ...
  fo.Dataset.from_dir(...)                  : โหลดกลับ
  dataset.annotate(anno_key, backend="cvat"): ยิงงานขึ้น CVAT (ต้องมี server จริง)
  dataset.load_annotations(anno_key)        : ดึง label ที่คนแก้แล้วกลับเข้ามา
"""
import fiftyone as fo
import pytest


# ---- export / import round-trip (ไม่ต้องมี server) -----------------------
@pytest.mark.needs_dataset
def test_yolov5_export_import_roundtrip(dataset_with_gt, tmp_path):
    out = tmp_path / "yolo_export"
    dataset_with_gt.export(
        export_dir=str(out),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="val",
    )
    assert (out / "dataset.yaml").is_file()

    reloaded = fo.Dataset.from_dir(
        dataset_dir=str(out), dataset_type=fo.types.YOLOv5Dataset, split="val"
    )
    try:
        assert len(reloaded) == len(dataset_with_gt)
        assert reloaded.count("ground_truth.detections") == dataset_with_gt.count(
            "ground_truth.detections"
        )
    finally:
        fo.delete_dataset(reloaded.name)


@pytest.mark.needs_dataset
def test_coco_export(dataset_with_gt, tmp_path):
    out = tmp_path / "coco_export"
    dataset_with_gt.export(
        export_dir=str(out),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
    )
    assert (out / "labels.json").is_file()


# ---- CVAT integration ---------------------------------------------------
def test_cvat_backend_is_registered():
    """ไม่ต้องมี server ก็เช็คได้ว่า FiftyOne รู้จัก backend 'cvat' และตั้งเป็นค่าเริ่มต้น"""
    backends = fo.annotation_config.backends
    assert "cvat" in backends, f"backends ที่มี: {sorted(backends)}"
    assert fo.annotation_config.default_backend == "cvat"
    print("annotation backends:", sorted(backends))


@pytest.mark.needs_cvat
def test_annotate_and_load_from_live_cvat(dataset_with_gt):
    """ต้องตั้ง env: CVAT_URL, CVAT_USERNAME, CVAT_PASSWORD (หรือ FIFTYONE_CVAT_*)"""
    import os

    url = os.environ.get("CVAT_URL")
    if not url:
        pytest.skip("ไม่ได้ตั้ง CVAT_URL — ข้ามการทดสอบกับ CVAT server จริง")

    anno_key = "pttlab_smoke"
    view = dataset_with_gt.limit(2)
    view.annotate(
        anno_key,
        backend="cvat",
        label_field="ground_truth",
        url=url,
        username=os.environ.get("CVAT_USERNAME"),
        password=os.environ.get("CVAT_PASSWORD"),
    )
    try:
        assert anno_key in dataset_with_gt.list_annotation_runs()
        dataset_with_gt.load_annotations(anno_key)
    finally:
        dataset_with_gt.delete_annotation_run(anno_key)
