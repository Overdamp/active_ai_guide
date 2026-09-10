"""
เทส 01 — Dataset / Sample / Metadata
เครื่องมือ FiftyOne:
  fo.Dataset, fo.Sample, dataset.add_samples(), dataset.compute_metadata(),
  dataset.persistent, dataset.clone(), dataset.first(), len(dataset)
พิสูจน์ว่า: โครงสร้างข้อมูลหลักของ FiftyOne ทำงานครบ และ compute_metadata()
เติมขนาดภาพ/ชนิดไฟล์ให้อัตโนมัติ (เป็น built-in ที่ lab_fiftyone_curation.py ใช้)
"""
import fiftyone as fo


def test_samples_added(dataset):
    assert len(dataset) >= 8
    s = dataset.first()
    assert s.filepath.endswith(".jpg")


def test_compute_metadata_fills_dimensions(dataset):
    # ก่อนเรียก: metadata ยังว่าง
    assert dataset.first().metadata is None or dataset.first().metadata.width is None

    dataset.compute_metadata()

    for s in dataset:
        assert s.metadata is not None
        assert s.metadata.width > 0 and s.metadata.height > 0
        assert s.metadata.mime_type in ("image/jpeg", "image/jpg")
        assert s.metadata.size_bytes > 0
    print("ตัวอย่างขนาดภาพ:", dataset.first().metadata.width, "x", dataset.first().metadata.height)


def test_persistent_flag_and_reload(dataset, fo_mod):
    name = dataset.name
    dataset.persistent = True
    dataset.save()
    reloaded = fo_mod.load_dataset(name)
    assert len(reloaded) == len(dataset)
    dataset.persistent = False  # ให้ teardown ลบได้ตามปกติ


def test_clone_is_independent(dataset, fo_mod):
    clone_name = dataset.name + "_clone"
    clone = dataset.clone(clone_name)
    try:
        assert len(clone) == len(dataset)
        clone.clear()
        assert len(clone) == 0
        assert len(dataset) >= 8  # ต้นฉบับไม่ถูกกระทบ
    finally:
        fo_mod.delete_dataset(clone_name)


def test_add_sample_field_schema(dataset):
    dataset.add_sample_field("reviewed", fo.BooleanField)
    dataset.set_values("reviewed", [True] * len(dataset))
    assert dataset.count("reviewed") == len(dataset)
    assert "reviewed" in dataset.get_field_schema()
