"""
เทส 00 — การเชื่อมต่อพื้นฐาน
เครื่องมือ: fiftyone core, MongoDB backend
พิสูจน์ว่า: import fiftyone ได้, ต่อ MongoDB (service 'mongo') ได้, สร้าง/ลบ dataset ได้
"""
import os


def test_fiftyone_version(fo_mod):
    assert fo_mod.__version__.startswith("1.")
    print("fiftyone version =", fo_mod.__version__)


def test_database_uri_points_to_mongo_service(fo_mod):
    uri = os.environ.get("FIFTYONE_DATABASE_URI", "")
    # ใน compose เราตั้งไว้เป็น mongodb://mongo:27017
    assert uri.startswith("mongodb://"), f"คาดหวัง external mongo URI, ได้: {uri!r}"


def test_list_datasets_roundtrip(fo_mod):
    name = "pttlab_conn_check"
    if name in fo_mod.list_datasets():
        fo_mod.delete_dataset(name)
    ds = fo_mod.Dataset(name)
    assert name in fo_mod.list_datasets()
    fo_mod.delete_dataset(name)
    assert name not in fo_mod.list_datasets()


def test_image_source_kind(image_source):
    print("image source =", image_source["kind"], "| n =", len(image_source["paths"]))
    assert len(image_source["paths"]) >= 8
