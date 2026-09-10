"""
เทส 05 — ความแปลกใหม่ของภาพ (Uniqueness)
เครื่องมือ FiftyOne (built-in): fiftyone.brain.compute_uniqueness()
  - ให้คะแนน 0..1 ต่อ sample: ยิ่งสูง = ยิ่งแตกต่างจากภาพอื่นในชุด (ควรส่งไป label/เทรนก่อน)
  - ยิ่งต่ำ = จำเจ ซ้ำ ๆ กับภาพส่วนใหญ่
รับ embeddings เป็นชื่อ field ได้ (offline) — ไม่งั้นจะไปโหลดโมเดล zoo
"""
from pathlib import Path

import _helpers as H


def _attach_embeddings(dataset):
    paths = [s.filepath for s in dataset]
    dataset.set_values("embedding", [e.tolist() for e in H.embeddings_for(paths)])


def test_compute_uniqueness_range_and_sort(brain_mod, dataset):
    _attach_embeddings(dataset)
    brain_mod.compute_uniqueness(dataset, embeddings="embedding")

    assert "uniqueness" in dataset.get_field_schema()
    lo, hi = dataset.bounds("uniqueness")
    assert 0.0 <= lo <= hi <= 1.0

    ranked = dataset.sort_by("uniqueness", reverse=True)
    top = ranked.first()
    print("ภาพแปลกใหม่ที่สุด:", Path(top.filepath).name, "=", round(top.uniqueness, 4))
    print("ภาพจำเจที่สุด   :", Path(ranked.last().filepath).name, "=", round(ranked.last().uniqueness, 4))


def test_duplicate_lowers_uniqueness(fo_mod, brain_mod, dataset, tmp_path):
    """เพิ่มไฟล์คัดลอก -> ภาพคู่ที่ซ้ำกันควรได้ uniqueness ต่ำกว่าค่ามัธยฐานของชุด"""
    src = Path(dataset.first().filepath)
    copy_path = H.duplicate_file(src, tmp_path)
    dataset.add_sample(fo_mod.Sample(filepath=str(copy_path)))

    _attach_embeddings(dataset)
    brain_mod.compute_uniqueness(dataset, embeddings="embedding")

    vals = dict(zip(
        [Path(s.filepath).name for s in dataset],
        [s.uniqueness for s in dataset],
    ))
    median = sorted(vals.values())[len(vals) // 2]
    assert vals[src.name] <= median
    assert vals[copy_path.name] <= median
