"""
เทส 07 — หา label ที่คนวาดผิด (Label Mistakenness)   [needs_dataset]
เครื่องมือ FiftyOne (built-in): fiftyone.brain.compute_mistakenness()
  - เทียบ predictions (Detections + confidence) กับ ground_truth (Detections)
  - เติม field 'mistakenness' ที่ระดับ sample + ที่ระดับกล่อง GT
  - จุดที่ GT กับโมเดลขัดแย้งกันแรง ๆ = น่าจะคน label ผิด
ต้องมี dataset จริง (label .txt) — ไม่งั้น skip
"""
import pytest

pytestmark = pytest.mark.needs_dataset


def test_compute_mistakenness(brain_mod, dataset_with_preds):
    ds = dataset_with_preds
    brain_mod.compute_mistakenness(
        ds, pred_field="predictions", label_field="ground_truth"
    )

    assert "mistakenness" in ds.get_field_schema()
    lo, hi = ds.bounds("mistakenness")
    assert 0.0 <= lo <= hi <= 1.0

    # FiftyOne เติม attribute 'mistakenness' ที่กล่อง ground_truth ที่จับคู่กับ prediction ได้
    tagged = [
        getattr(d, "mistakenness", None)
        for s in ds
        for d in s.ground_truth.detections
    ]
    assert any(v is not None for v in tagged), "ควรมีกล่อง GT อย่างน้อย 1 กล่องที่ได้คะแนน mistakenness"

    worst = ds.sort_by("mistakenness", reverse=True).first()
    print("sample ที่น่าสงสัยว่า label ผิดที่สุด: mistakenness =", round(worst.mistakenness, 4))


def test_possible_missing_and_spurious_flags(brain_mod, dataset_with_preds):
    """compute_mistakenness ตั้งค่า possible_missing / possible_spurious ให้ด้วย"""
    ds = dataset_with_preds
    brain_mod.compute_mistakenness(ds, "predictions", "ground_truth")
    schema = set(ds.get_field_schema())
    assert "possible_missing" in schema, f"schema ที่ได้: {sorted(schema)}"   # GT ที่อาจตกหล่น (คนลืมวาด)
    assert "possible_spurious" in schema                                       # GT ที่อาจเกินมา (คนวาดมั่ว)
    print("possible_missing รวม =", ds.sum("possible_missing"),
          "| possible_spurious รวม =", ds.sum("possible_spurious"))
