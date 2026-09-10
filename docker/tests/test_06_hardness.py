"""
เทส 06 — ความยากของภาพต่อโมเดล (Sample Hardness)
เครื่องมือ FiftyOne (built-in): fiftyone.brain.compute_hardness()
  - ต้องมี field เป็น fo.Classification ที่ "มี logits"
  - คำนวณจาก entropy/ความไม่ชัดเจนของการกระจายความน่าจะเป็น -> field 'hardness'
  - ภาพ hardness สูง = โมเดลลังเล ควรส่งรีวิว/เทรนซ้ำ
ที่นี่สังเคราะห์ Classification + logits เอง (seed คู่ = 2 ยอดใกล้กัน = hard)
"""
import numpy as np

import _helpers as H


def _attach_pred_classification(dataset, class_names):
    preds = [H.synth_classification(class_names, seed=i) for i in range(len(dataset))]
    dataset.set_values("pred_cls", preds)


def test_compute_hardness(brain_mod, dataset, class_names):
    _attach_pred_classification(dataset, class_names)
    brain_mod.compute_hardness(dataset, label_field="pred_cls")

    assert "hardness" in dataset.get_field_schema()
    lo, hi = dataset.bounds("hardness")
    assert lo >= 0.0 and hi > lo, "hardness ควรมีการกระจาย ไม่ใช่ค่าเดียวทั้งชุด"

    hardest = dataset.sort_by("hardness", reverse=True).first()
    print("ภาพยากสุด: hardness =", round(hardest.hardness, 4), "| pred =", hardest.pred_cls.label)


def test_ambiguous_samples_are_harder(brain_mod, dataset, class_names):
    """sample index คู่ (seed คู่) ถูกสร้างให้ ambiguous -> hardness เฉลี่ยต้องสูงกว่ากลุ่ม index คี่"""
    _attach_pred_classification(dataset, class_names)
    brain_mod.compute_hardness(dataset, label_field="pred_cls")

    vals = [s.hardness for s in dataset]
    even = np.mean([v for i, v in enumerate(vals) if i % 2 == 0])
    odd = np.mean([v for i, v in enumerate(vals) if i % 2 == 1])
    print(f"hardness เฉลี่ย  even(ambiguous)={even:.4f}  odd={odd:.4f}")
    assert even > odd
