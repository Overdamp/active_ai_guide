"""
เทส 08 — ประเมินผล Object Detection (Evaluate Detections)   [needs_dataset]
เครื่องมือ FiftyOne (built-in): dataset.evaluate_detections()
  - เทียบ predictions กับ ground_truth ด้วย COCO-style matching (IoU)
  - คืน DetectionResults: .mAP(), .report(), .print_report()
  - เติม field ต่อ sample: eval_tp / eval_fp / eval_fn
  - เติม field ต่อกล่อง: <eval_key> = "tp"/"fp"/"fn"  -> ใช้ View กรองหา False Negative ได้
นี่คือฟังก์ชันที่มาแทน logic คำนวณ IoU เองใน lab_c / lab_d
"""
import pytest

pytestmark = pytest.mark.needs_dataset


def test_evaluate_detections_coco(dataset_with_preds):
    ds = dataset_with_preds
    results = ds.evaluate_detections(
        "predictions", gt_field="ground_truth", eval_key="eval",
        method="coco", compute_mAP=True,     # ต้องระบุ เพื่อให้ results มีเมธอด .mAP()
    )

    # per-sample counts
    for f in ("eval_tp", "eval_fp", "eval_fn"):
        assert f in ds.get_field_schema()
    assert ds.sum("eval_tp") > 0, "ควรมี true positive อย่างน้อยบ้าง (pred สร้างจาก GT)"
    assert ds.sum("eval_fn") > 0, "เราตั้งใจตัดกล่อง GT 20% ทิ้ง -> ต้องมี FN"

    mAP = results.mAP()
    assert 0.0 <= mAP <= 1.0
    print(f"mAP = {mAP:.3f} | TP={ds.sum('eval_tp')} FP={ds.sum('eval_fp')} FN={ds.sum('eval_fn')}")

    report = results.report()          # dict per-class: precision/recall/f1-score/support
    assert isinstance(report, dict) and report


def test_filter_false_negatives_via_view(dataset_with_preds):
    """ใช้ผลลัพธ์ eval กรองเฉพาะกล่องที่โมเดล 'มองข้าม' (fn) — เตรียมส่ง CVAT"""
    from fiftyone import ViewField as F

    ds = dataset_with_preds
    ds.evaluate_detections("predictions", gt_field="ground_truth", eval_key="ev", method="coco")

    fn_view = ds.filter_labels("ground_truth", F("ev") == "fn")
    n_fn_boxes = fn_view.count("ground_truth.detections")
    n_hard_images = len(fn_view.match(F("ground_truth.detections").length() > 0))
    assert n_fn_boxes >= 1
    print(f"กล่อง FN ทั้งหมด = {n_fn_boxes} ใน {n_hard_images} ภาพเคสยาก")
