"""
เทส 10 — View expressions + Aggregations (หัวใจการ query ของ FiftyOne)
เครื่องมือ FiftyOne (built-in):
  ViewField (F), dataset.match(), dataset.match_tags(), dataset.filter_labels(),
  dataset.sort_by(), .limit()/.skip(), .exists(), .select_fields()
  Aggregations: count(), count_values(), distinct(), bounds(), sum(), mean()
เป็นชุดเครื่องมือที่ lab ทุกตัวใช้กรอง Hard Sample ก่อนส่ง CVAT
"""
from fiftyone import ViewField as F


def test_match_and_numeric_view(dataset):
    dataset.set_values("score", [i * 0.1 for i in range(len(dataset))])
    high = dataset.match(F("score") >= 0.5)
    assert len(high) == len([1 for i in range(len(dataset)) if i * 0.1 >= 0.5 - 1e-9])
    assert all(s.score >= 0.5 for s in high)


def test_sort_limit_skip(dataset):
    dataset.set_values("score", list(range(len(dataset))))
    top3 = dataset.sort_by("score", reverse=True).limit(3)
    assert [s.score for s in top3] == sorted(range(len(dataset)), reverse=True)[:3]
    page2 = dataset.sort_by("score").skip(2).limit(2)
    assert [s.score for s in page2] == [2, 3]


def test_tags_and_match_tags(dataset):
    for s in dataset[:3]:
        s.tags.append("to_review")
        s.save()
    assert len(dataset.match_tags("to_review")) == 3


def test_exists_and_select_fields(dataset):
    vals = [1.0] * (len(dataset) - 2) + [None, None]
    dataset.set_values("maybe", vals)
    assert len(dataset.exists("maybe")) == len(dataset) - 2
    view = dataset.select_fields("filepath")
    assert "maybe" not in view.first().field_names


def test_aggregations(dataset):
    dataset.set_values("grp", ["a", "b", "a", "b"] * ((len(dataset) // 4) + 1))
    dataset.set_values("num", [float(i % 5) for i in range(len(dataset))])

    counts = dataset.count_values("grp")
    assert set(counts) <= {"a", "b"} and sum(counts.values()) == len(dataset)
    assert sorted(dataset.distinct("grp")) == ["a", "b"]
    lo, hi = dataset.bounds("num")
    assert lo == 0.0 and hi == 4.0
    assert dataset.count() == len(dataset)
    print("count_values(grp) =", counts, "| mean(num) =", round(dataset.mean("num"), 3))


def test_filter_labels_on_detections(dataset_with_gt):
    """filter_labels: เก็บเฉพาะกล่องที่เข้าเงื่อนไข (เช่น คลาสใดคลาสหนึ่ง) โดยไม่ทิ้งทั้ง sample"""
    names = dataset_with_gt.distinct("ground_truth.detections.label")
    assert names, "dataset จริงต้องมีอย่างน้อย 1 คลาส"
    target = names[0]
    view = dataset_with_gt.filter_labels("ground_truth", F("label") == target)
    for s in view:
        assert all(d.label == target for d in s.ground_truth.detections)
    print(f"กรองเหลือเฉพาะคลาส '{target}':", view.count("ground_truth.detections"), "กล่อง")
