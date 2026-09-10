"""
เทส 02 — Custom fields + View expression (สะท้อน lab_fiftyone_curation.py โดยตรง)
เครื่องมือ FiftyOne:
  dataset.set_values(), dataset.add_sample_field(),
  fiftyone.ViewField (F), dataset.match(), dataset.bounds()
Custom logic (ไม่มีใน FiftyOne — ต้องเขียนเอง):
  - Blur score = Variance of Laplacian (OpenCV)
  - Glare ratio = สัดส่วนพิกเซล V>240 & S<40 ใน HSV
พิสูจน์ว่า: เราคำนวณ field เองด้วย OpenCV แล้วโยนกลับเข้า FiftyOne schema
           จากนั้นใช้ View expression กรอง Hard/Clean sample ได้
"""
import cv2
import numpy as np
from fiftyone import ViewField as F

BLUR_THRESHOLD = 120.0
GLARE_RATIO_THRESHOLD = 0.15


def _quality(path: str) -> tuple[float, float]:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    glare = float(((hsv[:, :, 2] >= 240) & (hsv[:, :, 1] <= 40)).mean())
    return blur, glare


def test_enrich_and_filter(dataset):
    blur_scores, glare_ratios = [], []
    for s in dataset.select_fields("filepath"):
        b, g = _quality(s.filepath)
        blur_scores.append(b)
        glare_ratios.append(g)

    dataset.set_values("blur_score", blur_scores)
    dataset.set_values("glare_ratio", glare_ratios)

    # field ถูกบันทึกลง schema จริง
    schema = dataset.get_field_schema()
    assert "blur_score" in schema and "glare_ratio" in schema

    # View expression: ภาพคุณภาพดี = คมชัดพอ + ไม่มีแสงสะท้อนจ้า
    clean = dataset.match((F("blur_score") > BLUR_THRESHOLD) & (F("glare_ratio") < GLARE_RATIO_THRESHOLD))
    hard = dataset.match((F("blur_score") <= BLUR_THRESHOLD) | (F("glare_ratio") >= GLARE_RATIO_THRESHOLD))

    assert len(clean) + len(hard) == len(dataset)
    lo, hi = dataset.bounds("blur_score")
    print(f"blur range = {lo:.1f}..{hi:.1f} | clean={len(clean)} hard={len(hard)} | kind={dataset.info['kind']}")


def test_synthetic_blur_and_glare_are_detected(dataset):
    """เฉพาะกรณีภาพสังเคราะห์: index 5 ต้องเบลอ, index 6 ต้องมี glare สูง (ผลกำหนดได้)"""
    if dataset.info.get("kind") != "synthetic":
        return  # ภาพจริงไม่การันตีว่ามีเคสเบลอ/แสงจ้า — ข้ามการเช็คนี้

    paths = [s.filepath for s in dataset]
    blurs = np.array([_quality(p)[0] for p in paths])
    glares = np.array([_quality(p)[1] for p in paths])

    assert blurs.argmin() == 5, f"ภาพเบลอควรเป็น index 5 (ได้ {blurs.argmin()})"
    assert glares.argmax() == 6, f"ภาพ glare ควรเป็น index 6 (ได้ {glares.argmax()})"
    assert glares[6] > GLARE_RATIO_THRESHOLD
