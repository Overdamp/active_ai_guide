"""
เทส 04 — ความคล้าย + ภาพเกือบซ้ำ (Similarity / Near-Duplicates)
เครื่องมือ FiftyOne (built-in): fiftyone.brain.compute_similarity()
  - รับ embeddings (array หรือชื่อ field) -> สร้าง similarity index (backend เริ่มต้น = sklearn)
  - dataset.sort_by_similarity(query_id, k, brain_key)  : ค้นภาพใกล้เคียง
  - index.find_duplicates(thresh) + index.duplicates_view() : จับภาพเกือบซ้ำ
Custom ที่ต้องเตรียมเอง: embeddings (ที่นี่ใช้ cheap_embedding แทนโมเดล เพื่อรัน offline)
"""
from pathlib import Path

import numpy as np

import _helpers as H


def _attach_embeddings(dataset):
    paths = [s.filepath for s in dataset]
    embs = H.embeddings_for(paths)
    dataset.set_values("embedding", [e.tolist() for e in embs])
    return embs


def test_compute_similarity_and_query(brain_mod, dataset):
    _attach_embeddings(dataset)
    index = brain_mod.compute_similarity(dataset, embeddings="embedding", brain_key="sim")
    assert "sim" in dataset.list_brain_runs()

    q = dataset.first().id
    view = dataset.sort_by_similarity(q, k=3, brain_key="sim")
    assert len(view) == 3
    assert view.first().id == q  # ภาพตัวเองต้องใกล้ตัวเองที่สุด
    print("ค้นภาพใกล้เคียง 3 อันดับแรก:", [Path(s.filepath).name for s in view])


def test_find_near_duplicates(fo_mod, brain_mod, dataset, tmp_path):
    """ใส่ไฟล์คัดลอกเข้าไป -> embedding เท่ากันเป๊ะ -> ต้องถูกจับเป็น near-duplicate"""
    src = Path(dataset.first().filepath)
    copy_path = H.duplicate_file(src, tmp_path)
    dataset.add_sample(fo_mod.Sample(filepath=str(copy_path)))

    _attach_embeddings(dataset)
    index = brain_mod.compute_similarity(dataset, embeddings="embedding", brain_key="sim2")
    index.find_duplicates(thresh=0.02)   # cosine distance ~0 สำหรับภาพเหมือนกัน

    assert len(index.duplicate_ids) >= 1, "ควรจับภาพเกือบซ้ำได้อย่างน้อย 1"
    dup_view = index.duplicates_view()
    dup_paths = {Path(s.filepath).name for s in dup_view}
    assert src.name in dup_paths or (src.stem + "__COPY.jpg") in {p for p in dup_paths}
    print("neighbors_map:", {k[:6]: [x[0][:6] for x in v] for k, v in index.neighbors_map.items()})


def test_similarity_matrix_is_symmetric_ish(dataset):
    """เช็คคุณสมบัติ cosine similarity ของ embeddings เอง (sanity ก่อนโยนเข้า brain)"""
    embs = _attach_embeddings(dataset)
    sim = embs @ embs.T
    assert np.allclose(np.diag(sim), 1.0, atol=1e-4)
    assert np.allclose(sim, sim.T, atol=1e-5)
