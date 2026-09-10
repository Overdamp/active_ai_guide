"""
เทส 09 — ลดมิติเพื่อ plot (Embeddings Visualization)
เครื่องมือ FiftyOne (built-in): fiftyone.brain.compute_visualization()
  - ลดมิติ embeddings -> 2D สำหรับ scatter plot ใน FiftyOne App (เห็นกลุ่ม/outlier)
  - method="pca"  : เร็ว ใช้ sklearn (ค่าเริ่มต้นของเทสนี้)
  - method="umap" : แยกกลุ่มดีกว่า แต่ต้องติดตั้ง umap-learn (มี marker slow)
ผลลัพธ์เก็บเป็น brain run, ดึงจุดผ่าน results.points (N x 2)
"""
import numpy as np
import pytest

import _helpers as H


def _attach_embeddings(dataset):
    paths = [s.filepath for s in dataset]
    dataset.set_values("embedding", [e.tolist() for e in H.embeddings_for(paths)])


def test_compute_visualization_pca(brain_mod, dataset):
    _attach_embeddings(dataset)
    results = brain_mod.compute_visualization(
        dataset, embeddings="embedding", method="pca", brain_key="viz_pca", num_dims=2
    )
    pts = np.asarray(results.points)
    assert pts.shape == (len(dataset), 2)
    assert np.isfinite(pts).all()
    assert "viz_pca" in dataset.list_brain_runs()

    reloaded = dataset.load_brain_results("viz_pca")
    assert np.allclose(np.asarray(reloaded.points), pts, atol=1e-5)
    print("PCA points bbox:", pts.min(axis=0).round(3), "->", pts.max(axis=0).round(3))


@pytest.mark.slow
def test_compute_visualization_umap(brain_mod, dataset):
    pytest.importorskip("umap", reason="ต้อง pip install umap-learn (ดู requirements.txt)")
    _attach_embeddings(dataset)
    n = len(dataset)
    results = brain_mod.compute_visualization(
        dataset, embeddings="embedding", method="umap", brain_key="viz_umap",
        num_dims=2, n_neighbors=min(5, n - 1), verbose=False,
    )
    assert np.asarray(results.points).shape == (n, 2)
