"""
เทส 03 — หาภาพซ้ำเป๊ะ (Exact Duplicates)
เครื่องมือ FiftyOne (built-in): fiftyone.brain.compute_exact_duplicates()
กลไก: hash เนื้อไฟล์ (md5) แล้วจับกลุ่มไฟล์ที่ byte เหมือนกันทุกประการ
พิสูจน์ว่า: เมื่อมีไฟล์คัดลอกแบบ byte-for-byte อยู่ในชุด FiftyOne จับคู่ได้ถูกต้อง
"""
from pathlib import Path

import _helpers as H


def test_compute_exact_duplicates(fo_mod, brain_mod, dataset, tmp_path):
    # คัดลอกภาพแรกแบบเป๊ะ ๆ ไปเป็นไฟล์ใหม่ แล้วเพิ่มเข้า dataset
    first_path = Path(dataset.first().filepath)
    copy_path = H.duplicate_file(first_path, tmp_path)
    assert H.md5(first_path) == H.md5(copy_path)

    dataset.add_sample(fo_mod.Sample(filepath=str(copy_path)))
    n_before = len(dataset)

    dups = brain_mod.compute_exact_duplicates(dataset)

    # dups = { sample_id_ที่เก็บไว้ : [sample_id_ของตัวซ้ำ, ...] }
    assert len(dups) >= 1, "ควรเจอกลุ่มไฟล์ซ้ำอย่างน้อย 1 กลุ่ม"
    involved = set(dups) | {sid for ids in dups.values() for sid in ids}

    ids_by_path = {s.filepath: s.id for s in dataset}
    assert ids_by_path[str(first_path)] in involved
    assert ids_by_path[str(copy_path)] in involved
    assert len(dataset) == n_before  # compute_* ไม่ลบอะไรเอง — แค่รายงาน
    print("กลุ่มซ้ำที่พบ:", {k[:8]: v for k, v in dups.items()})


def test_removing_one_of_each_duplicate_group(fo_mod, brain_mod, dataset, tmp_path):
    """โชว์ 'business logic' ที่ต้องเขียนเอง: เก็บภาพแรก ลบที่เหลือของแต่ละกลุ่ม"""
    first_path = Path(dataset.first().filepath)
    copy_path = H.duplicate_file(first_path, tmp_path)
    dataset.add_sample(fo_mod.Sample(filepath=str(copy_path)))

    dups = brain_mod.compute_exact_duplicates(dataset)
    # dict = {ตัวที่เก็บ: [ตัวซ้ำ...]}  -> ลบเฉพาะ value (ตัวซ้ำ) เก็บ key ไว้
    to_delete = [sid for ids in dups.values() for sid in ids]

    if to_delete:
        dataset.delete_samples(to_delete)
    assert len(brain_mod.compute_exact_duplicates(dataset)) == 0
