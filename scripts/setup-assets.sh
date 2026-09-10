#!/usr/bin/env bash
# ดึงชุดข้อมูล + โมเดล เข้ามาไว้ในตัว repo (ครั้งเดียวหลัง clone)
# ใช้ hardlink (cp -al) — ถ้าอยู่ filesystem เดียวกันจะไม่กินดิสก์เพิ่ม
#
#   datasets/  และ  models/  ถูก gitignore ไว้ (ใหญ่เกินเก็บใน git)
#   สคริปต์นี้เติมให้ครบเพื่อให้ทุก path ในโปรเจกต์ชี้เข้า PTT_ai_mini ล้วน ๆ
#
# แก้ SRC ได้ถ้า repo ต้นทางอยู่ที่อื่น
set -euo pipefail
cd "$(dirname "$0")/.."          # -> repo root

SRC="${PTT_ASSET_SRC:-/home/luke/ai_training/PTT_smart_ai_platform}"
DS_NAME="overall-ptt-object-detection.v11i.yolov11"
MODEL_NAME="PTT_YOLO12n_v11i_Baseline_v1.0.0_best.pt"

echo "ต้นทาง: $SRC"
[ -d "$SRC/datasets/$DS_NAME" ] || { echo "ไม่พบ $SRC/datasets/$DS_NAME"; exit 1; }
[ -f "$SRC/models/$MODEL_NAME" ] || { echo "ไม่พบ $SRC/models/$MODEL_NAME"; exit 1; }

# 1) base dataset ที่ active_learning_split/*/images ชี้หา (symlink สัมพัทธ์)
mkdir -p datasets
if [ ! -e "datasets/$DS_NAME" ]; then
  cp -al "$SRC/datasets/$DS_NAME" "datasets/$DS_NAME" 2>/dev/null \
    || cp -a  "$SRC/datasets/$DS_NAME" "datasets/$DS_NAME"        # ข้าม filesystem: copy จริง
  echo "✓ datasets/$DS_NAME"
else
  echo "• datasets/$DS_NAME มีอยู่แล้ว ข้าม"
fi

# 2) โมเดล YOLO
mkdir -p models
if [ ! -e "models/$MODEL_NAME" ]; then
  cp -al "$SRC/models/$MODEL_NAME" "models/$MODEL_NAME" 2>/dev/null \
    || cp -a  "$SRC/models/$MODEL_NAME" "models/$MODEL_NAME"
  echo "✓ models/$MODEL_NAME"
else
  echo "• models/$MODEL_NAME มีอยู่แล้ว ข้าม"
fi

# 3) ตรวจว่า symlink ของ active_learning_split resolve ได้ และไม่หลุดออกนอก repo
ok=$(find datasets/active_learning_split/seed_dataset/valid/images -type l -xtype f 2>/dev/null | wc -l)
esc=$(find datasets/ -type l 2>/dev/null | while read l; do
        case "$(readlink -f "$l" 2>/dev/null)" in
          "$PWD"/*) ;; *) echo x ;;
        esac; done | wc -l)
echo
echo "seed_dataset/valid resolve ได้: $ok / 773"
echo "symlink หลุดออกนอก repo       : $esc  (ควรเป็น 0)"
[ "$ok" -eq 773 ] && [ "$esc" -eq 0 ] && echo "✅ พร้อมใช้งาน — ทุก path อยู่ใน PTT_ai_mini" || { echo "⚠️ ยังไม่ครบ"; exit 1; }
