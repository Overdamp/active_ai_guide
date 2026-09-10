#!/usr/bin/env bash
# รันชุดเทส FiftyOne ทั้งหมดใน container
# ใช้: docker/scripts/run-tests.sh [อาร์กิวเมนต์ pytest เพิ่มเติม]
# ตัวอย่าง:
#   docker/scripts/run-tests.sh                     # รันทั้งหมด
#   docker/scripts/run-tests.sh -m "not slow"       # ข้ามเทสช้า
#   docker/scripts/run-tests.sh tests/test_04_similarity_near_duplicates.py
set -euo pipefail
cd "$(dirname "$0")/.."          # -> docker/

docker compose up -d
exec docker compose exec -w /workspace/docker fiftyone pytest "$@"
