#!/usr/bin/env bash
# รันสคริปต์ lab ของ fiftyone_module ใน container
# ใช้: docker/scripts/run-lab.sh [ชื่อไฟล์ lab]   (ค่าเริ่มต้น = lab_fiftyone_curation.py)
set -euo pipefail
cd "$(dirname "$0")/.."          # -> docker/

LAB="${1:-lab_fiftyone_curation.py}"

docker compose up -d
exec docker compose exec -w /workspace fiftyone python "fiftyone_module/${LAB}"
