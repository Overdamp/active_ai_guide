#!/usr/bin/env bash
# เปิด FiftyOne App (เว็บ) ชี้ไปที่ dataset ที่ต้องการ
# ใช้: docker/scripts/launch-app.sh [ชื่อ dataset]
#   - ถ้าไม่ระบุชื่อ จะ list dataset ที่มีให้เลือก
# เปิดเบราว์เซอร์ที่ http://localhost:5151  (หรือพอร์ตตาม FIFTYONE_APP_PORT ใน .env)
set -euo pipefail
cd "$(dirname "$0")/.."          # -> docker/

docker compose up -d

if [ $# -eq 0 ]; then
  echo "dataset ที่มีอยู่:"
  docker compose exec fiftyone python -c "import fiftyone as fo; print('\n'.join(fo.list_datasets()) or '(ยังไม่มี — รัน lab หรือเทสก่อน)')"
  echo
  echo "ใช้: $0 <ชื่อ dataset>"
  exit 0
fi

echo "เปิด FiftyOne App สำหรับ dataset '$1' ที่ http://localhost:${FIFTYONE_APP_PORT:-5151} (Ctrl+C เพื่อหยุด)"
exec docker compose exec fiftyone fiftyone app launch "$1" --address 0.0.0.0 --port 5151
