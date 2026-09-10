# =============================================================================
# Image สำหรับรัน FiftyOne + OpenCV + ชุดเทสของ PTT_ai_mini
# ฐาน: python:3.12-slim (ตรงกับ conda env `ai_training` ในโปรเจกต์จริง)
# =============================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # ค่าเริ่มต้น เผื่อรัน image นี้เดี่ยว ๆ นอก compose (compose จะ override อีกที)
    FIFTYONE_DATABASE_URI=mongodb://mongo:27017 \
    FIFTYONE_DATABASE_NAME=ptt_fiftyone

# ---- system libraries -------------------------------------------------------
# libgl1 / libglib2.0-0 : ให้ OpenCV โหลดได้ (แม้ใช้ build headless ก็ยังต้องมี libglib)
# curl                   : ใช้เช็ก health / ดีบั๊ก
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- python dependencies --------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt

# ตรวจว่า import ได้จริงตั้งแต่ตอน build (fail fast)
RUN python -c "import fiftyone as fo, cv2, sklearn, numpy; print('OK fiftyone', fo.__version__, '| cv2', cv2.__version__)"

EXPOSE 5151

# ค่าเริ่มต้น: แค่พิมพ์เวอร์ชัน แล้วจบ (compose จะสั่ง `sleep infinity` แทน)
CMD ["python", "-c", "import fiftyone as fo; print('fiftyone', fo.__version__)"]
