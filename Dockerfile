# AI Gaze Streamlit studio — Railway
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    AIGAZE_DATA_DIR=/app/.data \
    AIGAZE_PUBLIC_URL=https://aigaze-production.up.railway.app

WORKDIR /app

# Headless OpenCV / MediaPipe runtime libs (no full X11 stack needed if GUI opencv is removed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install --force-reinstall --no-cache-dir "opencv-python-headless>=4.9.0,<5"

COPY . .

# Replace Streamlit boot splash title/logo with AI Gaze branding
RUN python scripts/patch_streamlit_branding.py

EXPOSE 8080

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
