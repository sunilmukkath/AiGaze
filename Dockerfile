# AI Gaze API + studio — Railway (FastAPI, no Streamlit)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    AIGAZE_DATA_DIR=/app/.data \
    AIGAZE_PUBLIC_URL=https://aigaze-production.up.railway.app \
    AIGAZE_HTTPS_ONLY=1

WORKDIR /app

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

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api_app:app --host 0.0.0.0 --port ${PORT:-8080}"]
