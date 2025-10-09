# Use a slim Python base
FROM python:3.11-slim

# metadata
LABEL maintainer="you <you@example.com>" \
      description="Receipt thesis backend (FastAPI + Tesseract + OCR fallback)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system deps (tesseract, libs for opencv, build tools for some wheels)
# Keep apt-get steps in one RUN to reduce layers. Using --no-install-recommends to shrink size.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      ca-certificates \
      pkg-config \
      libjpeg-dev \
      libpng-dev \
      libtiff5-dev \
      libwebp-dev \
      libsm6 \
      libxext6 \
      libxrender1 \
      libgl1 \
      tesseract-ocr \
      libtesseract-dev \
      libleptonica-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache-able)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# Copy app code
COPY . /app

# Expose the port
EXPOSE ${PORT}

# Create runtime directories (optional)
RUN mkdir -p /app/data /app/models /app/runs && \
    chown -R root:root /app

# Start command — use environment PORT (default 8000)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
