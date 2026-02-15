FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app/

# Create model directory
RUN mkdir -p python_Scripts

# Download YOLO weights from Google Drive
RUN curl -L -o python_Scripts/yolov3.weights \
    "https://drive.google.com/uc?export=download&id=1CueninL7gOQg-oHMIddHRE6VHd53PjT7"

EXPOSE 7860
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 180"]
