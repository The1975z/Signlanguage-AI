FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ src/
COPY tests/ tests/
COPY models/hand_landmarker.task models/hand_landmarker.task

RUN mkdir -p data/processed data/raw results models

CMD ["python", "-I", "src/extract.py"]
