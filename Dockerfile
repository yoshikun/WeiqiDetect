FROM python:3.11-slim AS trainer

RUN apt-get update \
  && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /train
COPY training/requirements-train.txt ./requirements-train.txt
RUN pip install --no-cache-dir -r requirements-train.txt

COPY training/ ./training/
RUN python training/train.py \
  --output /models/stone_cls.onnx \
  --meta /models/stone_cls.meta.json \
  --epochs 12 \
  --samples-per-class 3500

FROM python:3.11-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py board_geometry.py cnn_classifier.py detector.py ./
COPY --from=trainer /models/ ./models/

ENV PORT=80
EXPOSE 80

CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app", "--workers", "1", "--threads", "4", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-"]
