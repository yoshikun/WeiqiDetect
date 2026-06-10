FROM python:3.11-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py detector.py image_decode.py moku_detect.py baduk_detect.py kaya_corners.py kaya_stones.py board_geometry.py ./

RUN mkdir -p models/moku \
  && curl -fsSL -o models/moku/model.onnx \
    "https://huggingface.co/kaya-go/moku-v3/resolve/main/model.onnx"

ENV PORT=80
EXPOSE 80

CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app", "--workers", "1", "--threads", "4", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-"]
