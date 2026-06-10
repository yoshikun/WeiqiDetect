FROM python:3.11-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=80
CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app", "--workers", "1", "--threads", "2", "--timeout", "120"]
