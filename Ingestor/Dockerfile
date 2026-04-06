FROM python:3.10-slim

WORKDIR /app

# ===== เพิ่ม tools ไว้ debug =====
RUN apt-get update && apt-get install -y \
    curl \
    dnsutils \
    procps \
    netcat-openbsd \
 && rm -rf /var/lib/apt/lists/*

# ===== install python deps =====
RUN pip install --no-cache-dir requests psycopg2-binary

COPY ingestor.py /app/ingestor.py

# สำคัญมาก → log realtime
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "ingestor.py"]
