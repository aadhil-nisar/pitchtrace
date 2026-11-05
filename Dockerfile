FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data && chmod 777 /app/data
ENV PORT=7860 DATA_DIR=/app/data MPLBACKEND=Agg
EXPOSE 7860
# one process (jobs live in memory) with threads for concurrent requests
CMD gunicorn -w 1 --threads 8 --timeout 0 -b 0.0.0.0:${PORT} app:app
