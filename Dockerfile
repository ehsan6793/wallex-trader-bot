FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot ./bot
ENV LOG_DIR=/app/logs DATA_DIR=/app/data PYTHONUNBUFFERED=1
CMD ["python", "-m", "bot", "live", "--yes"]
