FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the models at build time so the container starts ready to use
RUN python -m app.ml.train_classifier && python -m app.ml.train_anomaly && python -m app.ml.train_extractor

# Keep the SQLite database on a volume so data survives container restarts
ENV OPSPILOT_DATABASE_URL=sqlite:////data/opspilot.db
RUN mkdir -p /data
VOLUME /data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
