# src/automations/Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

# Sólo necesitas las libs de GCP
RUN pip install --no-cache-dir \
      google-cloud-storage \
      google-cloud-firestore

# Al no exponer un puerto, CMD corre el migrator y finaliza
CMD ["python", "cloud_migrator.py"]
