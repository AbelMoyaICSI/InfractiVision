FROM python:3.10-slim

WORKDIR /app
COPY . .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir flask google-cloud-storage google-cloud-firestore

EXPOSE 8080
CMD ["python", "infracti_backend.py"]
