FROM python:3.10-slim

WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer puerto que usa Cloud Run
EXPOSE 8080

# Usar gunicorn para producción
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 infracti_backend:app
