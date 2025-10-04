#!/bin/bash
# Script para desplegar InfractiVision backend en Cloud Run

# Variables
PROJECT_ID="infractivision-474103"
SERVICE_NAME="infractivision-backend"
REGION="us-central1"

echo "🚀 Desplegando InfractiVision Backend en Cloud Run..."
echo "📁 Proyecto: $PROJECT_ID"
echo "🌍 Región: $REGION"
echo "⚙️  Servicio: $SERVICE_NAME"

# Construir y desplegar
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10

echo "✅ Despliegue completado!"
echo "🔗 Tu backend estará disponible en:"
echo "   https://$SERVICE_NAME-[hash].$REGION.run.app"