# Script PowerShell para desplegar en Cloud Run desde Windows
param(
    [string]$ProjectId = "infractivision-474103",
    [string]$ServiceName = "infractivision-backend", 
    [string]$Region = "us-central1"
)

Write-Host "🚀 Desplegando InfractiVision Backend en Cloud Run..." -ForegroundColor Green
Write-Host "📁 Proyecto: $ProjectId" -ForegroundColor Cyan
Write-Host "🌍 Región: $Region" -ForegroundColor Cyan  
Write-Host "⚙️  Servicio: $ServiceName" -ForegroundColor Cyan

# Cambiar al directorio del backend
Set-Location -Path $PSScriptRoot

# Configurar proyecto de gcloud
gcloud config set project $ProjectId

# Construir y desplegar
gcloud run deploy $ServiceName `
    --source . `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --memory 512Mi `
    --cpu 1 `
    --timeout 300 `
    --max-instances 10

Write-Host "✅ Despliegue completado!" -ForegroundColor Green
Write-Host "🔗 Tu backend estará disponible en:" -ForegroundColor Yellow
Write-Host "   https://$ServiceName-[hash].$Region.run.app" -ForegroundColor Yellow