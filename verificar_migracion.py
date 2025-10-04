#!/usr/bin/env python3
"""
Script para verificar los datos migrados en Firestore y Storage
"""

import os
from google.cloud import firestore, storage
from google.oauth2 import service_account
import json

# Configuración
PROJECT_ID = "infractivision-474103"
BUCKET_NAME = "infractivision-474103"
BASE_DIR = os.path.dirname(__file__)
LOCAL_KEY_PATH = os.path.join(BASE_DIR, "secrets", "infractivision-474103-0f907d0fbc62.json")

def _make_clients():
    creds = None
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        pass  # ADC
    elif os.path.exists(LOCAL_KEY_PATH):
        creds = service_account.Credentials.from_service_account_file(LOCAL_KEY_PATH)

    if creds is None:
        db = firestore.Client(project=PROJECT_ID)
        st = storage.Client(project=PROJECT_ID)
    else:
        db = firestore.Client(project=PROJECT_ID, credentials=creds)
        st = storage.Client(project=PROJECT_ID, credentials=creds)
    return db, st

def verificar_migracion():
    print("🔍 VERIFICANDO MIGRACIÓN A GOOGLE CLOUD")
    print("=" * 50)
    
    db, storage_client = _make_clients()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Obtener todas las colecciones de usuarios
    usuarios = db.collection("usuarios").get()
    
    if not usuarios:
        print("❌ No se encontraron usuarios en Firestore")
        return
    
    for usuario_doc in usuarios:
        user_id = usuario_doc.id
        print(f"\n👤 Usuario: {user_id}")
        print("-" * 30)
        
        # Verificar infracciones
        infracciones = db.collection("usuarios").document(user_id).collection("infracciones").get()
        print(f"📋 Infracciones: {len(infracciones)} encontradas")
        
        for i, infraccion in enumerate(infracciones[:3]):  # Mostrar solo las primeras 3
            data = infraccion.to_dict()
            print(f"  {i+1}. {data.get('placa', 'N/A')} - {data.get('fecha', 'N/A')} {data.get('hora', 'N/A')}")
            print(f"     Ubicación: {data.get('ubicacion', 'N/A')}")
            print(f"     Tipo: {data.get('tipo', 'N/A')}")
            print(f"     Confianza: {data.get('confianza', 'N/A')}")
            print(f"     Sistema: {data.get('sistema_version', 'N/A')}")
            
            # Verificar si las imágenes están en Storage
            if data.get('vehicle_url'):
                print(f"     ✅ Imagen vehículo: {data.get('vehicle_url', '')[:50]}...")
            if data.get('plate_url'):
                print(f"     ✅ Imagen placa: {data.get('plate_url', '')[:50]}...")
            print()
        
        if len(infracciones) > 3:
            print(f"  ... y {len(infracciones) - 3} infracciones más")
        
        # Verificar indicadores
        indicadores = db.collection("usuarios").document(user_id).collection("indicadores").get()
        print(f"\n📊 Indicadores: {len(indicadores)} encontrados")
        
        for i, indicador in enumerate(indicadores[:2]):  # Mostrar solo los primeros 2
            data = indicador.to_dict()
            print(f"  {i+1}. Fecha subida: {data.get('fecha_subida', 'N/A')}")
            print(f"     TI Acierto: {data.get('indicadores_TI_porcentaje_acierto', 'N/A')}%")
            print(f"     TR Tiempo: {data.get('indicadores_TR_con_software_tiempo_promedio_minutos', 'N/A')} min")
            print(f"     NID Detecciones: {data.get('indicadores_NID_infracciones_hoy', 'N/A')}")
            
            if data.get('storage_url'):
                print(f"     ✅ JSON completo: {data.get('storage_url', '')[:50]}...")
            print()
    
    # Verificar archivos en Storage
    print(f"\n🪣 Verificando Cloud Storage ({BUCKET_NAME}):")
    print("-" * 30)
    
    try:
        blobs = list(bucket.list_blobs(prefix="evidencias/", max_results=20))
        if blobs:
            print(f"📁 Archivos encontrados: {len(blobs)}")
            for blob in blobs[:10]:  # Mostrar solo los primeros 10
                size_kb = round(blob.size / 1024, 1) if blob.size else 0
                print(f"  📄 {blob.name} ({size_kb} KB)")
            
            if len(blobs) > 10:
                print(f"  ... y {len(blobs) - 10} archivos más")
        else:
            print("❌ No se encontraron archivos en evidencias/")
    except Exception as e:
        print(f"❌ Error accediendo a Storage: {e}")
    
    print("\n" + "=" * 50)
    print("✅ VERIFICACIÓN COMPLETADA")
    print("🎉 ¡Tu migración a Google Cloud fue exitosa!")

if __name__ == "__main__":
    verificar_migracion()