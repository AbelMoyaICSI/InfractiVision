"""
Script para regenerar indicadores_rendimiento.json con los nuevos cálculos corregidos
"""
import json
import os

# Importar la función corregida
from src.gui.infractions_management_window import generate_performance_indicators_json

def regenerar():
    # Leer infracciones guardadas
    infractions_file = "data/infracciones.json"
    nie_file = "data/nie_infracciones.json"
    
    infracciones = []
    if os.path.exists(infractions_file):
        with open(infractions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'infracciones' in data:
                infracciones = data['infracciones']
            elif isinstance(data, list):
                infracciones = data
    
    nie_infracciones = []
    if os.path.exists(nie_file):
        with open(nie_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'infracciones' in data:
                nie_infracciones = data['infracciones']
            elif isinstance(data, list):
                nie_infracciones = data
    
    # Combinar todas las infracciones
    todas = infracciones + nie_infracciones
    
    print(f"📋 Infracciones cargadas: {len(infracciones)} NID + {len(nie_infracciones)} NIE = {len(todas)} total")
    
    # Extraer tiempos de procesamiento individuales
    tiempos_individuales = [
        inf.get('tiempo_procesamiento', 0) 
        for inf in todas 
        if inf.get('tiempo_procesamiento', 0) > 0
    ]
    
    print(f"⏱️ Tiempos individuales extraídos: {tiempos_individuales} segundos")
    
    # Regenerar JSON con función corregida
    print("\n🔄 Regenerando indicadores con cálculos corregidos...")
    generate_performance_indicators_json(todas, tiempos_individuales)
    
    print("\n✅ Indicadores regenerados correctamente!")
    print("📄 Verifica: data/indicadores_rendimiento.json")

if __name__ == "__main__":
    regenerar()
