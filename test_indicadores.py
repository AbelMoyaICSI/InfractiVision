"""
Script de prueba para verificar que los indicadores se leen correctamente
"""
import json
import os

# Rutas
INDICADORES_PATH = os.path.join("data", "indicadores_rendimiento.json")
INFRACCIONES_PATH = os.path.join("data", "infracciones.json")
NIE_PATH = os.path.join("data", "nie_infracciones.json")

print("=" * 60)
print("TEST: Verificación de archivos JSON")
print("=" * 60)

# 1. Verificar archivo de indicadores
print("\n1. INDICADORES:")
if os.path.exists(INDICADORES_PATH):
    with open(INDICADORES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"   ✅ Archivo existe: {INDICADORES_PATH}")
    print(f"   📊 Ubicación: {data.get('ubicacion', 'N/A')}")
    print(f"   📅 Fecha generación: {data.get('fecha_generacion', 'N/A')}")
    print(f"   📆 Días analizados: {data.get('dias_analizados', 0)}")
    
    indicadores = data.get("indicadores", {})
    nid = indicadores.get("NID", {})
    nie = indicadores.get("NIE", {})
    ti = indicadores.get("TI", {})
    tr = indicadores.get("TR", {})
    
    print(f"\n   NID:")
    print(f"      - Total: {nid.get('total', nid.get('infracciones_hoy', 0))}")
    print(f"      - Hoy: {nid.get('infracciones_hoy', 0)}")
    print(f"      - Promedio diario: {nid.get('promedio_diario', 0)}")
    
    print(f"\n   NIE:")
    print(f"      - Total: {nie.get('total', nie.get('infracciones_incorrectas', 0))}")
    print(f"      - Incorrectas: {nie.get('infracciones_incorrectas', 0)}")
    
    print(f"\n   TI:")
    print(f"      - Porcentaje acierto: {ti.get('porcentaje_acierto', 0)}%")
    
    print(f"\n   TR:")
    tr_con_sw = tr.get("con_software", {})
    print(f"      - Tiempo promedio (min): {tr_con_sw.get('tiempo_promedio_minutos', 0)}")
    print(f"      - Muestras analizadas: {tr_con_sw.get('muestras_analizadas', 0)}")
    
    resumen = data.get("resumen_global", {})
    print(f"\n   RESUMEN GLOBAL:")
    print(f"      - NID total: {resumen.get('nid_total', 0)}")
    print(f"      - NIE total: {resumen.get('nie_total', 0)}")
    print(f"      - TIR total: {resumen.get('tir_total', 0)}")
else:
    print(f"   ❌ No existe: {INDICADORES_PATH}")

# 2. Verificar archivo de infracciones
print("\n2. INFRACCIONES (NID):")
if os.path.exists(INFRACCIONES_PATH):
    with open(INFRACCIONES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    infracciones = data.get("infracciones", []) if isinstance(data, dict) else data
    print(f"   ✅ Archivo existe: {INFRACCIONES_PATH}")
    print(f"   📝 Total infracciones NID: {len(infracciones)}")
    if infracciones:
        print(f"   📌 Primera infracción:")
        primera = infracciones[0]
        print(f"      - Placa: {primera.get('placa', 'N/A')}")
        print(f"      - Fecha: {primera.get('fecha', 'N/A')}")
        print(f"      - Clasificación: {primera.get('clasificacion', 'N/A')}")
        print(f"      - Confianza: {primera.get('confianza', 0)}")
else:
    print(f"   ❌ No existe: {INFRACCIONES_PATH}")

# 3. Verificar archivo de NIE
print("\n3. INFRACCIONES NIE:")
if os.path.exists(NIE_PATH):
    with open(NIE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    nie_infracciones = data.get("infracciones", []) if isinstance(data, dict) else data
    print(f"   ✅ Archivo existe: {NIE_PATH}")
    print(f"   📝 Total infracciones NIE: {len(nie_infracciones)}")
    if nie_infracciones:
        print(f"   📌 Primera NIE:")
        primera = nie_infracciones[0]
        print(f"      - Placa: {primera.get('placa', 'N/A')}")
        print(f"      - Fecha: {primera.get('fecha', 'N/A')}")
        print(f"      - Clasificación: {primera.get('clasificacion', 'N/A')}")
        print(f"      - Confianza: {primera.get('confianza', 0)}")
else:
    print(f"   ❌ No existe: {NIE_PATH}")

print("\n" + "=" * 60)
print("TEST COMPLETADO")
print("=" * 60)
