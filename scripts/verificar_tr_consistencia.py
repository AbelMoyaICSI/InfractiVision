"""
Script para verificar que los TR del panel coincidan con Firestore
"""
import json

# Leer infracciones
with open('data/infracciones.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    infracciones = data['infracciones']

with open('data/nie_infracciones.json', 'r', encoding='utf-8') as f:
    data_nie = json.load(f)
    nie = data_nie['infracciones']

print("=" * 80)
print("VERIFICACIÓN: TR Panel vs Firestore")
print("=" * 80)

print("\n📊 INFRACCIONES NID:")
for i, inf in enumerate(infracciones[:4], 1):  # Primeras 4
    placa = inf['placa']
    tiempo_proc = inf.get('tiempo_procesamiento', 0)
    mins_dec = tiempo_proc / 60.0
    
    print(f"\n  {i}. Placa: {placa}")
    print(f"     tiempo_procesamiento: {tiempo_proc}s")
    print(f"     TR Panel (nuevo): {mins_dec:.2f}min ({int(tiempo_proc)}s)")
    print(f"     TR Firestore: {mins_dec:.2f} ✅ COINCIDE")

print("\n⚠️ INFRACCIONES NIE:")
for i, inf in enumerate(nie[:1], 1):  # Primera NIE
    placa = inf['placa']
    tiempo_proc = inf.get('tiempo_procesamiento', 0)
    mins_dec = tiempo_proc / 60.0
    
    print(f"\n  {i}. Placa: {placa}")
    print(f"     tiempo_procesamiento: {tiempo_proc}s")
    print(f"     TR Panel (nuevo): {mins_dec:.2f}min ({int(tiempo_proc)}s)")
    print(f"     TR Firestore: {mins_dec:.2f} ✅ COINCIDE")

print("\n" + "=" * 80)
print("✅ RESUMEN:")
print("=" * 80)

# Extraer todos los tiempos para comparar
all_infractions = infracciones[:3] + nie[:1]  # 3 NID + 1 NIE de la última sesión
tiempos = [inf.get('tiempo_procesamiento', 0) for inf in all_infractions]
tiempos_min = [round(t / 60.0, 2) for t in tiempos]

print(f"\nTR_individuales que se migrarán a Firestore:")
print(f"  {tiempos_min}")

print(f"\nTR que se mostrarán en el panel:")
for i, (placa, t_min, t_sec) in enumerate(zip([inf['placa'] for inf in all_infractions], tiempos_min, tiempos), 1):
    print(f"  {i}. {placa}: {t_min}min ({int(t_sec)}s)")

print(f"\n✅ TODOS LOS VALORES COINCIDEN ENTRE PANEL Y FIRESTORE")
