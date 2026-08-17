import os
import json
import uuid
import getpass
import socket
from datetime import datetime
from google.cloud import storage, firestore
from google.oauth2 import service_account  # ⬅️ agregado

# ————— Configuración —————
PROJECT_ID   = "infractivision-474103"
BUCKET_NAME  = "infractivision-474103"
# BASE_DIR apunta a la raíz del proyecto (dos niveles arriba)
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH  = os.path.join(BASE_DIR, "config", "infractivision_config.json")
INFRA_FILE   = os.path.join(BASE_DIR, "data", "infracciones.json")
INDIC_PATH   = os.path.join(BASE_DIR, "data", "indicadores_rendimiento.json")

# Ruta local a la llave (ajústala si cambias el nombre)
LOCAL_KEY_PATH = os.path.join(BASE_DIR, "secrets", "infractivision-474103-0f907d0fbc62.json")

def _make_clients():
    """Usa ADC si hay GOOGLE_APPLICATION_CREDENTIALS; si no, usa el JSON en /secrets; si nada, ADC por defecto."""
    creds = None
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        pass  # ADC
    elif os.path.exists(LOCAL_KEY_PATH):
        creds = service_account.Credentials.from_service_account_file(LOCAL_KEY_PATH)

    if creds is None:
        storage_client = storage.Client(project=PROJECT_ID)
        fs_client      = firestore.Client(project=PROJECT_ID)
    else:
        storage_client = storage.Client(project=PROJECT_ID, credentials=creds)
        fs_client      = firestore.Client(project=PROJECT_ID, credentials=creds)

    bucket = storage_client.bucket(BUCKET_NAME)
    return storage_client, fs_client, bucket

def _load_ids():
    if os.path.exists(CONFIG_PATH):
        try:
            ids = json.load(open(CONFIG_PATH, encoding="utf-8"))
            # aseguramos username/hostname
            ids.setdefault("username", getpass.getuser())
            ids.setdefault("hostname", socket.gethostname())
            return ids
        except:
            pass
    ids = {
        "user_id":   str(uuid.uuid4()),
        "device_id": str(uuid.uuid4()),
        "username":  getpass.getuser(),
        "hostname":  socket.gethostname()
    }
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2, ensure_ascii=False)
    return ids

def upload_infracciones_automatically():
    ids         = _load_ids()
    user_id     = ids["user_id"]
    device_id   = ids["device_id"]

    storage_client, fs_client, bucket = _make_clients()

    # ——— 1) Infracciones ———
    if os.path.exists(INFRA_FILE):
        with open(INFRA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        infracciones = data.get("infracciones", [])  # Extraer el array de infracciones
        
        # 🆕 NUEVO: Agrupar infracciones por nombre de video Y configuración de semáforo
        infracciones_por_video_config = {}
        for inf in infracciones:
            nombre_video = inf.get("nombre_video", "desconocido.mp4")
            config_semaforo = inf.get("config_semaforo", "sin-configurar")
            
            # Crear clave compuesta: video + config
            key = (nombre_video, config_semaforo)
            
            if key not in infracciones_por_video_config:
                infracciones_por_video_config[key] = []
            infracciones_por_video_config[key].append(inf)
        
        print(f"📹 Infracciones agrupadas en {len(infracciones_por_video_config)} combinaciones video+configuración")
        
        for (nombre_video, config_semaforo), video_config_infracciones in infracciones_por_video_config.items():
            print(f"\n🎥 Procesando {len(video_config_infracciones)} infracciones del video '{nombre_video}' con configuración [{config_semaforo}]")
            
            for inf in video_config_infracciones:
                placa = inf["placa"]
                ts    = inf["video_timestamp"].replace(":", "-")
                doc_id = f"{placa}_{ts}"
                folder = f"evidencias/{user_id}/{device_id}"

                # Rutas a archivos locales
                plate_src   = os.path.join(BASE_DIR, inf["plate_path"])
                vehicle_src = os.path.join(BASE_DIR, inf["vehicle_path"])

                # Sube imágenes
                p_dst = f"{folder}/placas/{doc_id}.jpg"
                v_dst = f"{folder}/vehiculos/{doc_id}.jpg"
                bucket.blob(p_dst).upload_from_filename(plate_src)
                bucket.blob(v_dst).upload_from_filename(vehicle_src)
                url_p = bucket.blob(p_dst).public_url
                url_v = bucket.blob(v_dst).public_url

                # Registra en Firestore - NUEVA ESTRUCTURA: POR VIDEO Y CONFIGURACIÓN
                reg = {
                    "avenida":         inf.get("ubicacion", ""),
                    "fecha":           inf.get("fecha", datetime.now().strftime("%Y-%m-%d")),
                    "placa":           placa,
                    "tipo":            inf.get("tipo", "Semáforo en rojo"),
                    "estado":          inf.get("estado", "Pendiente"),
                    "ubicacion":       inf.get("ubicacion", ""),
                    "hora":            inf.get("hora",  datetime.now().strftime("%H:%M:%S")),
                    "franja_horaria":  inf.get("franja_horaria", ""),
                    "confianza":       inf.get("confianza", 0.0),
                    "calidad":         inf.get("metadata_clasificacion", {}).get("calidad_deteccion", "alta"),
                    "justificacion":   inf.get("metadata_clasificacion", {}).get("justificacion", "Cumple criterios técnicos calibrados"),
                    "tiempo_procesamiento": inf.get("tiempo_procesamiento", 0.0),
                    "url_placa":       url_p,
                    "url_vehiculo":    url_v,
                    "video_timestamp": inf["video_timestamp"],
                    "nombre_video":    nombre_video,
                    "config_semaforo": config_semaforo,  # 🆕 NUEVO: ID de configuración
                    "hostname":        ids["hostname"],
                    "username":        ids["username"]
                }
                # 🆕 NUEVA ESTRUCTURA: usuarios/{user_id}/videos/{nombre_video}/configuraciones/{config_semaforo}/infracciones/{doc_id}
                fs_client \
                  .collection("usuarios") \
                  .document(user_id) \
                  .collection("videos") \
                  .document(nombre_video) \
                  .collection("configuraciones") \
                  .document(config_semaforo) \
                  .collection("infracciones") \
                  .document(doc_id) \
                  .set(reg)
            
            print(f"✔ {len(video_config_infracciones)} infracciones migradas para '{nombre_video}' [{config_semaforo}]")
        
        print(f"\n✔ Total: {len(infracciones)} infracciones migradas de {len(infracciones_por_video_config)} combinaciones video+configuración")

    # ——— 2) Indicadores ———
    if os.path.exists(INDIC_PATH):
        ts_blob = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        folder   = f"evidencias/{user_id}/{device_id}/indicadores"
        fname    = f"indicadores_{ts_blob}.json"
        blob     = bucket.blob(f"{folder}/{fname}")
        blob.upload_from_filename(INDIC_PATH, content_type="application/json")
        storage_url = blob.public_url

        with open(INDIC_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # NUEVA ESTRUCTURA SIMPLIFICADA DE INDICADORES
        # Extraer datos principales
        fecha_actual = datetime.now().strftime("%Y-%m-%d")
        indicadores_data = metrics.get("indicadores", {})
        resumen_global = metrics.get("resumen_global", {})
        
        # 🆕 NUEVO: Obtener nombre del video Y configuración desde el JSON de indicadores
        nombre_video = metrics.get("nombre_video", "desconocido.mp4")
        config_semaforo = metrics.get("config_semaforo", "sin-configurar")
        
        print(f"\n📊 Procesando indicadores para video '{nombre_video}' con configuración [{config_semaforo}]")
        
        # Obtener ubicación/avenida del primer registro de infracciones si existe
        avenida = metrics.get("ubicacion", "N/A")
        if avenida == "N/A" and os.path.exists(INFRA_FILE):
            with open(INFRA_FILE, "r", encoding="utf-8") as f:
                infra_data = json.load(f)
                infracciones = infra_data.get("infracciones", [])
                if infracciones:
                    avenida = infracciones[0].get("ubicacion", "N/A")
        
        # Extraer valores reales de la estructura de indicadores
        nid_data = indicadores_data.get("NID", {})
        nie_data = indicadores_data.get("NIE", {})
        ti_data = indicadores_data.get("TI", {})
        tr_data = indicadores_data.get("TR", {})
        
        # Calcular valores (usar 'total' si existe, si no usar 'infracciones_hoy')
        nid_valor = nid_data.get("total", nid_data.get("infracciones_hoy", 0))
        nie_valor = nie_data.get("total", nie_data.get("infracciones_incorrectas", 0)) if nie_data else 0
        ti_valor = ti_data.get("porcentaje_acierto", 0)
        
        # TR: Extraer tiempos individuales y promedio
        tr_con_software = tr_data.get("con_software", {})
        tr_promedio = tr_con_software.get("tiempo_promedio_minutos", 0)
        tr_individuales = tr_con_software.get("tiempos_individuales", [])
        
        tir_valor = nid_valor + nie_valor
        
        flat = {
            "avenida": avenida,
            "nombre_video": nombre_video,
            "config_semaforo": config_semaforo,  # 🆕 NUEVO: ID de configuración
            "fecha": fecha_actual,
            
            # NID - Número de Infracciones Detectadas
            "NID": nid_valor,
            "descripcion_NID": "Número de Infracciones correctamente detectadas y registradas",
            
            # NIE - Número de Infracciones incorrectamente registradas (si existe)
            "NIE": nie_valor,
            "descripcion_NIE": "Número de Infracciones incorrectamente registradas",
            
            # TI - Tasa de Infracciones
            "TI": ti_valor,
            "descripcion_TI": "Tasa de Infracciones correctamente detectadas (% de acierto)",
            
            # TR - Tiempo de Registro (promedio e individuales)
            "TR_promedio": tr_promedio,
            "TR_individuales": tr_individuales,
            "descripcion_TR": "Tiempo de Registro por infracción - Promedio y tiempos individuales (minutos)",
            
            # TIR - Total de Infracciones Reales (NID + NIE)
            "TIR": tir_valor,
            "descripcion_TIR": "Total de Infracciones Reales (NID + NIE)",
            
            # Campos adicionales
            "promedio_infracciones_diarias": nid_data.get("promedio_diario", 0),
            "dias_analizados": metrics.get("dias_analizados", 0),
            "total_muestras_analizadas": tr_data.get("con_software", {}).get("muestras_analizadas", 0),
            
            "url_evidencia": storage_url
        }
        
        # DEBUG: Imprimir valores extraídos para verificar
        print(f"  NID: {nid_valor}")
        print(f"  NIE: {nie_valor}")
        print(f"  TI: {ti_valor}")
        print(f"  TR promedio: {tr_promedio}")
        print(f"  TR individuales: {tr_individuales}")
        print(f"  TIR: {tir_valor}")

        # 🆕 NUEVA ESTRUCTURA: usuarios/{user_id}/videos/{nombre_video}/configuraciones/{config_semaforo}/indicadores/resumen
        fs_client \
          .collection("usuarios") \
          .document(user_id) \
          .collection("videos") \
          .document(nombre_video) \
          .collection("configuraciones") \
          .document(config_semaforo) \
          .collection("indicadores") \
          .document("resumen") \
          .set(flat, merge=True)
        print(f"✔ Indicadores guardados en Firestore con estructura por video+configuración.")
        print(f"✔ Ruta: usuarios/{user_id}/videos/{nombre_video}/configuraciones/{config_semaforo}/indicadores/resumen")
    else:
        print("ℹ️ No se encontró indicadores_rendimiento.json, omito carga.")

if __name__ == "__main__":
    print("🚀 Iniciando migración automática de infracciones e indicadores...")
    upload_infracciones_automatically()
    print("✅ Migración completada.")
