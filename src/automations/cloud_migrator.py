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
        for inf in infracciones:
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

            # Registra en Firestore - TODOS LOS CAMPOS
            reg = {
                # Campos básicos
                "placa":           placa,
                "fecha":           inf.get("fecha", datetime.now().strftime("%d/%m/%Y")),
                "hora":            inf.get("hora",  datetime.now().strftime("%H:%M:%S")),
                "video_timestamp": inf["video_timestamp"],
                "ubicacion":       inf.get("ubicacion", ""),
                "tipo":            inf.get("tipo", "Semáforo en rojo"),
                "estado":          inf.get("estado", "Pendiente"),
                
                # Campos nuevos de la estructura actualizada
                "tiempo_video":    inf.get("tiempo_video", ""),
                "franja_horaria":  inf.get("franja_horaria", ""),
                "clasificacion":   inf.get("clasificacion", ""),
                "confianza":       inf.get("confianza", 0.0),
                "tiempo_procesamiento": inf.get("tiempo_procesamiento", 0.0),
                "sistema_version": inf.get("sistema_version", ""),
                
                # Metadata de clasificación (aplanado)
                "metadata_placa_final": inf.get("metadata_clasificacion", {}).get("placa_final", ""),
                "metadata_confianza": inf.get("metadata_clasificacion", {}).get("confianza", 0.0),
                "metadata_calidad": inf.get("metadata_clasificacion", {}).get("calidad_deteccion", ""),
                "metadata_justificacion": inf.get("metadata_clasificacion", {}).get("justificacion", ""),
                
                # Campos del sistema
                "device_id":       device_id,
                "user_id":         user_id,
                "username":        ids["username"],
                "hostname":        ids["hostname"],
                "plate_url":       url_p,
                "vehicle_url":     url_v,
                "uploaded_at":     datetime.utcnow()
            }
            fs_client \
              .collection("usuarios") \
              .document(user_id) \
              .collection("infracciones") \
              .document(doc_id) \
              .set(reg)
        print("✔ Infracciones migradas.")

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

        flat = {
            "user_id":      user_id,
            "device_id":    device_id,
            "username":     ids["username"],   # 👈 agregado
            "hostname":     ids["hostname"],   # 👈 agregado
            "fecha_subida": ts_blob,
            "storage_url":  storage_url
        }
        for key, val in metrics.items():
            if key not in ("indicadores", "resumen_global"):
                flat[key] = val

        for sec, secdict in metrics.get("indicadores", {}).items():
            for subk, subv in secdict.items():
                if isinstance(subv, dict):
                    for inner_k, inner_v in subv.items():
                        flat[f"indicadores_{sec}_{subk}_{inner_k}"] = inner_v
                else:
                    flat[f"indicadores_{sec}_{subk}"] = subv

        for subk, subv in metrics.get("resumen_global", {}).items():
            flat[f"resumen_{subk}"] = subv

        fs_client \
          .collection("usuarios") \
          .document(user_id) \
          .collection("indicadores") \
          .document(ts_blob) \
          .set(flat, merge=True)
        print("✔ Indicadores guardados en Firestore de forma plana.")
    else:
        print("ℹ️ No se encontró indicadores_rendimiento.json, omito carga.")

if __name__ == "__main__":
    upload_infracciones_automatically()
