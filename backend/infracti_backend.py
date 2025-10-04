import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from google.cloud import storage, firestore
from google.oauth2 import service_account  # ⬅️ agregado

app = Flask(__name__)

# ======== Credenciales (mínimo e inocuo) ========
PROJECT_ID  = "infractivision-474103"
BUCKET_NAME = "infractivision-474103"

# Ruta local a la llave (ajústala si cambias el nombre)
LOCAL_KEY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "secrets", "infractivision-474103-0f907d0fbc62.json")
)

def _make_clients():
    """
    - Si hay GOOGLE_APPLICATION_CREDENTIALS, usa ADC.
    - Si existe el JSON en /secrets, úsalo explícitamente.
    - Si estás en Cloud Run con SA adjunta, ADC funciona sin JSON.
    """
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

# Inicializa clientes de Firestore y Storage (usando helper)
db, storage_client = _make_clients()

# Configuración del bucket (ajusta el nombre si cambia)
bucket = storage_client.bucket(BUCKET_NAME)
# ======== fin credenciales ========

# ————————————————
# En Cloud Run no necesitamos rutas locales, todo va a Firestore/Storage
# ————————————————

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok", 
        "message": "InfractiVision Backend Operativo",
        "version": "1.0.0",
        "project": PROJECT_ID,
        "bucket": BUCKET_NAME
    }), 200

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de salud para verificar que el servicio funciona"""
    try:
        # Probar conexión a Firestore
        db.collection("health").document("test").set({"timestamp": datetime.utcnow()})
        return jsonify({"status": "healthy", "firestore": "ok", "storage": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/migrar-json", methods=["POST"])
def migrar_json_completo():
    """Migrar infracciones desde JSON local completo"""
    try:
        user_id = request.form.get("user_id", "usuario_local")
        
        # Recibir el JSON de infracciones
        json_file = request.files.get("infracciones_json")
        if not json_file:
            return jsonify({"error": "Falta archivo infracciones.json"}), 400
        
        # Leer y parsear JSON
        infracciones_data = json.loads(json_file.read().decode('utf-8'))
        infracciones = infracciones_data.get("infracciones", [])
        
        migradas = 0
        errores = []
        
        for infraccion in infracciones:
            try:
                placa = infraccion["placa"]
                ts = infraccion["video_timestamp"].replace(":", "-")
                doc_id = f"{placa}_{ts}"
                
                # Preparar datos completos para Firestore
                reg = {
                    # Campos básicos
                    "placa": placa,
                    "fecha": infraccion.get("fecha", ""),
                    "hora": infraccion.get("hora", ""),
                    "video_timestamp": infraccion["video_timestamp"],
                    "ubicacion": infraccion.get("ubicacion", ""),
                    "tipo": infraccion.get("tipo", "Semáforo en rojo"),
                    "estado": infraccion.get("estado", "Pendiente"),
                    
                    # Campos nuevos
                    "tiempo_video": infraccion.get("tiempo_video", ""),
                    "franja_horaria": infraccion.get("franja_horaria", ""),
                    "clasificacion": infraccion.get("clasificacion", ""),
                    "confianza": infraccion.get("confianza", 0.0),
                    "tiempo_procesamiento": infraccion.get("tiempo_procesamiento", 0.0),
                    "sistema_version": infraccion.get("sistema_version", ""),
                    
                    # Metadata aplanado
                    "metadata_placa_final": infraccion.get("metadata_clasificacion", {}).get("placa_final", ""),
                    "metadata_confianza": infraccion.get("metadata_clasificacion", {}).get("confianza", 0.0),
                    "metadata_calidad": infraccion.get("metadata_clasificacion", {}).get("calidad_deteccion", ""),
                    "metadata_justificacion": infraccion.get("metadata_clasificacion", {}).get("justificacion", ""),
                    
                    # Sistema
                    "user_id": user_id,
                    "migrated_at": datetime.utcnow()
                }
                
                # Guardar en Firestore
                db.collection("usuarios").document(user_id).collection("infracciones").document(doc_id).set(reg)
                migradas += 1
                
            except Exception as e:
                errores.append(f"Error con {infraccion.get('placa', 'UNKNOWN')}: {str(e)}")
        
        return jsonify({
            "status": "ok",
            "migradas": migradas,
            "total": len(infracciones),
            "errores": errores
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/migrar", methods=["POST"])
def migrar_datos():
    try:
        user_id = request.form.get("user_id", "anonimo")
        placa   = request.form.get("placa", "SIN_PLACA")
        ts      = request.form.get("video_timestamp", "00-00")
        doc_id  = f"{placa}_{ts}"

        fv = request.files.get("img_vehiculo")
        fp = request.files.get("img_placa")
        if not fv or not fp:
            return jsonify({"status": "error", "msg": "Faltan archivos"}), 400

        # 1) Subir imágenes al bucket
        blob_v = bucket.blob(f"evidencias/vehiculos/{user_id}/{doc_id}.jpg")
        blob_p = bucket.blob(f"evidencias/placas/{user_id}/{doc_id}.jpg")
        blob_v.upload_from_file(fv.stream, content_type=fv.content_type)
        blob_p.upload_from_file(fp.stream, content_type=fp.content_type)
        url_v = blob_v.public_url
        url_p = blob_p.public_url

        # 2) Guardar meta datos en Firestore
        doc_ref = (
            db.collection("usuarios")
              .document(user_id)
              .collection("infracciones")
              .document(doc_id)
        )
        doc_ref.set({
            "placa":           placa,
            "fecha":           request.form.get("fecha", ""),
            "hora":            request.form.get("hora", ""),
            "video_timestamp": ts,
            "ubicacion":       request.form.get("ubicacion", ""),
            "tipo":            request.form.get("tipo", ""),
            "estado":          request.form.get("estado", ""),
            "vehicle_url":     url_v,
            "plate_url":       url_p
        })

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route("/indicadores/<user_id>", methods=["GET", "POST"])
def indicadores(user_id):
    if request.method == "GET":
        # Obtener indicadores desde Firestore
        try:
            docs = db.collection("usuarios").document(user_id).collection("indicadores").get()
            indicadores_list = [doc.to_dict() for doc in docs]
            return jsonify({"indicadores": indicadores_list}), 200
        except Exception as e:
            return jsonify({"status": "error", "msg": str(e)}), 500

    try:
        # Recibir datos JSON desde el cliente
        metrics = request.get_json()
        if not metrics:
            return jsonify({"status": "error", "msg": "No se enviaron datos"}), 400

        # 1) Subir JSON completo de indicadores a Storage
        ts     = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        folder = f"evidencias/{user_id}/indicadores"
        fname  = f"indicadores_{ts}.json"
        blob   = bucket.blob(f"{folder}/{fname}")
        blob.upload_from_string(json.dumps(metrics, ensure_ascii=False, indent=2), content_type="application/json")
        storage_url = blob.public_url

        # 3) Aplanar en dos niveles
        flat = {
            "user_id":      user_id,
            "fecha_subida": ts,
            "storage_url":  storage_url
        }
        # Campos de nivel superior
        for key, val in metrics.items():
            if key not in ("indicadores", "resumen_global"):
                flat[key] = val

        # indicadores.TI.*, indicadores.TR.*, indicadores.IR.* con dos niveles
        for sec, secdict in metrics.get("indicadores", {}).items():
            for subk, subv in secdict.items():
                if isinstance(subv, dict):
                    # Desciende un nivel más
                    for inner_k, inner_v in subv.items():
                        flat[f"indicadores_{sec}_{subk}_{inner_k}"] = inner_v
                else:
                    flat[f"indicadores_{sec}_{subk}"] = subv

        # resumen_global.* (ya todo escalar)
        for subk, subv in metrics.get("resumen_global", {}).items():
            flat[f"resumen_{subk}"] = subv

        # 4) Guardar objeto plano en Firestore
        doc_ref = (
            db.collection("usuarios")
              .document(user_id)
              .collection("indicadores")
              .document(ts)
        )
        doc_ref.set(flat)

        return jsonify({"ok": True, "doc_id": ts}), 200

    except Exception as e:
        import traceback
        print("❌ Error en /indicadores/<user_id>:")
        traceback.print_exc()  # ← Esto imprimirá el error en los logs de Cloud Run
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
