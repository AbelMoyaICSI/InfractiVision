import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from google.cloud import storage, firestore
from google.oauth2 import service_account  # ⬅️ agregado

app = Flask(__name__)

# ======== Credenciales (mínimo e inocuo) ========
PROJECT_ID  = "infractivision-461115"
BUCKET_NAME = "infractivision-2025"

# Ruta local a la llave (ajústala si cambias el nombre)
LOCAL_KEY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "secrets", "infractivision-461115-010a42885008.json")
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
# Detecta dinámicamente la raíz del proyecto (la carpeta que contiene 'data/')
root = os.path.abspath(os.path.dirname(__file__))
while True:
    if os.path.isdir(os.path.join(root, "data")):
        break
    parent = os.path.dirname(root)
    if parent == root:
        raise RuntimeError("No se encontró la carpeta 'data/' en los niveles superiores.")
    root = parent

INDICADORES_PATH = os.path.join(root, "data", "indicadores_rendimiento.json")
# ————————————————

@app.route("/", methods=["GET"])
def home():
    return "✅ Backend de InfractiVision operativo", 200

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
        if not os.path.exists(INDICADORES_PATH):
            return jsonify({"status": "error", "msg": "No hay indicadores disponibles"}), 404
        return send_file(INDICADORES_PATH, mimetype="application/json")

    try:
        # 1) Subir JSON completo de indicadores a Storage
        ts     = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        folder = f"evidencias/{user_id}/indicadores"
        fname  = f"indicadores_{ts}.json"
        blob   = bucket.blob(f"{folder}/{fname}")
        blob.upload_from_filename(INDICADORES_PATH, content_type="application/json")
        storage_url = blob.public_url

        # 2) Leer localmente el JSON
        with open(INDICADORES_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

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
