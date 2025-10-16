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
                
                # Preparar datos completos para Firestore - NUEVA ESTRUCTURA SIMPLIFICADA
                reg = {
                    "avenida": infraccion.get("ubicacion", ""),
                    "fecha": infraccion.get("fecha", ""),
                    "placa": placa,
                    "tipo": infraccion.get("tipo", "Semáforo en rojo"),
                    "estado": infraccion.get("estado", "Pendiente"),
                    "ubicacion": infraccion.get("ubicacion", ""),
                    "hora": infraccion.get("hora", ""),
                    "franja_horaria": infraccion.get("franja_horaria", ""),
                    "confianza": infraccion.get("confianza", 0.0),
                    "calidad": infraccion.get("metadata_clasificacion", {}).get("calidad_deteccion", "alta"),
                    "justificacion": infraccion.get("metadata_clasificacion", {}).get("justificacion", "Cumple criterios técnicos calibrados"),
                    "tiempo_procesamiento": infraccion.get("tiempo_procesamiento", 0.0),
                    "video_timestamp": infraccion["video_timestamp"],
                    "hostname": infraccion.get("hostname", ""),
                    "username": infraccion.get("username", "")
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
        # Obtener indicadores desde Firestore (todos los videos+configs del usuario)
        try:
            # Obtener todos los documentos de videos del usuario
            videos_ref = db.collection("usuarios").document(user_id).collection("videos")
            videos = videos_ref.get()
            
            indicadores_list = []
            for video_doc in videos:
                nombre_video = video_doc.id
                # 🆕 NUEVO: Obtener todas las configuraciones de cada video
                configs_ref = videos_ref.document(nombre_video).collection("configuraciones")
                configs = configs_ref.get()
                
                for config_doc in configs:
                    config_semaforo = config_doc.id
                    # Obtener indicadores de cada configuración
                    indicadores_doc = configs_ref.document(config_semaforo).collection("indicadores").document("resumen").get()
                    if indicadores_doc.exists:
                        data = indicadores_doc.to_dict()
                        data["nombre_video"] = nombre_video
                        data["config_semaforo"] = config_semaforo
                        indicadores_list.append(data)
            
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

        # NUEVA ESTRUCTURA SIMPLIFICADA DE INDICADORES
        # Extraer datos principales
        fecha_actual = datetime.utcnow().strftime("%Y-%m-%d")
        indicadores_data = metrics.get("indicadores", {})
        resumen_global = metrics.get("resumen_global", {})
        
        # 🆕 NUEVO: Obtener nombre del video Y configuración desde el JSON de indicadores
        nombre_video = metrics.get("nombre_video", "desconocido.mp4")
        config_semaforo = metrics.get("config_semaforo", "sin-configurar")
        
        print(f"📊 Backend: Guardando indicadores para video '{nombre_video}' con configuración [{config_semaforo}]")
        
        # Obtener ubicación/avenida si está disponible
        avenida = metrics.get("ubicacion", "N/A")
        
        # Extraer valores reales de la estructura de indicadores
        nid_data = indicadores_data.get("NID", {})
        nie_data = indicadores_data.get("NIE", {})
        ti_data = indicadores_data.get("TI", {})
        tr_data = indicadores_data.get("TR", {})
        
        # Calcular valores (usar 'total' si existe, si no usar 'infracciones_hoy')
        nid_valor = nid_data.get("total", nid_data.get("infracciones_hoy", 0))
        nie_valor = nie_data.get("total", nie_data.get("infracciones_incorrectas", 0)) if nie_data else 0
        ti_valor = ti_data.get("porcentaje_acierto", 0)
        tr_valor = tr_data.get("con_software", {}).get("tiempo_promedio_minutos", 0)
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
            "descripcion_TI": "Tasa de Infracciones correctamente detectadas y registradas (%)",
            
            # TR - Tiempo de Registro
            "TR": tr_valor,
            "descripcion_TR": "Tiempo de Registro por infracción (minutos)",
            
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
        print(f"  TR: {tr_valor}")
        print(f"  TIR: {tir_valor}")

        # 4) Guardar objeto plano en Firestore - NUEVA ESTRUCTURA POR VIDEO Y CONFIGURACIÓN
        # Ruta: usuarios/{user_id}/videos/{nombre_video}/configuraciones/{config_semaforo}/indicadores/resumen
        doc_ref = (
            db.collection("usuarios")
              .document(user_id)
              .collection("videos")
              .document(nombre_video)
              .collection("configuraciones")
              .document(config_semaforo)
              .collection("indicadores")
              .document("resumen")
        )
        doc_ref.set(flat, merge=True)
        
        print(f"✔ Indicadores guardados en Firestore")
        print(f"✔ Ruta: usuarios/{user_id}/videos/{nombre_video}/configuraciones/{config_semaforo}/indicadores/resumen")

        return jsonify({"ok": True, "doc_id": ts, "video": nombre_video, "config": config_semaforo}), 200

    except Exception as e:
        import traceback
        print("❌ Error en /indicadores/<user_id>:")
        traceback.print_exc()  # ← Esto imprimirá el error en los logs de Cloud Run
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
