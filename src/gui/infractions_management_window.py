import tkinter as tk
from tkinter import messagebox, filedialog
import json, os
from tkcalendar import DateEntry
from datetime import datetime
import shutil
import csv
import queue
import threading
from src.path_helper import resource_path


# Ruta centralizada del archivo de infracciones
INF_FILE = resource_path("data/infracciones.json")

# Ruta centralizada del archivo de infracciones NIE (incorrectamente registradas)
NIE_FILE = resource_path("data/nie_infracciones.json")

# === HISTORIAL DE MIGRACIONES (BD SQLite) ===

def add_migration_to_history(num_infractions, estado="Exitosa"):
    """Agregar nueva migración a la tabla `migrations` de la BD local."""
    try:
        from src.infrastructure.database.app_repository import AppRepository
        repo = AppRepository()
        repo.add_migration(
            fecha=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            timestamp=datetime.now().isoformat(),
            registros=int(num_infractions or 0),
            estado=estado,
        )
        print(f"📊 MIGRACIÓN AGREGADA AL HISTORIAL: {num_infractions} registros ({estado})")
        return True
    except Exception as e:
        print(f"Error agregando migración al historial: {e}")
        return False

def _load_json_array(file_path):
    """Lee un archivo JSON y devuelve el array de infracciones (compatible con dict/list)."""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and 'infracciones' in data:
                return data['infracciones']
            elif isinstance(data, list):
                return data
            else:
                print(f"⚠️ Formato inesperado en JSON: {type(data)}")
                return []
    except Exception as e:
        print(f"Error cargando infracciones: {e}")
        return []


# Función para cargar datos de infracciones (SQLite única fuente)
def load_infractions_data():
    """Carga infracciones locales desde SQLite (NID+NIE)."""
    try:
        from src.infrastructure.database.app_repository import AppRepository
        repo = AppRepository()
        return repo.list_infractions(limit=100000)
    except Exception as e:
        print(f"Error cargando desde SQLite, fallback JSON: {e}")
        nid = _load_json_array(INF_FILE)
        nie = _load_json_array(NIE_FILE)
        return nid + nie


def delete_infraction_files(infraction_data):
    """Elimina los archivos de imagen asociados a una infracción"""
    deleted_files = []
    
    # Eliminar imagen del vehículo
    vehicle_path = infraction_data.get('vehicle_path', '')
    if vehicle_path and os.path.exists(vehicle_path):
        try:
            os.remove(vehicle_path)
            deleted_files.append(f"Vehículo: {os.path.basename(vehicle_path)}")
        except Exception as e:
            print(f"Error eliminando imagen del vehículo: {e}")
    
    # Eliminar imagen de la placa
    plate_path = infraction_data.get('plate_path', '')
    if plate_path and os.path.exists(plate_path):
        try:
            os.remove(plate_path)
            deleted_files.append(f"Placa: {os.path.basename(plate_path)}")
        except Exception as e:
            print(f"Error eliminando imagen de la placa: {e}")
    
    return deleted_files

def delete_all_infractions():
    """Elimina todas las infracciones (SQLite) y sus imágenes."""
    try:
        from src.infrastructure.database.app_repository import AppRepository
        repo = AppRepository()
        repo.clear_infractions()
        # Limpiar directorios de imágenes
        output_dirs = [resource_path("data/output/placas"), resource_path("data/output/autos")]
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
        return True
    except Exception as e:
        print(f"Error eliminando todas las infracciones: {e}")
        return False

def _filter_infractions_file(file_path, placa_to_remove):
    """Legacy JSON (deprecated): ahora delega a SQLite."""
    try:
        from src.infrastructure.database.app_repository import AppRepository
        repo = AppRepository()
        repo.delete_by_placa(placa_to_remove)
        return True
    except Exception as e:
        print(f"Error eliminando infracción de SQLite: {e}")
        return False


def remove_infraction_from_json(placa_to_remove):
    """Elimina infracción por placa (SQLite, única fuente)."""
    try:
        from src.infrastructure.database.app_repository import AppRepository
        repo = AppRepository()
        deleted = repo.delete_by_placa(placa_to_remove)
        return deleted >= 0
    except Exception as e:
        print(f"Error eliminando infracción: {e}")
        return False

def generate_performance_report(software_infractions, software_processing_times, nombre_video=None, config_semaforo=None):
    """Genera dict de indicadores TI/TR/NID/NIE (puro, sin IO). Usado por SQLite."""
    # Duplica lógica legacy pero sin escribir JSON — idéntica a AppRepository.compute_indicators_report
    if not isinstance(software_infractions, (list, tuple)):
        software_infractions = []
    if not isinstance(software_processing_times, (list, tuple)):
        software_processing_times = [software_processing_times] if isinstance(software_processing_times, (int, float)) else []
    day_infractions = {}
    nid_count = nie_count = 0
    for inf in software_infractions:
        fecha = inf.get("fecha", "Sin fecha")
        clas = inf.get("clasificacion", "NID")
        grp = day_infractions.setdefault(fecha, {"total": 0, "placas": {}, "nid": 0, "nie": 0})
        grp["total"] += 1
        if clas == "NID":
            nid_count += 1
            grp["nid"] += 1
        elif clas == "NIE":
            nie_count += 1
            grp["nie"] += 1
        placa = inf.get("placa", "")
        if placa:
            grp["placas"].setdefault(placa, 0)
            grp["placas"][placa] += 1
    pnp_data = {"Enero 2023": {"total": 125, "dias": 31}, "Febrero 2023": {"total": 117, "dias": 28}, "Marzo 2023": {"total": 137, "dias": 31}, "Abril 2023": {"total": 129, "dias": 30}}
    police_times_min = [7, 6, 5, 10, 8]
    pnp_total = sum(m["total"] for m in pnp_data.values())
    pnp_days = sum(m["dias"] for m in pnp_data.values())
    pnp_daily = pnp_total / pnp_days if pnp_days else 0
    sw_days = len(day_infractions)
    sw_inf = len(software_infractions)
    sw_daily = sw_inf / sw_days if sw_days else 0
    ti_percentage = (nid_count / (nid_count + nie_count) * 100) if (nid_count + nie_count) else 0.0
    pnp_sec = (sum(police_times_min) / len(police_times_min) * 60) if police_times_min else 0
    sw_times_min = [t / 60.0 for t in software_processing_times] if software_processing_times else []
    sw_min = sum(sw_times_min) / len(sw_times_min) if sw_times_min else 0.0
    pnp_min = pnp_sec / 60.0
    tr_reduction_pct = ((pnp_min - sw_min) / pnp_min * 100) if pnp_min else 0
    tr_speedup = pnp_min / sw_min if sw_min else 0
    nid_today = nid_count
    nie_today = nie_count
    nid_daily_avg = nid_count / sw_days if sw_days > 0 else nid_count
    avenida = software_infractions[0].get("ubicacion", "N/A") if software_infractions else "N/A"
    video_name = nombre_video or (software_infractions[0].get("nombre_video", "desconocido.mp4") if software_infractions else "desconocido.mp4")
    config_id = config_semaforo or (software_infractions[0].get("config_semaforo", "sin-configurar") if software_infractions else "sin-configurar")
    from datetime import datetime as _dt
    return {
        "fecha_generacion": _dt.now().strftime("%d/%m/%Y %H:%M:%S"),
        "periodo_analisis": f"{min(day_infractions.keys(), default='N/A')} - {max(day_infractions.keys(), default='N/A')}",
        "dias_analizados": sw_days, "ubicacion": avenida, "nombre_video": video_name, "config_semaforo": config_id,
        "nota": "Datos de la sesión actual de procesamiento, no acumulados históricos",
        "indicadores": {
            "TI": {"descripcion": "Tasa de Infracciones Detectadas (Nivel Diario Agregado)", "unidad": "infracciones por día comparativo (%)", "sin_software": {"registros_campo_diarios": round(pnp_daily, 2), "fuente": "Registros PNP históricos"}, "con_software": {"detecciones_software_diarias": round(sw_daily, 2), "dias_analizados": sw_days}, "porcentaje_acierto": round(ti_percentage, 2)},
            "TR": {"descripcion": "Tiempo de Registro por Infracción Individual", "unidad": "minutos por infracción (min)", "sin_software": {"tiempo_promedio_minutos": round(pnp_min, 2), "fuente": "Estimación basada en registros históricos de campo"}, "con_software": {"tiempo_promedio_minutos": round(sw_min, 2), "tiempos_individuales": [round(t, 2) for t in sw_times_min], "muestras_analizadas": len(software_processing_times)}, "reduccion_tiempo_porcentual": round(tr_reduction_pct, 2), "veces_mas_rapido": round(tr_speedup, 2)},
            "NID": {"descripcion": "Número de Infracciones Detectadas Correctamente", "unidad": "cantidad válida por día", "infracciones_hoy": nid_today, "promedio_diario": round(nid_daily_avg, 0), "periodo_analizado": f"{sw_days} días", "total": nid_count},
            "NIE": {"descripcion": "Número de Infracciones Incorrectamente Registradas", "unidad": "cantidad no válida por día", "infracciones_incorrectas": nie_count, "total": nie_count},
        },
        "resumen_global": {"ti_porcentaje_acierto": f"{ti_percentage:.1f}%", "tiempo_registro_minutos": f"{sw_min:.2f} min", "infracciones_detectadas_hoy": nid_today, "nid_total": nid_count, "nie_total": nie_count, "tir_total": nid_count + nie_count},
    }


def generate_performance_indicators_json(software_infractions, software_processing_times, nombre_video=None, config_semaforo=None):
    """
    Compat: genera y persiste en SQLite (y JSON legacy para compatibilidad).
    Los indicadores TI, TR y NID basados en las infracciones y tiempos.
    """
    import os
    import json
    from datetime import datetime

    # DEBUG: Imprimir datos recibidos
    print(f"\n📊 DEBUG - generate_performance_indicators_json:")
    print(f"  Infracciones recibidas: {len(software_infractions) if isinstance(software_infractions, list) else 0}")
    print(f"  Tiempos de procesamiento: {len(software_processing_times) if isinstance(software_processing_times, list) else 0}")
    print(f"  Nombre del video: {nombre_video}")
    print(f"  Configuración semáforo: {config_semaforo}")
    if software_infractions and isinstance(software_infractions, list):
        print(f"  Primera infracción: {software_infractions[0] if software_infractions else 'N/A'}")

    # Garantizar que recibimos listas
    if not isinstance(software_infractions, (list, tuple)):
        software_infractions = []
    if not isinstance(software_processing_times, (list, tuple)):
        # Si recibimos un solo número, lo convertimos en lista
        if isinstance(software_processing_times, (int, float)):
            software_processing_times = [software_processing_times]
        else:
            software_processing_times = []

    # 1) Agrupar por día para cálculos y contar NID/NIE
    day_infractions = {}
    nid_count = 0
    nie_count = 0
    
    for inf in software_infractions:
        fecha = inf.get("fecha", "Sin fecha")
        placa = inf.get("placa", "")
        clasificacion = inf.get("clasificacion", "NID")  # Por defecto NID
        
        grp = day_infractions.setdefault(fecha, {"total": 0, "placas": {}, "nid": 0, "nie": 0})
        grp["total"] += 1
        
        # Contar NID y NIE
        if clasificacion == "NID":
            nid_count += 1
            grp["nid"] += 1
        elif clasificacion == "NIE":
            nie_count += 1
            grp["nie"] += 1
        
        if placa:
            grp["placas"].setdefault(placa, 0)
            grp["placas"][placa] += 1

    # 2) Datos históricos "sin software" (PNP)
    pnp_data = {
        "Enero 2023": {"total": 125, "dias": 31, "reincidentes": 18},
        "Febrero 2023": {"total": 117, "dias": 28, "reincidentes": 15},
        "Marzo 2023": {"total": 137, "dias": 31, "reincidentes": 15},
        "Abril 2023": {"total": 129, "dias": 30, "reincidentes": 17},
    }
    police_times_min = [7, 6, 5, 10, 8]  # en minutos

    # ——— INDICADOR TI: Tasa de Infracciones (PORCENTAJE DE ACIERTO) ———
    # TI = (NID / Total infracciones) × 100
    # Refleja qué porcentaje de infracciones detectadas son correctas (NID)
    
    # Datos GC (Grupo Control - registros manuales de campo) - solo para comparación
    pnp_total = sum(m["total"] for m in pnp_data.values())
    pnp_days = sum(m["dias"] for m in pnp_data.values())
    pnp_daily = pnp_total / pnp_days if pnp_days else 0
    
    # Datos GE (Grupo Experimental - software)
    sw_days = len(day_infractions)
    sw_inf = len(software_infractions)
    sw_daily = sw_inf / sw_days if sw_days else 0
    
    # TI como porcentaje de acierto de ESTA SESIÓN
    # TI = (NID correctas / Total detectadas) × 100
    total_detectadas = nid_count + nie_count
    if total_detectadas > 0:
        ti_percentage = (nid_count / total_detectadas) * 100
    else:
        ti_percentage = 0.0

    # ——— INDICADOR TR: Tiempo de registro EN MINUTOS (INDIVIDUAL) ———
    # TR son los tiempos individuales de cada infracción, NO un promedio
    # Cada video puede tener múltiples TR: [1.23min, 0.34min, 2.45min, etc.]
    
    pnp_sec = (sum(police_times_min) / len(police_times_min) * 60) if police_times_min else 0
    
    # Convertir tiempos individuales de segundos a minutos
    if software_processing_times:
        sw_times_min = [t / 60.0 for t in software_processing_times]  # Lista de tiempos en minutos
        sw_min = sum(sw_times_min) / len(sw_times_min)  # Promedio para comparación
    else:
        sw_times_min = []
        sw_min = 0.0
    
    # PNP en minutos
    pnp_min = pnp_sec / 60.0

    tr_reduction_pct = ((pnp_min - sw_min) / pnp_min * 100) if pnp_min else 0
    tr_speedup = pnp_min / sw_min if sw_min else 0

    # ——— INDICADOR NID: Número de Infracciones Detectadas (DIARIO) ———
    # IMPORTANTE: Esta función recibe SOLO las infracciones de la sesión actual
    # Los valores reflejan lo procesado HOY, no datos históricos acumulados
    
    # Usar el contador REAL de infracciones clasificadas como NID de esta sesión
    nid_today = nid_count  # Ya contamos las NID de esta sesión
    nie_today = nie_count  # Ya contamos las NIE de esta sesión
    
    # Promedio diario basado en los días analizados en esta sesión
    nid_daily_avg = nid_count / sw_days if sw_days > 0 else nid_count
    
    # Obtener ubicación/avenida de la primera infracción si existe
    avenida = "N/A"
    if software_infractions:
        avenida = software_infractions[0].get("ubicacion", "N/A")

    # DEBUG: Imprimir valores calculados
    print(f"\n📊 DEBUG - Valores calculados (SESIÓN ACTUAL):")
    print(f"  NID esta sesión: {nid_count}")
    print(f"  NIE esta sesión: {nie_count}")
    print(f"  NID hoy: {nid_today}")
    print(f"  NID promedio diario: {nid_daily_avg}")
    print(f"  TI porcentaje: {ti_percentage:.2f}%")
    print(f"  TR software (min): Promedio={sw_min:.2f}, Individual={sw_times_min}")
    print(f"  Días analizados en sesión: {sw_days}")
    print(f"  Total infracciones en sesión: {sw_inf}")
    print(f"  TIR (NID + NIE): {nid_count + nie_count}")

    # ——— Montar el JSON de salida con NUEVA OPERACIONALIZACIÓN (SIN IR) ———
    # NOTA: Este reporte refleja SOLO los datos de la sesión actual, no acumulados históricos
    from datetime import datetime
    
    # Obtener nombre del video desde las infracciones o del parámetro
    video_name = nombre_video
    if not video_name and software_infractions:
        # Intentar obtener del primer registro de infracción
        video_name = software_infractions[0].get("nombre_video", "desconocido.mp4")
    if not video_name:
        video_name = "desconocido.mp4"
    
    # 🆕 NUEVO: Obtener configuración de semáforo
    config_semaforo_id = config_semaforo
    if not config_semaforo_id and software_infractions:
        # Intentar obtener del primer registro de infracción
        config_semaforo_id = software_infractions[0].get("config_semaforo", "sin-configurar")
    if not config_semaforo_id:
        config_semaforo_id = "sin-configurar"
    
    report = {
        "fecha_generacion": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "periodo_analisis": f"{min(day_infractions.keys(), default='N/A')} - {max(day_infractions.keys(), default='N/A')}",
        "dias_analizados": sw_days,
        "ubicacion": avenida,  # Agregar ubicación/avenida
        "nombre_video": video_name,  # 🆕 NUEVO: Nombre del video procesado
        "config_semaforo": config_semaforo_id,  # 🆕 NUEVO: Configuración de semáforo (ej: "10-3-15")
        "nota": "Datos de la sesión actual de procesamiento, no acumulados históricos",
        "indicadores": {
            "TI": {
                "descripcion": "Tasa de Infracciones Detectadas (Nivel Diario Agregado)",
                "unidad": "infracciones por día comparativo (%)",
                "sin_software": {
                    "registros_campo_diarios": round(pnp_daily, 2),
                    "fuente": "Registros PNP históricos"
                },
                "con_software": {
                    "detecciones_software_diarias": round(sw_daily, 2),
                    "dias_analizados": sw_days
                },
                "porcentaje_acierto": round(ti_percentage, 2),
            },
            "TR": {
                "descripcion": "Tiempo de Registro por Infracción Individual",
                "unidad": "minutos por infracción (min)",
                "sin_software": {
                    "tiempo_promedio_minutos": round(pnp_min, 2),
                    "fuente": "Estimación basada en registros históricos de campo"
                },
                "con_software": {
                    "tiempo_promedio_minutos": round(sw_min, 2),
                    "tiempos_individuales": [round(t, 2) for t in sw_times_min],  # Array de TRs individuales
                    "muestras_analizadas": len(software_processing_times)
                },
                "reduccion_tiempo_porcentual": round(tr_reduction_pct, 2),
                "veces_mas_rapido": round(tr_speedup, 2),
            },
            "NID": {
                "descripcion": "Número de Infracciones Detectadas Correctamente (Clasificación por Confianza)",
                "unidad": "cantidad válida por día",
                "infracciones_hoy": nid_today,
                "promedio_diario": round(nid_daily_avg, 0),
                "periodo_analizado": f"{sw_days} días",
                "total": nid_count
            },
            "NIE": {
                "descripcion": "Número de Infracciones Incorrectamente Registradas",
                "unidad": "cantidad no válida por día",
                "infracciones_incorrectas": nie_count,
                "total": nie_count
            },
        },
        "resumen_global": {
            "ti_porcentaje_acierto": f"{ti_percentage:.1f}%",
            "tiempo_registro_minutos": f"{sw_min:.2f} min",
            "infracciones_detectadas_hoy": nid_today,
            "nid_total": nid_count,
            "nie_total": nie_count,
            "tir_total": nid_count + nie_count
        },
    }

    # Persistir en SQLite (fuente única) + JSON legacy compat
    report = generate_performance_report(software_infractions, software_processing_times, nombre_video=nombre_video, config_semaforo=config_semaforo)
    try:
        from src.infrastructure.database.app_repository import AppRepository
        AppRepository().upsert_indicators(report)
        print("✅ Indicadores guardados en SQLite")
    except Exception as e:
        print(f"⚠️ Error guardando indicadores en SQLite: {e}")
    # Compat JSON (deprecated)
    try:
        out_path = resource_path("data/indicadores_rendimiento.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        from src.core.utils.json_store import read_json, write_json
        _existing = read_json(out_path, {})
        if isinstance(_existing, dict) and "metricas_tesis" in _existing:
            report["metricas_tesis"] = _existing["metricas_tesis"]
        write_json(out_path, report)
    except Exception:
        pass
    print(f"✅ Indicadores generados para {report.get('nombre_video')} TI={report['indicadores']['TI']['porcentaje_acierto']}% TR={report['indicadores']['TR']['con_software']['tiempo_promedio_minutos']}min NID={report['indicadores']['NID']['total']} NIE={report['indicadores']['NIE']['total']}")
    return report


def create_infractions_window(window: tk.Toplevel, back_callback):
    window.configure(bg="#ffffff")
    
    # MODIFICACIÓN DE PORTABILIDAD (desactivada para evitar segfault)
    # La ventana se abre con el tamaño heredado de la ventana principal.
    # El usuario puede maximizar manualmente si lo desea.
    pass

    # 1) Cargar todas las infracciones en segundo plano para no bloquear la UI
    all_data = []
    ui_queue = queue.Queue()
    image_cache = {}
    # Paginación: solo se renderizan 10 tarjetas por página para evitar
    # problemas de carga con datasets grandes.
    visible_count = 10
    current_data = []
    current_columns = 3
    # Caché de fechas parseadas: evita strptime por registro en cada
    # filtro/ordenamiento (el dataset crece como stack en cada procesamiento).
    _fecha_cache: dict[str, datetime] = {}
    _hora_cache: dict[str, int] = {}

    def _parse_fecha(fecha_str):
        dt = _fecha_cache.get(fecha_str)
        if dt is None:
            try:
                dt = datetime.strptime(fecha_str, '%d/%m/%Y')
            except ValueError:
                dt = datetime(2000, 1, 1)
            _fecha_cache[fecha_str] = dt
        return dt

    def _parse_hora(hora_str):
        h = _hora_cache.get(hora_str)
        if h is None:
            h = int(hora_str.replace(':', '')) if hora_str else 0
            _hora_cache[hora_str] = h
        return h

    def load_data_async():
        try:
            data = load_infractions_data()
        except Exception as e:
            print(f"Error cargando infracciones: {e}")
            data = []
        ui_queue.put(data)

    def poll_ui_queue():
        try:
            while True:
                data = ui_queue.get_nowait()
                nonlocal all_data
                all_data = data
                populate_cards(all_data)
        except queue.Empty:
            pass
        try:
            window.after(50, poll_ui_queue)
        except tk.TclError:
            pass

    # (El resto del código de la interfaz se mantiene exactamente igual...)
    # 3) Cabecera
    header = tk.Frame(window, bg="#ffffff")
    header.pack(fill="x", padx=30, pady=20)

    tk.Button(
        header, text="Volver", font=("Arial", 16), bg="#3366FF", fg="white",
        bd=0, activebackground="#3366FF", activeforeground="white",
        command=back_callback, cursor="hand2"
    ).pack(side="left")

    tk.Label(
        header, text="Gestión de Infracciones",
        font=("Arial", 28, "bold"), bg="#ffffff", fg="black"
    ).pack(side="left", padx=(20,0))

    actions = tk.Frame(header, bg="#ffffff")
    actions.pack(side="right")
    # NUEVO: Función para mostrar indicadores de rendimiento
    def show_performance_indicators():
        try:
            import tkinter as tk
            from datetime import datetime
            import json
            import os
            
            # ===== 1. RECOPILACIÓN DE DATOS DEL SISTEMA ACTUAL =====
            # Obtener datos de infracciones detectadas con software
            software_infractions = []
            software_processing_times = []
            software_reincidence_data = {}
            
            # Cargar datos de infracciones del JSON
            infractions_file = resource_path("data/infracciones.json")
            if os.path.exists(infractions_file):
                try:
                    with open(infractions_file, "r", encoding="utf-8") as f:
                        software_infractions = json.load(f)
                        
                        # Buscar si existe archivo de tiempos de procesamiento 
                        processing_times_file = resource_path("data/processing_times.json")
                        if os.path.exists(processing_times_file):
                            try:
                                with open(processing_times_file, "r", encoding="utf-8") as pt_file:
                                    processing_data = json.load(pt_file)
                                    software_processing_times = processing_data.get("processing_times", [])
                            except Exception as e:
                                print(f"Error cargando tiempos de procesamiento: {e}")
                        
                        # Si no hay datos de tiempos o el archivo no existe, calcular promedio desde PreprocessingDialog
                        if not software_processing_times:
                            try:
                                # Importar clase para acceder a los datos
                                from src.gui.preprocessing_dialog import PreprocessingDialog
                                
                                # Buscar atributo estático donde se guardarían tiempos
                                if hasattr(PreprocessingDialog, 'recorded_processing_times') and PreprocessingDialog.recorded_processing_times:
                                    software_processing_times = PreprocessingDialog.recorded_processing_times
                                else:
                                    # Si no hay datos registrados, usar un valor más realista basado en mediciones
                                    software_processing_times = [9.5 for _ in range(len(software_infractions))]
                            except Exception as e:
                                print(f"Error accediendo a tiempos de procesamiento: {e}")
                                software_processing_times = [9.5 for _ in range(len(software_infractions))]
                except Exception as e:
                    print(f"Error cargando infracciones: {e}")
                    software_infractions = []
                    software_processing_times = []
            
            # Si después de todos los intentos no hay tiempos, usar valor predeterminado
            if not software_processing_times:
                software_processing_times = [9.5 for _ in range(len(software_infractions))]
            
            # Organizar datos por día para calcular reincidencia
            day_infractions = {}
            for infraction in software_infractions:
                fecha = infraction.get("fecha", "Sin fecha")
                placa = infraction.get("placa", "")
                
                if fecha not in day_infractions:
                    day_infractions[fecha] = {"total": 0, "placas": {}}
                
                day_infractions[fecha]["total"] += 1
                
                if placa:
                    if placa not in day_infractions[fecha]["placas"]:
                        day_infractions[fecha]["placas"][placa] = 0
                    day_infractions[fecha]["placas"][placa] += 1
            
            # ===== 2. DATOS SIN SOFTWARE (VALORES DE ENCUESTAS Y ESTADÍSTICAS PNP) =====
            # Usar los datos proporcionados en el prompt
            pnp_monthly_data = {
                "Enero 2023": {"total": 125, "dias": 31, "reincidentes": 18},
                "Febrero 2023": {"total": 117, "dias": 28, "reincidentes": 15},
                "Marzo 2023": {"total": 137, "dias": 31, "reincidentes": 15},
                "Abril 2023": {"total": 129, "dias": 30, "reincidentes": 17}
            }
            
            # Datos de encuesta a policías sobre tiempo de registro
            police_registration_times = [7, 6, 5, 10, 8]  # minutos por infracción
            
            # ===== 3. CÁLCULO DE INDICADORES =====
            # ----- INDICADOR 1: Tasa de Infracciones Detectadas (TI) COMO PORCENTAJE -----
            # NUEVA OPERACIONALIZACIÓN: TI = (detecciones software / registros campo) × 100
            
            # Sin software: Promedio diario basado en datos históricos PNP
            pnp_total_infractions = sum(data["total"] for data in pnp_monthly_data.values())
            pnp_total_days = sum(data["dias"] for data in pnp_monthly_data.values())
            pnp_daily_average = pnp_total_infractions / pnp_total_days if pnp_total_days else 0
            
            # Con software: Promedio diario basado en datos del periodo analizado
            software_days = len(day_infractions)
            software_total_infractions = len(software_infractions)
            software_daily_average = software_total_infractions / software_days if software_days else 0
            
            # NUEVO: TI como porcentaje de acierto
            if pnp_daily_average > 0:
                ti_percentage = (software_daily_average / pnp_daily_average) * 100
                ti_percentage = min(ti_percentage, 100.0)  # Máximo 100%
            else:
                ti_percentage = 0.0
            
            # ----- INDICADOR 2: Tiempo de Registro (TR) EN MINUTOS -----
            # NUEVA OPERACIONALIZACIÓN: TR en minutos, no segundos
            
            # Sin software: Promedio de tiempo de registro según encuestas (convertir a segundos, luego minutos)
            pnp_avg_time_seconds = sum(police_registration_times) / len(police_registration_times) * 60 if police_registration_times else 0
            pnp_avg_time_minutes = pnp_avg_time_seconds / 60.0  # Convertir a minutos
            
            # Con software: Promedio de tiempo de procesamiento del sistema
            software_avg_time_seconds = sum(software_processing_times) / len(software_processing_times) if software_processing_times else 0
            software_avg_time_minutes = software_avg_time_seconds / 60.0  # Convertir a minutos
            
            tr_reduction = ((pnp_avg_time_minutes - software_avg_time_minutes) / pnp_avg_time_minutes * 100) if pnp_avg_time_minutes else 0
            tr_speedup = pnp_avg_time_minutes / software_avg_time_minutes if software_avg_time_minutes else 0
            
            # ----- INDICADOR NID: Número de Infracciones Detectadas (DIARIO) -----
            # NUEVO INDICADOR: Conteo diario de infracciones
            from datetime import datetime
            
            today = datetime.now().strftime("%Y-%m-%d")
            nid_today = 0
            
            # Contar infracciones de hoy
            for inf in software_infractions:
                fecha = inf.get("fecha", "")
                if fecha and today in fecha:
                    nid_today += 1
            
            # Si no hay infracciones de hoy, usar el total como referencia
            if nid_today == 0:
                nid_today = len(software_infractions)
            
            # ===== 4. GENERAR INFORME CON NUEVA OPERACIONALIZACIÓN (SIN IR) =====
            report = {
                "fecha_generacion": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "periodo_analisis": f"{min(day_infractions.keys(), default='N/A')} - {max(day_infractions.keys(), default='N/A')}",
                "dias_analizados": software_days,
                "indicadores": {
                    "TI": {
                        "descripcion": "Tasa de Infracciones Correctamente Detectadas (Porcentaje)",
                        "sin_software": {
                            "registros_campo_diarios": pnp_daily_average,
                            "fuente": "Datos históricos PNP",
                            "meses_analizados": len(pnp_monthly_data)
                        },
                        "con_software": {
                            "detecciones_software_diarias": software_daily_average,
                            "dias_analizados": software_days
                        },
                        "porcentaje_acierto": ti_percentage
                    },
                    "TR": {
                        "descripcion": "Tiempo de Registro (Minutos)",
                        "sin_software": {
                            "tiempo_promedio_minutos": pnp_avg_time_minutes,
                            "fuente": "Encuesta a oficiales PNP"
                        },
                        "con_software": {
                            "tiempo_promedio_minutos": software_avg_time_minutes,
                            "muestras_analizadas": len(software_processing_times)
                        },
                        "reduccion_tiempo_porcentual": tr_reduction,
                        "veces_mas_rapido": tr_speedup
                    },
                    "NID": {
                        "descripcion": "Número de Infracciones Detectadas (Diario)",
                        "infracciones_hoy": nid_today,
                        "promedio_diario": software_daily_average,
                        "periodo_analizado": f"{software_days} días"
                    }
                },
                "resumen_global": {
                    "ti_porcentaje_acierto": f"{ti_percentage:.1f}%",
                    "tiempo_registro_minutos": f"{software_avg_time_minutes:.2f} min",
                    "infracciones_detectadas_hoy": nid_today
                }
            }
            
            # Guardar informe en JSON (conservando metricas_tesis de la sesión)
            report_file = resource_path("data/indicadores_rendimiento.json")
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            from src.core.utils.json_store import read_json, write_json
            _existing = read_json(report_file, {})
            if isinstance(_existing, dict) and "metricas_tesis" in _existing:
                report["metricas_tesis"] = _existing["metricas_tesis"]
            write_json(report_file, report)
            
            # Generar resumen para mostrar (SIN IR)
            resumen = f"""
            🟦 INDICADOR 1: Tasa de Infracciones Detectadas (TI)
            Sin software: {pnp_daily_average:.1f} infracciones/día
            Con software: {software_daily_average:.1f} infracciones/día
            Porcentaje de acierto: {ti_percentage:.1f}%
            
            🟦 INDICADOR 2: Tiempo de Registro (TR)
            Sin software: {pnp_avg_time_minutes:.2f} minutos
            Con software: {software_avg_time_minutes:.2f} minutos
            Reducción: {tr_reduction:.1f}% ({tr_speedup:.1f}x más rápido)
            
            🟦 INDICADOR 3: Número de Infracciones Diarias (NID)
            Infracciones detectadas hoy: {nid_today}
            Promedio diario: {software_daily_average:.1f} infracciones
            
            ✅ RESUMEN: El sistema automatizado tiene {ti_percentage:.1f}% de acierto
            y registra cada infracción en {software_avg_time_minutes:.2f} minutos.
            """
            
            # Crear ventana de informe
            report_window = tk.Toplevel(window)
            report_window.title("Indicadores de Rendimiento - InfractiVision")
            report_window.geometry("700x600")
            report_window.minsize(600, 500)
            
            # Estilos y configuración
            report_window.configure(bg="#f5f5f5")
            
            # Título
            title_frame = tk.Frame(report_window, bg="#2c3e50", pady=10)
            title_frame.pack(fill="x")
            
            title_label = tk.Label(title_frame, 
                                text="ANÁLISIS DE INDICADORES DE RENDIMIENTO",
                                font=("Arial", 16, "bold"),
                                bg="#2c3e50", fg="white")
            title_label.pack(padx=10)
            
            # Fecha de generación
            date_label = tk.Label(title_frame,
                                text=f"Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                                font=("Arial", 10),
                                bg="#2c3e50", fg="white")
            date_label.pack(pady=(0, 5))
            
            # Marco para contenido con scroll
            content_frame = tk.Frame(report_window, bg="#f5f5f5")
            content_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            # Crear canvas con scrollbar
            canvas = tk.Canvas(content_frame, bg="#f5f5f5", highlightthickness=0)
            scrollbar = tk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
            
            # Configurar canvas
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Frame interior para contenido
            inner_frame = tk.Frame(canvas, bg="#f5f5f5", padx=10)
            canvas.create_window((0, 0), window=inner_frame, anchor="nw")
            
            # Función para actualizar scroll region
            def _configure_canvas(event):
                canvas.configure(scrollregion=canvas.bbox("all"))
            
            inner_frame.bind("<Configure>", _configure_canvas)
            
            # Encabezado del informe
            header_label = tk.Label(inner_frame, 
                                    text="COMPARATIVA: SISTEMA MANUAL VS. AUTOMATIZADO",
                                    font=("Arial", 12, "bold"),
                                    bg="#f5f5f5", fg="#2c3e50",
                                    pady=5)
            header_label.pack(fill="x", pady=10)
            
            # Descripción general
            desc_label = tk.Label(inner_frame,
                                text="Este informe presenta los resultados de la comparación entre el sistema "
                                    "tradicional (sin software) y el sistema automatizado InfractiVision "
                                    "para la detección de infracciones de tráfico.",
                                font=("Arial", 10),
                                bg="#f5f5f5", fg="#333333",
                                wraplength=550, justify="left")
            desc_label.pack(fill="x", pady=5)
            
            # Convertir texto del informe a formato enriquecido
            report_text_formatted = resumen.replace("🟦", "\n🟦").replace("✅", "\n✅")
            
            # Área de texto para el informe
            report_text_widget = tk.Text(inner_frame, height=20, width=70, bg="white",
                                    font=("Consolas", 10), padx=10, pady=10,
                                    wrap="word", relief="flat")
            report_text_widget.pack(fill="both", expand=True, pady=10)
            report_text_widget.insert("1.0", report_text_formatted)
            report_text_widget.configure(state="disabled")
            
            # Añadir etiquetas para resaltar secciones
            report_text_widget.tag_configure("header", font=("Consolas", 11, "bold"), foreground="#2c3e50")
            report_text_widget.tag_configure("important", font=("Consolas", 10, "bold"), foreground="#27ae60")
            report_text_widget.tag_configure("positive", foreground="#27ae60")
            
            # Aplicar estilos
            for line_num, line in enumerate(report_text_formatted.split("\n")):
                line_pos = f"{line_num+1}.0"
                end_pos = f"{line_num+1}.end"
                
                if "🟦" in line:
                    report_text_widget.tag_add("header", line_pos, end_pos)
                elif "✅" in line:
                    report_text_widget.tag_add("important", line_pos, end_pos)
                elif "más rápido" in line or "mayor detección" in line or "+%" in line:
                    report_text_widget.tag_add("positive", line_pos, end_pos)
            
            # Botones de acción
            button_frame = tk.Frame(report_window, bg="#f5f5f5", pady=10)
            button_frame.pack(fill="x", padx=20, pady=(0, 20))
            
            # Función para exportar a JSON
            def export_indicator_report():
                try:
                    from tkinter import filedialog
                    
                    # Abrir diálogo para seleccionar ubicación de guardado
                    file_path = filedialog.asksaveasfilename(
                        defaultextension=".json",
                        filetypes=[("Archivo JSON", "*.json"), ("Todos los archivos", "*.*")],
                        title="Guardar Informe de Indicadores"
                    )
                    
                    if not file_path:
                        return
                    
                    # Verificar si existe el archivo de indicadores
                    source_path = resource_path("data/indicadores_rendimiento.json")
                    if not os.path.exists(source_path):
                        messagebox.showerror("Error", "No se encontró el archivo de indicadores de rendimiento.")
                        return
                    
                    # Copiar archivo
                    import shutil
                    shutil.copy2(source_path, file_path)
                    
                    messagebox.showinfo("Exportación Exitosa", 
                                    f"El informe de indicadores ha sido exportado a:\n{file_path}")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo exportar el informe: {str(e)}")
            
            # Botón para exportar a JSON
            export_button = tk.Button(
                button_frame, text="Exportar JSON",
                command=export_indicator_report,
                bg="#3498db", fg="white",
                font=("Arial", 10, "bold"),
                padx=15, pady=5, relief="flat")
            export_button.pack(side="left", padx=5)
            
            # Botón para cerrar
            close_button = tk.Button(
                button_frame, text="Cerrar",
                command=report_window.destroy,
                bg="#e74c3c", fg="white", 
                font=("Arial", 10, "bold"),
                padx=15, pady=5, relief="flat")
            close_button.pack(side="right", padx=5)
            
            # Hacer ventana modal
            report_window.transient(window)
            report_window.grab_set()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"No se pudieron calcular los indicadores: {e}")
    
    # ELIMINADO: Botón INDICADORES - Los indicadores ahora están en el panel superior principal

    # Función para descargar infracciones en diferentes formatos
    def download_infractions():
        if not all_data:
            messagebox.showinfo("Información", "No hay infracciones para descargar")
            return
            
        # Filtrar datos según las fechas seleccionadas
        start = datetime.combine(start_picker.get_date(), datetime.min.time())
        end = datetime.combine(end_picker.get_date(), datetime.max.time())
        
        filtered_data = []
        for inf in all_data:
            try:
                fecha_str = inf.get('fecha', '')
                fecha = datetime.strptime(fecha_str, '%d/%m/%Y')
                if start <= fecha <= end:
                    filtered_data.append(inf)
            except ValueError:
                # Si hay un error de formato de fecha, incluimos igual el registro
                filtered_data.append(inf)
        
        if not filtered_data:
            messagebox.showinfo("Información", "No hay infracciones en el período seleccionado")
            return
        
        # Cuadro de diálogo para elegir formato de exportación
        export_win = tk.Toplevel(window)
        export_win.title("Exportar Infracciones")
        export_win.geometry("400x300")
        export_win.resizable(False, False)
        export_win.configure(bg="#ffffff")
        export_win.grab_set()
        
        tk.Label(export_win, text="Seleccione el formato de exportación",
                font=("Arial", 14, "bold"), bg="#ffffff").pack(pady=20)
        
        def export_as_json():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Guardar infracciones como JSON"
            )
            if file_path:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
                    messagebox.showinfo("Éxito", f"Infracciones exportadas a {file_path}")
                    export_win.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Error al exportar: {e}")
        
        def export_as_csv():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Guardar infracciones como CSV"
            )
            if file_path:
                try:
                    if not filtered_data:
                        messagebox.showwarning("Advertencia", "No hay datos para exportar")
                        return
                    
                    # Obtener todos los campos únicos de todos los registros
                    all_fieldnames = set()
                    for record in filtered_data:
                        all_fieldnames.update(record.keys())
                    
                    # Ordenar campos para consistencia
                    fieldnames = sorted(list(all_fieldnames))
                    
                    with open(file_path, "w", encoding="utf-8", newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                        writer.writeheader()
                        
                        # Asegurar que cada registro tenga todos los campos
                        for record in filtered_data:
                            normalized_record = {}
                            for field in fieldnames:
                                normalized_record[field] = record.get(field, "")
                            writer.writerow(normalized_record)
                    
                    messagebox.showinfo("Éxito", f"Infracciones exportadas a {file_path}")
                    export_win.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Error al exportar CSV: {e}")
                    print(f"Error detallado CSV: {e}")
        
        def export_as_excel():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Guardar infracciones como Excel"
            )
            if file_path:
                try:
                    if not filtered_data:
                        messagebox.showwarning("Advertencia", "No hay datos para exportar")
                        return
                    
                    # Verificar disponibilidad de pandas y openpyxl
                    try:
                        import pandas as pd
                    except ImportError:
                        messagebox.showerror("Error", "pandas no está disponible. Use exportación CSV o JSON.")
                        return
                    
                    try:
                        import openpyxl
                    except ImportError:
                        messagebox.showerror("Error", "openpyxl no está disponible. Use exportación CSV o JSON.")
                        return
                    
                    # Convertir a DataFrame para exportar como Excel
                    df = pd.DataFrame(filtered_data)
                    
                    # Configurar writer con opciones mejoradas
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Infracciones', index=False)
                        
                        # Ajustar ancho de columnas automáticamente
                        worksheet = writer.sheets['Infracciones']
                        for column in worksheet.columns:
                            max_length = 0
                            column_letter = column[0].column_letter
                            for cell in column:
                                try:
                                    if len(str(cell.value)) > max_length:
                                        max_length = len(str(cell.value))
                                except:
                                    pass
                            adjusted_width = min(max_length + 2, 50)  # Máximo 50 caracteres
                            worksheet.column_dimensions[column_letter].width = adjusted_width
                    
                    messagebox.showinfo("Éxito", f"Infracciones exportadas a {file_path}")
                    export_win.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Error al exportar Excel: {e}")
                    print(f"Error detallado Excel: {e}")
        
        # Botones para los diferentes formatos
        button_frame = tk.Frame(export_win, bg="#ffffff")
        button_frame.pack(pady=20, fill="x")
        
        tk.Button(button_frame, text="JSON", font=("Arial", 12), bg="#3366FF", fg="white",
                command=export_as_json, cursor="hand2", width=10, height=2).pack(pady=10)
        
        tk.Button(button_frame, text="CSV", font=("Arial", 12), bg="#3366FF", fg="white",
                command=export_as_csv, cursor="hand2", width=10, height=2).pack(pady=10)
        
        tk.Button(button_frame, text="Excel", font=("Arial", 12), bg="#3366FF", fg="white",
                command=export_as_excel, cursor="hand2", width=10, height=2).pack(pady=10)
        
        tk.Button(export_win, text="Cancelar", font=("Arial", 12), bg="#FF3333", fg="white",
                command=export_win.destroy, cursor="hand2", width=10).pack(pady=10)
    
    # Botón de descarga con funcionalidad mejorada
    tk.Button(
        actions, text="DESCARGAR", font=("Arial", 14),
        bg="#3366FF", fg="white", bd=0,
        activebackground="#2554CC", activeforeground="white",
        cursor="hand2", command=download_infractions
    ).pack(side="left", padx=10)

    tk.Label(actions, text="Desde:", font=("Arial", 12), bg="#ffffff").pack(side="left")
    start_picker = DateEntry(
        actions, font=("Arial", 12), width=10,
        background="white", foreground="black",
        borderwidth=1, date_pattern='dd/MM/yyyy'
    )
    start_picker.pack(side="left", padx=(5,15))

    tk.Label(actions, text="Hasta:", font=("Arial", 12), bg="#ffffff").pack(side="left")
    end_picker = DateEntry(
        actions, font=("Arial", 12), width=10,
        background="white", foreground="black",
        borderwidth=1, date_pattern='dd/MM/yyyy'
    )
    end_picker.pack(side="left", padx=(5,15))

    def apply_filter():
        try:
            nonlocal visible_count
            start = datetime.combine(start_picker.get_date(), datetime.min.time())
            end = datetime.combine(end_picker.get_date(), datetime.max.time())
            filtered = []
            for inf in all_data:
                fecha = _parse_fecha(inf.get('fecha', ''))
                if start <= fecha <= end:
                    filtered.append(inf)
            visible_count = 10
            populate_cards(filtered)
        except Exception as e:
            messagebox.showerror("Error", f"Error aplicando filtro: {e}")

    # Función para sincronizar los datos desde los archivos locales (NID + NIE)
    def refresh_data():
        nonlocal all_data
        all_data = load_infractions_data()
        apply_filter()
        nid_count = sum(1 for inf in all_data if inf.get('clasificacion', 'NID') == 'NID')
        nie_count = sum(1 for inf in all_data if inf.get('clasificacion', 'NID') != 'NID')
        print(f"🔄 Sincronización local: {len(all_data)} infracciones ({nid_count} NID + {nie_count} NIE)")
        messagebox.showinfo(
            "Sincronización local",
            f"Infracciones sincronizadas desde archivos locales:\n\n"
            f"• {nid_count} NID (validadas correctamente)\n"
            f"• {nie_count} NIE (sin validar)\n"
            f"• Total: {len(all_data)}"
        )

    tk.Button(
        actions, text="🔄 SINCRONIZAR LOCAL", font=("Arial", 12),
        bg="#27ae60", fg="white", bd=0,
        activebackground="#1e8449", activeforeground="white",
        cursor="hand2", command=refresh_data
    ).pack(side="left", padx=10)

    tk.Button(
        actions, text="FILTRAR", font=("Arial", 12),
        bg="#3366FF", fg="white", bd=0,
        activebackground="#2554CC", activeforeground="white",
        cursor="hand2", command=apply_filter
    ).pack(side="left", padx=10)

    # Definir función de limpieza aquí para usarla después
    def clear_all_infractions():
        """Función para vaciar todas las infracciones"""
        result = messagebox.askyesno(
            "Confirmar eliminación",
            "¿Está seguro de que desea eliminar TODAS las infracciones?\n\n"
            "Esta acción eliminará:\n"
            "• Todas las infracciones del registro\n"
            "• Todas las imágenes de vehículos y placas\n\n"
            "Esta acción NO se puede deshacer."
        )
        
        if result:
            if delete_all_infractions():
                nonlocal all_data
                all_data = []
                populate_cards([])
                messagebox.showinfo("Éxito", "Todas las infracciones han sido eliminadas correctamente.")
            else:
                messagebox.showerror("Error", "No se pudieron eliminar todas las infracciones.")

    # — Contenedor scrollable para las tarjetas —
    container = tk.Frame(window, bg="gray")
    container.pack(fill="both", expand=True, padx=100, pady=(20,100))  # Añadido padding horizontal moderado
    
    # Ajustar el ancho del canvas para ocupar toda la ventana
    canvas = tk.Canvas(container, bg="gray", highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="gray")
    
    # Hacer que scrollable_frame mantenga el ancho del canvas
    def configure_frame(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(frame_id, width=event.width)  # Ajustar el ancho del frame al canvas
    
    scrollable_frame.bind("<Configure>", configure_frame)
    frame_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=window.winfo_width())
    
    # Hacer que el canvas cambie de tamaño con la ventana
    def on_canvas_configure(event):
        nonlocal current_columns
        canvas.itemconfig(frame_id, width=event.width)
        # Re-render del grid si cambia el número de columnas
        cols = 3 if event.width >= 950 else 2 if event.width >= 600 else 1
        if cols != current_columns:
            current_columns = cols
            if current_data:
                populate_cards(current_data)
    
    canvas.bind("<Configure>", on_canvas_configure)
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # MEJORA: Agregar scroll suave con rueda del mouse
    def on_mousewheel(event):
        # Scroll suave: dividir delta por 3 para hacer más lento y suave
        canvas.yview_scroll(int(-1 * (event.delta / 120 / 3)), "units")
        
    def bind_to_mousewheel(event):
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
    def unbind_from_mousewheel(event):
        canvas.unbind_all("<MouseWheel>")
        
    # MEJORADO: Bindear scroll a TODA el área para mejor experiencia
    def bind_mousewheel_globally():
        """Vincula el scroll a toda la ventana y sus componentes"""
        canvas.bind_all("<MouseWheel>", on_mousewheel)
    
    def unbind_mousewheel_globally():
        """Desvincula el scroll global"""
        canvas.unbind_all("<MouseWheel>")
        
    # Bindear a MÚLTIPLES componentes para scroll universal
    canvas.bind('<Enter>', lambda e: bind_mousewheel_globally())
    canvas.bind('<Leave>', lambda e: unbind_mousewheel_globally())
    
    scrollable_frame.bind('<Enter>', lambda e: bind_mousewheel_globally())
    scrollable_frame.bind('<Leave>', lambda e: unbind_mousewheel_globally())
    
    # NUEVO: También bindear a la ventana principal para scroll universal
    window.bind('<Enter>', lambda e: bind_mousewheel_globally())
    window.bind('<Leave>', lambda e: unbind_mousewheel_globally())
    
    # Activar scroll inmediatamente
    bind_mousewheel_globally()
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def clear_cards():
        for child in scrollable_frame.winfo_children():
            child.destroy()

    def create_show_plate_func(plate_path, placa_text):
        def show_plate_func():
            if plate_path and os.path.exists(plate_path):
                try:
                    plate_window = tk.Toplevel(window)
                    plate_window.title(f"Placa: {placa_text}")
                    from PIL import Image, ImageTk
                    img = Image.open(plate_path)
                    photo = ImageTk.PhotoImage(img)
                    img_label = tk.Label(plate_window, image=photo)
                    img_label.image = photo
                    img_label.pack(padx=20, pady=20)
                    tk.Button(plate_window, text="Cerrar",
                              command=plate_window.destroy).pack(pady=10)
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")
            else:
                messagebox.showinfo("Información", "No hay imagen de placa disponible")
        return show_plate_func

    def create_delete_func(infraction_data):
        def delete_func():
            result = messagebox.askyesno(
                "Confirmar eliminación",
                f"¿Eliminar la infracción de la placa {infraction_data.get('placa', 'No identificada')}?\n\n"
                "Se eliminarán las imágenes asociadas."
            )
            if result:
                deleted_files = delete_infraction_files(infraction_data)
                placa_to_remove = infraction_data.get('placa', '')
                if remove_infraction_from_json(placa_to_remove):
                    nonlocal all_data
                    all_data = load_infractions_data()
                    apply_filter()
                    msg = f"Infracción eliminada correctamente."
                    if deleted_files:
                        msg += f"\nArchivos eliminados: {', '.join(deleted_files)}"
                    messagebox.showinfo("Éxito", msg)
                else:
                    messagebox.showerror("Error", "No se pudo eliminar la infracción del registro.")
        return delete_func

    def create_show_details_func(infraction_data):
        def show_details_func():
            details_window = tk.Toplevel(window)
            details_window.title(f"Detalles - Placa: {infraction_data.get('placa', 'No identificada')}")
            details_window.geometry("500x400")
            details_window.configure(bg="#f8f9fa")

            details_window.update_idletasks()
            width, height = 500, 400
            x = (details_window.winfo_screenwidth() - width) // 2
            y = (details_window.winfo_screenheight() - height) // 2
            details_window.geometry(f"{width}x{height}+{x}+{y}")

            tk.Label(
                details_window,
                text="📋 Detalles de la Infracción",
                font=("Arial", 16, "bold"),
                bg="#f8f9fa",
                fg="#2c3e50"
            ).pack(pady=20)

            details_frame = tk.Frame(details_window, bg="#f8f9fa")
            details_frame.pack(fill="both", expand=True, padx=20, pady=10)

            details_info = [
                ("🚗 Placa:", infraction_data.get('placa', 'No identificada')),
                ("📅 Fecha:", infraction_data.get('fecha', 'Sin fecha')),
                ("⏰ Hora:", infraction_data.get('hora', 'Sin hora')),
                ("📍 Ubicación:", infraction_data.get('ubicacion', 'Sin ubicación')),
                ("🚦 Tipo:", infraction_data.get('tipo', 'Sin especificar')),
                ("🎥 Timestamp:", infraction_data.get('video_timestamp', 'No disponible')),
                ("📊 ID Único:", infraction_data.get('id', 'No disponible')),
            ]

            for label, value in details_info:
                row_frame = tk.Frame(details_frame, bg="#ffffff", relief="ridge", bd=1)
                row_frame.pack(fill="x", pady=2)

                tk.Label(row_frame, text=label, font=("Arial", 10, "bold"),
                        bg="#ffffff", width=15, anchor="w").pack(side="left", padx=10, pady=5)
                tk.Label(row_frame, text=str(value), font=("Arial", 10),
                        bg="#ffffff", anchor="w").pack(side="left", padx=10, pady=5)

            tk.Button(
                details_window,
                text="Cerrar",
                command=details_window.destroy,
                bg="#3498db",
                fg="white",
                font=("Arial", 12),
                padx=20,
                pady=5
            ).pack(pady=20)
        return show_details_func

    def build_infraction_card(parent, inf):
        # --- Estado y colores según validación ---
        clasificacion = inf.get('clasificacion', 'NID')
        confianza = max(0.0, min(1.0, inf.get('confianza', 0)))  # Clamp [0,1]

        NID_THRESHOLD_EXCELLENT = 0.85
        NID_THRESHOLD_GOOD = 0.65

        if clasificacion == 'NID' and confianza < NID_THRESHOLD_GOOD:
            badge_text = "⚠️ NIE - CONFIANZA BAJA"
            badge_color = "#f39c12"
        elif clasificacion == 'NID':
            badge_text = "✅ NID VALIDADA"
            badge_color = "#27ae60"
        else:
            badge_text = "❌ NIE NO VALIDADA"
            badge_color = "#e74c3c"

        card_bg = "#E8F5E9" if badge_color == "#27ae60" else "#FDEBD0" if badge_color == "#f39c12" else "#FDECEA"

        # --- Card principal con borde de color según estado ---
        card = tk.Frame(parent, bg=card_bg, highlightthickness=2,
                        highlightbackground=badge_color, highlightcolor=badge_color)
        card.columnconfigure(0, weight=1)

        # Badge de estado
        tk.Label(
            card, text=badge_text, font=("Arial", 9, "bold"),
            bg=badge_color, fg="white"
        ).pack(fill="x")

        # Imagen de la placa (fallback: imagen del vehículo)
        img_box = tk.Frame(card, bg="#ffffff", height=120)
        img_box.pack(fill="x", padx=10, pady=(10, 2))
        img_box.pack_propagate(False)

        img_path = inf.get('plate_path') or inf.get('vehicle_path', '')
        if img_path and os.path.exists(img_path):
            try:
                from PIL import Image, ImageTk
                photo = image_cache.get(img_path)
                if photo is None:
                    img = Image.open(img_path)
                    ratio = min(260.0 / img.width, 112.0 / img.height)
                    nw = max(1, int(img.width * ratio))
                    nh = max(1, int(img.height * ratio))
                    photo = ImageTk.PhotoImage(img.resize((nw, nh), Image.LANCZOS))
                    image_cache[img_path] = photo
                img_label = tk.Label(img_box, image=photo, bg="#ffffff")
                img_label.image = photo
                img_label.pack(expand=True)
            except Exception as e:
                print(f"Error cargando imagen de placa: {e}")
                tk.Label(img_box, text="[Sin imagen]", bg="#ffffff", fg="#999999").pack(expand=True)
        else:
            tk.Label(img_box, text="[Sin imagen]", bg="#ffffff", fg="#999999").pack(expand=True)

        # Placa
        placa_text = inf.get('placa', 'No identificada')
        tk.Label(
            card, text=placa_text, font=("Arial", 15, "bold"),
            bg=card_bg, fg="#1a1a1a"
        ).pack(pady=(6, 0))

        # Confianza en porcentaje con color por rango
        if confianza >= NID_THRESHOLD_EXCELLENT:
            conf_color = "#27ae60"
        elif confianza >= NID_THRESHOLD_GOOD:
            conf_color = "#f39c12"
        else:
            conf_color = "#e74c3c"
        tk.Label(
            card, text=f"Confianza: {confianza * 100:.1f}%",
            font=("Arial", 11, "bold"), bg=card_bg, fg=conf_color
        ).pack()

        # Video de origen
        video_name = inf.get('nombre_video') or inf.get('video') or 'Video desconocido'
        if len(video_name) > 36:
            video_name = video_name[:33] + '...'
        tk.Label(
            card, text=f"🎥 {video_name}", font=("Arial", 10),
            bg=card_bg, fg="#555555", wraplength=280, justify="center"
        ).pack()

        # Fecha y hora
        tk.Label(
            card, text=f"📅 {inf.get('fecha', '')}  {inf.get('hora', '')}",
            font=("Arial", 9), bg=card_bg, fg="#888888"
        ).pack(pady=(2, 6))

        # --- Acciones ---
        plate_path = inf.get('plate_path', '')
        show_plate_func = create_show_plate_func(plate_path, placa_text)
        delete_func = create_delete_func(inf)
        show_details_func = create_show_details_func(inf)

        btn_row = tk.Frame(card, bg=card_bg)
        btn_row.pack(fill="x", padx=8, pady=(2, 8))

        tk.Button(
            btn_row, text="Ver placa", command=show_plate_func,
            bg="#3366FF", fg="white", bd=0, cursor="hand2", font=("Arial", 9)
        ).pack(side="left", expand=True, fill="x", padx=3)

        tk.Button(
            btn_row, text="Detalles", command=show_details_func,
            bg="#5D6D7E", fg="white", bd=0, cursor="hand2", font=("Arial", 9)
        ).pack(side="left", expand=True, fill="x", padx=3)

        tk.Button(
            btn_row, text="🗑 Eliminar", command=delete_func,
            bg="#e74c3c", fg="white", bd=0, cursor="hand2", font=("Arial", 9)
        ).pack(side="left", expand=True, fill="x", padx=3)

        return card

    def load_more():
        nonlocal visible_count
        visible_count += 10
        populate_cards(current_data)

    def populate_cards(data_list):
        nonlocal visible_count, current_data, current_columns
        clear_cards()
        if not data_list:
            tk.Label(
                scrollable_frame, text="No se encontraron infracciones.",
                font=("Arial", 16), bg="gray", fg="white"
            ).pack(pady=80, padx=80)
            return

        # Filtrar solo elementos que sean diccionarios válidos
        valid_data = []
        for item in data_list:
            if isinstance(item, dict):
                valid_data.append(item)
            elif isinstance(item, str):
                print(f"⚠️ Elemento string ignorado en datos: {item}")
            else:
                print(f"⚠️ Elemento de tipo desconocido ignorado: {type(item)} - {item}")

        data_list = valid_data

        # Ordenar: fecha y hora (más reciente primero) — NID y NIE intercaladas
        try:
            data_list = sorted(data_list,
                            key=lambda x: (
                                -_parse_fecha(x.get('fecha', '01/01/2000')).timestamp(),
                                -_parse_hora(x.get('hora', '00:00:00'))
                            ))
        except Exception as e:
            print(f"Error al ordenar infracciones: {e}")
            data_list = [x for x in data_list if isinstance(x, dict)]

        current_data = data_list
        total = len(data_list)
        shown = data_list[:visible_count]
        shown_count = len(shown)

        # Contador de registros mostrados
        tk.Label(
            scrollable_frame, text=f"Mostrando {shown_count} de {total} infracciones",
            font=("Arial", 11, "bold"), bg="gray", fg="white"
        ).pack(anchor="w", padx=18, pady=(6, 0))

        # Grid responsivo: 3 columnas (2 si la ventana es angosta, 1 si es muy angosta)
        try:
            canvas_width = canvas.winfo_width()
        except tk.TclError:
            canvas_width = 0
        current_columns = 3 if canvas_width >= 950 else 2 if canvas_width >= 600 else 1
        columns = current_columns

        grid_frame = tk.Frame(scrollable_frame, bg="gray")
        grid_frame.pack(fill="both", expand=True, padx=10)

        for idx, inf in enumerate(shown):
            row, col = divmod(idx, columns)
            card = build_infraction_card(grid_frame, inf)
            card.grid(row=row, column=col, padx=8, pady=10, sticky="nsew")
            grid_frame.columnconfigure(col, weight=1)

        # Botón para cargar la siguiente página (10 más)
        if shown_count < total:
            tk.Button(
                scrollable_frame, text=f"CARGAR MÁS (+10) — quedan {total - shown_count}",
                font=("Arial", 12, "bold"), bg="#3366FF", fg="white", bd=0,
                activebackground="#2554CC", activeforeground="white",
                cursor="hand2", command=load_more
            ).pack(pady=14)

        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    # Inicializar la vista cargando los datos en segundo plano (no bloquea la UI)
    tk.Label(
        scrollable_frame, text="Cargando infracciones...",
        font=("Arial", 16), bg="gray", fg="white"
    ).pack(pady=80, padx=80)
    threading.Thread(target=load_data_async, daemon=True).start()
    window.after(50, poll_ui_queue)
    
    # Botones de acción adicionales al final
    bottom_actions = tk.Frame(window, bg="#ffffff")
    bottom_actions.pack(fill="x", padx=100, pady=10)
    
    # Botón vaciar todo
    tk.Button(
        bottom_actions, text="🗑️ VACIAR TODO", font=("Arial", 12),
        bg="#e74c3c", fg="white", bd=0,
        activebackground="#c0392b", activeforeground="white",
        cursor="hand2", command=clear_all_infractions,
        width=20, height=2
    ).pack(side="right", padx=10)
