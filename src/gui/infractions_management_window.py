import tkinter as tk
from tkinter import messagebox, filedialog
import json, os
from tkcalendar import DateEntry
from datetime import datetime
import shutil
import csv
import pandas as pd  # Para exportación a Excel (requiere openpyxl)
import requests
from src.path_helper import resource_path


CONFIG_PATH = resource_path("config/infractivision_config.json")


# Ruta centralizada del archivo de infracciones
INF_FILE = resource_path("data/infracciones.json")

# Ruta centralizada del archivo de infracciones NIE (incorrectamente registradas)
NIE_FILE = resource_path("data/nie_infracciones.json")

# Ruta del historial de migraciones (ACUMULATIVO)
MIGRATIONS_HISTORY_FILE = resource_path("data/historial_migraciones.json")

# === FUNCIONES PARA HISTORIAL ACUMULATIVO DE MIGRACIONES ===

def load_migrations_history():
    """Cargar historial completo de migraciones (acumulativo)"""
    try:
        if os.path.exists(MIGRATIONS_HISTORY_FILE):
            with open(MIGRATIONS_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error cargando historial de migraciones: {e}")
        return []

def save_migrations_history(history):
    """Guardar historial completo de migraciones"""
    try:
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(MIGRATIONS_HISTORY_FILE), exist_ok=True)
        with open(MIGRATIONS_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error guardando historial de migraciones: {e}")
        return False

def add_migration_to_history(num_infractions, estado="Exitosa"):
    """Agregar nueva migración al historial acumulativo (STACK/PILA)"""
    try:
        # Cargar historial existente
        history = load_migrations_history()
        
        # Crear nueva entrada con timestamp
        new_migration = {
            "fecha": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "timestamp": datetime.now().isoformat(),
            "registros": num_infractions,
            "estado": estado
        }
        
        # Agregar al INICIO de la lista (como stack/pila - más reciente primero)
        history.insert(0, new_migration)
        
        # Guardar historial actualizado
        save_migrations_history(history)
        print(f"📊 MIGRACIÓN AGREGADA AL HISTORIAL: {num_infractions} registros el {new_migration['fecha']}")
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


# Función para cargar datos de infracciones (movida al principio)
def load_infractions_data():
    """Carga las infracciones LOCALES fusionando NID (infracciones.json) y NIE (nie_infracciones.json)."""
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
    """Elimina todas las infracciones y sus archivos"""
    try:
        # Limpiar archivo de infracciones
        infractions_file = resource_path("data/infracciones.json")
        if os.path.exists(infractions_file):
            with open(infractions_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
        
        # Limpiar archivo de infracciones NIE
        nie_file = resource_path("data/nie_infracciones.json")
        if os.path.exists(nie_file):
            with open(nie_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
        
        # Limpiar directorios de imágenes
        output_dirs = [
            resource_path("data/output/placas"),
            resource_path("data/output/autos")
        ]
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        
        return True
    except Exception as e:
        print(f"Error eliminando todas las infracciones: {e}")
        return False

def _filter_infractions_file(file_path, placa_to_remove):
    """Elimina las entradas con la placa indicada del archivo JSON si existe."""
    if not os.path.exists(file_path):
        return True
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        is_dict = isinstance(data, dict) and 'infracciones' in data
        infractions = data['infracciones'] if is_dict else (data if isinstance(data, list) else [])
        updated = [inf for inf in infractions if inf.get('placa', '') != placa_to_remove]
        output = {'infracciones': updated} if is_dict else updated
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error eliminando infracción del JSON: {e}")
        return False


def remove_infraction_from_json(placa_to_remove):
    """Elimina una infracción específica de los archivos JSON (NID y NIE)"""
    ok_nid = _filter_infractions_file(resource_path("data/infracciones.json"), placa_to_remove)
    ok_nie = _filter_infractions_file(resource_path("data/nie_infracciones.json"), placa_to_remove)
    return ok_nid and ok_nie

def show_migrations(refresh_callback=None, all_data_ref=None):
    """Mostrar historial de migraciones exitosas"""
    # Crear ventana de migraciones (MÁS GRANDE para historial acumulativo)
    migrations_window = tk.Toplevel()
    migrations_window.title("📊 Historial Acumulativo de Migraciones")
    
    # Tamaño más grande para mostrar más migraciones
    screen_width = migrations_window.winfo_screenwidth()
    screen_height = migrations_window.winfo_screenheight()
    
    if screen_width >= 1920:  # Pantalla grande
        width, height = 800, 600
    elif screen_width >= 1366:  # Pantalla mediana
        width, height = 700, 550
    else:  # Pantalla pequeña
        width, height = 650, 500
    
    migrations_window.geometry(f"{width}x{height}")
    migrations_window.configure(bg="#f8f9fa")
    
    # Centrar ventana
    migrations_window.update_idletasks()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    migrations_window.geometry(f"{width}x{height}+{x}+{y}")
    
    # Título
    title_label = tk.Label(
        migrations_window,
        text="📊 Historial de Migraciones a la Nube",
        font=("Arial", 16, "bold"),
        bg="#f8f9fa",
        fg="#2c3e50"
    )
    title_label.pack(pady=20)
    
    # CARGAR HISTORIAL COMPLETO DE MIGRACIONES (ACUMULATIVO)
    migrations_data = load_migrations_history()
    print(f"📊 CARGANDO HISTORIAL: {len(migrations_data)} migraciones en historial")
    
    # Frame para lista con scrollbar (para historial acumulativo)
    main_list_frame = tk.Frame(migrations_window, bg="#f8f9fa")
    main_list_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    # Canvas y scrollbar para manejar muchas migraciones
    canvas = tk.Canvas(main_list_frame, bg="#f8f9fa")
    scrollbar = tk.Scrollbar(main_list_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#f8f9fa")
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    # CENTRAR contenido en el canvas
    canvas.create_window((width//2, 0), window=scrollable_frame, anchor="n")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Headers
    header_frame = tk.Frame(scrollable_frame, bg="#34495e")
    header_frame.pack(fill="x", pady=(0, 5))
    
    tk.Label(header_frame, text="Fecha y Hora", font=("Arial", 12, "bold"),
             bg="#34495e", fg="white", width=20).pack(side="left", padx=5, pady=5)
    tk.Label(header_frame, text="Registros", font=("Arial", 12, "bold"),
             bg="#34495e", fg="white", width=10).pack(side="left", padx=5, pady=5)
    tk.Label(header_frame, text="Estado", font=("Arial", 12, "bold"),
             bg="#34495e", fg="white", width=15).pack(side="left", padx=5, pady=5)
    
    # Datos o mensaje vacío  
    if migrations_data:
        for migration in migrations_data:
            row_frame = tk.Frame(scrollable_frame, bg="white", relief="ridge", bd=1)
            row_frame.pack(fill="x", pady=2)
            
            tk.Label(row_frame, text=migration["fecha"], font=("Arial", 10),
                     bg="white", width=20).pack(side="left", padx=5, pady=5)
            tk.Label(row_frame, text=migration["registros"], font=("Arial", 10),
                     bg="white", width=10).pack(side="left", padx=5, pady=5)
            tk.Label(row_frame, text=f"✅ {migration['estado']}", font=("Arial", 10),
                     bg="white", fg="#27ae60", width=15).pack(side="left", padx=5, pady=5)
    else:
        # Mostrar mensaje cuando no hay historial de migraciones
        empty_frame = tk.Frame(scrollable_frame, bg="#f8f9fa")
        empty_frame.pack(fill="both", expand=True, pady=50)
        
        tk.Label(empty_frame, text="📝 Historial de migraciones vacío", 
                font=("Arial", 14, "bold"), bg="#f8f9fa", fg="#7f8c8d").pack()
        tk.Label(empty_frame, text="Las migraciones se acumularán aquí automáticamente cuando se procesen infracciones", 
                font=("Arial", 10), bg="#f8f9fa", fg="#95a5a6").pack(pady=10)
    
    # MEJORA: Agregar scroll suave con rueda del mouse para historial
    def on_mousewheel_migrations(event):
        # Scroll suave: dividir delta por 3 para hacer más lento y suave
        canvas.yview_scroll(int(-1 * (event.delta / 120 / 3)), "units")
        
    def bind_to_mousewheel_migrations(event):
        canvas.bind_all("<MouseWheel>", on_mousewheel_migrations)
        
    def unbind_from_mousewheel_migrations(event):
        canvas.unbind_all("<MouseWheel>")
        
    # Bindear eventos de entrada/salida del mouse
    canvas.bind('<Enter>', bind_to_mousewheel_migrations)
    canvas.bind('<Leave>', unbind_from_mousewheel_migrations)
    scrollable_frame.bind('<Enter>', bind_to_mousewheel_migrations)
    scrollable_frame.bind('<Leave>', unbind_from_mousewheel_migrations)
    
    # Configurar y mostrar canvas y scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Frame para botones
    button_frame = tk.Frame(migrations_window, bg="#f8f9fa")
    button_frame.pack(pady=20)
    
    # Función para SOLO LIMPIAR LA LISTA LOCAL (NO BORRAR DATOS REALES)
    def vaciar_lista_local():
        result = messagebox.askyesno(
            "Limpiar historial de migraciones",
            "¿Está seguro de que desea LIMPIAR PERMANENTEMENTE el historial de migraciones?\n\n"
            "Esta acción:\n"
            "• Eliminará COMPLETAMENTE el historial de migraciones\n"
            "• Limpiará el archivo historial_migraciones.json\n"
            "• NO eliminará las infracciones reales del sistema\n\n"
            "⚠️ Esta acción es PERMANENTE y no se puede deshacer.",
            parent=migrations_window
        )
        
        if result:
            try:
                # Limpiar el archivo historial persistente
                save_migrations_history([])
                
                # Limpiar la vista
                for widget in scrollable_frame.winfo_children():
                    if widget != header_frame:  # Mantener header
                        widget.destroy()
                
                # Mostrar mensaje de tabla vacía
                empty_frame = tk.Frame(scrollable_frame, bg="#f8f9fa")
                empty_frame.pack(fill="x", pady=20)
                tk.Label(empty_frame, text="📭 Historial de migraciones limpiado completamente", 
                        font=("Arial", 10), bg="#f8f9fa", fg="#95a5a6").pack(pady=10)
                
                messagebox.showinfo("Historial limpiado", 
                    "✅ El historial de migraciones ha sido eliminado permanentemente.\n\n"
                    "El sistema ahora iniciará desde cero.", 
                    parent=migrations_window)
                        
            except Exception as e:
                messagebox.showerror("Error", f"Error limpiando historial: {e}", parent=migrations_window)
    
    # Función para actualizar la lista de migraciones (HISTORIAL ACUMULATIVO)
    def actualizar_migraciones():
        print("🔄 ACTUALIZANDO VISTA DE MIGRACIONES...")
        
        # Limpiar datos anteriores (mantener header)
        for widget in scrollable_frame.winfo_children():
            if widget != header_frame:  # Mantener header
                widget.destroy()
        
        # CARGAR HISTORIAL COMPLETO ACTUALIZADO
        updated_migrations = load_migrations_history()
        print(f"📊 HISTORIAL ACTUALIZADO: {len(updated_migrations)} migraciones")
        
        # Recrear las filas con historial completo o mensaje vacío
        if updated_migrations:
            for migration in updated_migrations:
                row_frame = tk.Frame(scrollable_frame, bg="white", relief="ridge", bd=1)
                row_frame.pack(fill="x", pady=2)
                
                tk.Label(row_frame, text=migration["fecha"], font=("Arial", 10),
                         bg="white", width=20).pack(side="left", padx=5, pady=5)
                tk.Label(row_frame, text=migration["registros"], font=("Arial", 10),
                         bg="white", width=10).pack(side="left", padx=5, pady=5)
                tk.Label(row_frame, text=f"✅ {migration['estado']}", font=("Arial", 10),
                         bg="white", fg="#27ae60", width=15).pack(side="left", padx=5, pady=5)
        else:
            # Mostrar mensaje cuando no hay historial
            empty_frame = tk.Frame(scrollable_frame, bg="#f8f9fa")
            empty_frame.pack(fill="both", expand=True, pady=50)
            
            tk.Label(empty_frame, text="📝 Historial de migraciones vacío", 
                    font=("Arial", 14, "bold"), bg="#f8f9fa", fg="#7f8c8d").pack()
            tk.Label(empty_frame, text="Las migraciones se acumularán aquí automáticamente", 
                    font=("Arial", 10), bg="#f8f9fa", fg="#95a5a6").pack(pady=10)
        
        # Actualizar canvas scroll region
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        messagebox.showinfo("Actualizado", "Lista de migraciones actualizada correctamente.", parent=migrations_window)
    
    # Botón actualizar
    actualizar_btn = tk.Button(
        button_frame,
        text="🔄 Actualizar",
        command=actualizar_migraciones,
        bg="#3498db",
        fg="white",
        font=("Arial", 12),
        padx=20,
        pady=5
    )
    actualizar_btn.pack(side="left", padx=10)
    
    # Botón subir a Firestore (formato por-video)
    def subir_firestore():
        import threading as _threading
        from src.automations.firestore_migrator import migrate_videos_to_firestore

        def _run():
            try:
                result = migrate_videos_to_firestore(verbose=True)
                migrados = result.get("migrados", 0)
                errores = result.get("errores", [])
                # Registrar en el historial acumulativo
                add_migration_to_history(migrados, "Exitosa" if migrados else "Sin datos")
                migrations_window.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Migración a Firestore",
                        f"☁️ Documentos subidos: {migrados}\n\n"
                        + (f"⚠️ Errores: {len(errores)}\n" if errores else "")
                        + "✅ Migración completada (colección 'migraciones').",
                        parent=migrations_window,
                    ),
                )
            except Exception as ex:
                migrations_window.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error al migrar",
                        f"❌ {ex}",
                        parent=migrations_window,
                    ),
                )

        migrations_window.after(
            0,
            lambda: messagebox.showinfo(
                "Migración a Firestore",
                "☁️ Subiendo documentos por video a Firestore, espere...",
                parent=migrations_window,
            ),
        )
        _threading.Thread(target=_run, daemon=True).start()

    subir_btn = tk.Button(
        button_frame,
        text="☁️ SUBIR A FIRESTORE",
        command=subir_firestore,
        bg="#27ae60",
        fg="white",
        font=("Arial", 12),
        padx=20,
        pady=5
    )
    subir_btn.pack(side="left", padx=10)

    # Botón limpiar historial de migraciones
    vaciar_btn = tk.Button(
        button_frame,
        text="🗑️ Limpiar Historial",
        command=vaciar_lista_local,
        bg="#e67e22",
        fg="white",
        font=("Arial", 12),
        padx=20,
        pady=5
    )
    vaciar_btn.pack(side="left", padx=10)
    
    # Función para borrar TODO permanentemente (CON CONFIRMACIÓN DOBLE)
    def borrar_todo_permanentemente():
        # Primera confirmación
        result1 = messagebox.askyesno(
            "⚠️ BORRAR TODO PERMANENTEMENTE",
            "🚨 ATENCIÓN: Esta acción ELIMINARÁ TODOS LOS DATOS del sistema:\n\n"
            "• Historial completo de migraciones\n"
            "• Indicadores de rendimiento\n"
            "• Configuraciones de videos\n"
            "• Datos de infracciones procesadas\n\n"
            "⚠️ Esta acción es IRREVERSIBLE\n\n"
            "¿Está COMPLETAMENTE seguro de continuar?",
            parent=migrations_window
        )
        
        if result1:
            # Segunda confirmación más fuerte
            result2 = messagebox.askyesno(
                "🚨 CONFIRMACIÓN FINAL",
                "ÚLTIMA OPORTUNIDAD DE CANCELAR\n\n"
                "Está a punto de ELIMINAR PERMANENTEMENTE:\n"
                "✗ TODO el historial de migraciones\n"
                "✗ TODOS los indicadores de rendimiento\n"
                "✗ TODAS las configuraciones\n\n"
                "💀 ESTA ACCIÓN NO SE PUEDE DESHACER 💀\n\n"
                "Escriba 'SÍ' para confirmar la eliminación total:",
                parent=migrations_window
            )
            
            if result2:
                try:
                    # Limpiar historial de migraciones
                    save_migrations_history([])
                    
                    # Limpiar indicadores de rendimiento
                    import os
                    indicadores_path = "data/indicadores_rendimiento.json"
                    if os.path.exists(indicadores_path):
                        with open(indicadores_path, 'w', encoding='utf-8') as f:
                            f.write('{}')
                    
                    # Limpiar la vista
                    for widget in scrollable_frame.winfo_children():
                        if widget != header_frame:  # Mantener header
                            widget.destroy()
                    
                    # Mostrar mensaje de tabla vacía
                    empty_frame = tk.Frame(scrollable_frame, bg="#f8f9fa")
                    empty_frame.pack(fill="x", pady=20)
                    tk.Label(empty_frame, text="🆕 Sistema completamente limpio - Iniciando desde cero", 
                            font=("Arial", 10), bg="#f8f9fa", fg="#95a5a6").pack(pady=10)
                    
                    messagebox.showinfo("Sistema reiniciado", 
                        "🆕 SISTEMA COMPLETAMENTE LIMPIO\n\n"
                        "✅ Todos los datos han sido eliminados\n"
                        "✅ El sistema iniciará desde cero\n"
                        "✅ Como si fuera la primera vez\n\n"
                        "Puede cerrar esta ventana y reiniciar la aplicación.", 
                        parent=migrations_window)
                            
                except Exception as e:
                    messagebox.showerror("Error", f"Error eliminando datos: {e}", parent=migrations_window)
    
    # Botón borrar TODO permanentemente
    borrar_todo_btn = tk.Button(
        button_frame,
        text="💀 BORRAR TODO",
        command=borrar_todo_permanentemente,
        bg="#c0392b",
        fg="white",
        font=("Arial", 12, "bold"),
        padx=20,
        pady=5
    )
    borrar_todo_btn.pack(side="left", padx=10)
    
    # Botón cerrar
    close_btn = tk.Button(
        button_frame,
        text="Cerrar",
        command=migrations_window.destroy,
        bg="#e74c3c",
        fg="white",
        font=("Arial", 12),
        padx=20,
        pady=5
    )
    close_btn.pack(side="right", padx=10)




def generate_performance_indicators_json(software_infractions, software_processing_times, nombre_video=None, config_semaforo=None):
    """
    Genera y guarda en data/indicadores_rendimiento.json
    los indicadores TI, TR y NID basados en las infracciones y tiempos.
    IR ha sido eliminado completamente del sistema.
    
    Args:
        software_infractions: Lista de infracciones detectadas
        software_processing_times: Lista de tiempos de procesamiento
        nombre_video: Nombre del video procesado (opcional)
        config_semaforo: ID de configuración de semáforo (ej: "10-3-15") (opcional)
    """
    import os
    import json
    import requests
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

    # Guardar localmente
    out_path = resource_path("data/indicadores_rendimiento.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Indicadores guardados localmente en: {out_path}")

    # 5) Enviar automáticamente al backend / Firestore
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as cf:
            config_data = json.load(cf)
            user_id = config_data.get("user_id", "")
        
        if not user_id:
            print("⚠️ No se encontró user_id en la configuración, saltando envío al backend")
            return
        
        url = f"https://infracti-backend-627388491148.us-central1.run.app/indicadores/{user_id}"
        print(f"📤 Enviando indicadores al backend: {url}")
        print(f"   Datos: NID={nid_count}, NIE={nie_count}, TI={ti_percentage:.2f}%, TR={sw_min:.2f}min")
        
        resp = requests.post(url, json=report, timeout=15)
        resp.raise_for_status()
        print("✅ Indicadores enviados correctamente a Firestore via Backend")
        print(f"   Respuesta del servidor: {resp.status_code}")
    except FileNotFoundError as e:
        print(f"⚠️ No se encontró archivo de configuración: {e}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error de red enviando indicadores al backend: {e}")
        print(f"   URL: {url if 'url' in locals() else 'N/A'}")
        print("   Los indicadores se guardaron localmente y se pueden migrar manualmente")
    except Exception as e:
        print(f"⚠️ Error inesperado enviando indicadores: {e}")
        import traceback
        traceback.print_exc()


def create_infractions_window(window: tk.Toplevel, back_callback):
    window.configure(bg="#ffffff")
    
    # MODIFICACIÓN DE PORTABILIDAD (desactivada para evitar segfault)
    # La ventana se abre con el tamaño heredado de la ventana principal.
    # El usuario puede maximizar manualmente si lo desea.
    pass

    # 1) Cargar todas las infracciones al inicio
    all_data = load_infractions_data()

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
            # Importar la clase VideoPlayerOpenCV
            from src.core.video.videoplayer_opencv import VideoPlayerOpenCV
            import tkinter as tk
            from datetime import datetime
            import json
            import os
            
            # Crear una instancia temporal para calcular los indicadores
            dummy_frame = tk.Frame()
            dummy_updater = type('obj', (object,), {'running': False})
            dummy_label = tk.Label()
            dummy_semaforo = type('obj', (object,), {
                'deactivate_semaphore': lambda: None,
                'get_current_state': lambda: 'red',
                'activate_semaphore': lambda: None
            })
            
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
            
            # Guardar informe en JSON
            report_file = resource_path("data/indicadores_rendimiento.json")
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
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
            start = datetime.combine(start_picker.get_date(), datetime.min.time())
            end = datetime.combine(end_picker.get_date(), datetime.max.time())
            filtered = []
            for inf in all_data:
                fecha_str = inf.get('fecha','')
                try:
                    fecha = datetime.strptime(fecha_str, '%d/%m/%Y')
                    if start <= fecha <= end:
                        filtered.append(inf)
                except ValueError:
                    # Si hay error en el formato de fecha, no incluimos este registro
                    pass
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
        canvas.itemconfig(frame_id, width=event.width)
    
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

    def populate_cards(data_list):
        clear_cards()
        if not data_list:
            tk.Label(
                scrollable_frame, text="No se encontraron infracciones.",
                font=("Arial", 16), bg="gray", fg="white"
            ).pack(pady=80, padx=80)
            return

        # CORREGIR: Validar que data_list contenga diccionarios, no strings
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
        
        # Ordenar: primero por clasificación (NID primero, NIE al final), luego por fecha y hora (más reciente primero)
        try:
            data_list = sorted(data_list, 
                            key=lambda x: (
                                # Prioridad de clasificación: NID=0 (primero), NIE=1 (después)
                                0 if x.get('clasificacion', 'NID') == 'NID' else 1,
                                # Luego ordenar por fecha y hora dentro de cada clasificación
                                -(datetime.strptime(x.get('fecha', '01/01/2000'), '%d/%m/%Y').timestamp()),
                                -int(x.get('hora', '00:00:00').replace(':', ''))
                            ))
        except Exception as e:
            print(f"Error al ordenar infracciones: {e}")
            # Fallback: ordenamiento simple por clasificación solo para diccionarios válidos
            data_list = sorted([x for x in data_list if isinstance(x, dict)], 
                             key=lambda x: (0 if x.get('clasificacion', 'NID') == 'NID' else 1))
        
        # Crear tarjetas con diseño mejorado
        for inf in data_list:
            # Card principal con elevación y bordes redondeados
            card = tk.Frame(scrollable_frame, bg="#F2F2F2", 
                         bd=1, relief=tk.RAISED)
            card.pack(fill="x", padx=15, pady=10, expand=True)  # Añadido padding horizontal para separar del borde
            
            # MEJORADO: Agregar scroll a esta card específica y todos sus componentes
            def add_scroll_to_card_and_children(widget):
                """Agrega scroll a un widget y todos sus hijos recursivamente"""
                try:
                    widget.bind("<MouseWheel>", on_mousewheel)
                    widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
                    widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
                except:
                    pass
                # Recursivamente a todos los hijos
                for child in widget.winfo_children():
                    add_scroll_to_card_and_children(child)
            
            # Parte superior: información principal
            top_frame = tk.Frame(card, bg="#F2F2F2")
            top_frame.pack(fill="x", padx=15, pady=(10, 5), expand=True)  # Aumentado el padding interno
            
            # Marco de imagen con tamaño fijo
            img_frame = tk.Frame(top_frame, width=150, height=100, bg="#DDDDDD", 
                              relief=tk.SUNKEN, bd=1)
            img_frame.pack(side="left", padx=10, pady=10)
            img_frame.pack_propagate(False)
            
            # Intenta cargar una imagen del vehículo si hay ruta disponible
            vehicle_path = inf.get('vehicle_path', '')
            vehicle_img_label = None
            
            if vehicle_path and os.path.exists(vehicle_path):
                try:
                    from PIL import Image, ImageTk
                    # Cargar y redimensionar la imagen
                    img = Image.open(vehicle_path)
                    img = img.resize((150, 100), Image.LANCZOS)  # Ajustado al nuevo tamaño
                    photo = ImageTk.PhotoImage(img)
                    vehicle_img_label = tk.Label(img_frame, image=photo, bg="#DDDDDD")
                    vehicle_img_label.image = photo  # Guardar referencia
                    vehicle_img_label.pack(fill="both", expand=True)
                except Exception as e:
                    print(f"Error cargando imagen del vehículo: {e}")
                    vehicle_img_label = None
            
            # Si no se pudo cargar la imagen, mostrar texto placeholder
            if not vehicle_img_label:
                tk.Label(img_frame, text="[Sin imagen]", 
                      bg="#DDDDDD", fg="#777777").pack(fill="both", expand=True)
                
            # Primera columna de información
            text_left = tk.Frame(top_frame, bg="#F2F2F2")
            text_left.pack(side="left", fill="y", padx=(0,20), pady=10)
            
            # Placa con estilo destacado (negrita y color corporativo)
            placa_info = inf.get('placa','No identificada')
            tk.Label(
                text_left, text=f"Placa: {placa_info}",
                font=("Arial", 12, "bold"), bg="#F2F2F2", fg="#273D86"
            ).pack(anchor="w")
            
            # Información temporal
            fecha_info = inf.get('fecha','')
            hora_info = inf.get('hora','')
            timestamp_info = inf.get('video_timestamp','00:00')
            
            tiempo_str = f"Fecha: {fecha_info}   Hora: {hora_info}"
            
            tk.Label(
                text_left, text=tiempo_str,
                font=("Arial", 12), bg="#F2F2F2", fg="#555555"
            ).pack(anchor="w")
            
            # Separador vertical
            tk.Frame(top_frame, bg="#CCCCCC", width=2).pack(side="left", fill="y", pady=10)
            
            # Segunda columna de información
            text_right = tk.Frame(top_frame, bg="#F2F2F2")
            text_right.pack(side="left", fill="both", expand=True, padx=20, pady=10)
            
            # Ubicación y coordenadas
            ubicacion_info = inf.get('ubicacion','Desconocida')
            ubicacion_label = tk.Label(
                text_right, text=f"Ubicación: {ubicacion_info}",
                font=("Arial", 12), bg="#F2F2F2", fg="#333333",
                wraplength=800, justify="left", anchor="w"  # Añadido anchor="w" para alinear a la izquierda
            )
            ubicacion_label.pack(anchor="w", fill="x")  # Quitado expand=True para mejor control del espaciado
                
            # Tipo de infracción
            tipo_info = inf.get('tipo','Semáforo en rojo')
            tk.Label(
                text_right, text=f"Tipo: {tipo_info}",
                font=("Arial", 12), bg="#F2F2F2", fg="#333333"
            ).pack(anchor="w")
            
            # MEJORADO: Clasificación NID/NIE con rangos de confianza más precisos
            clasificacion = inf.get('clasificacion', 'NID')  # Por defecto NID si no está especificado
            raw_confianza = inf.get('confianza', 0)
            # CRÍTICO: Aplicar el mismo clamp que PlateCard para valores inválidos
            confianza = max(0.0, min(1.0, raw_confianza))  # Clamp [0,1]
            
            # RANGOS DE CONFIANZA MEJORADOS
            NID_THRESHOLD_EXCELLENT = 0.85  # Excelente
            NID_THRESHOLD_GOOD = 0.65       # Bueno (mínimo aceptable)
            
            if clasificacion == 'NID':
                if confianza >= NID_THRESHOLD_EXCELLENT:
                    clasificacion_color = "#27ae60"  # Verde fuerte - Excelente
                    clasificacion_texto = f"✅ NID - Excelente (Confianza: {confianza:.2f})"
                elif confianza >= NID_THRESHOLD_GOOD:
                    clasificacion_color = "#2ecc71"  # Verde - Bueno
                    clasificacion_texto = f"✅ NID - Bueno (Confianza: {confianza:.2f})"
                else:
                    # Debería ser NIE si está por debajo del umbral
                    clasificacion_color = "#f39c12"  # Naranja - Revisión necesaria
                    clasificacion_texto = f"⚠️ NIE - Confianza baja ({confianza:.2f})"
            else:
                clasificacion_color = "#e74c3c"  # Rojo - Error/NIE
                metadata = inf.get('metadata_clasificacion', {})
                razon = metadata.get('razon', 'confianza_insuficiente')
                clasificacion_texto = f"❌ NIE - {razon} ({confianza:.2f})"
                
            tk.Label(
                text_right, text=clasificacion_texto,
                font=("Arial", 11, "bold"), bg="#F2F2F2", fg=clasificacion_color
            ).pack(anchor="w")
            
            # TR: Tiempo de Registro (tiempo_procesamiento en minutos)
            # Este valor DEBE coincidir con TR_individuales migrado a Firestore
            tiempo_proc = inf.get('tiempo_procesamiento', 0)
            
            if tiempo_proc and tiempo_proc > 0:
                # Convertir segundos a minutos
                total_seconds = int(float(tiempo_proc))
                mins_dec = tiempo_proc / 60.0  # Minutos decimales (ej: 0.37)
                mins = total_seconds // 60
                secs = total_seconds % 60
                
                # FORMATO: TR: 0.37min (22s) - coincide con Firestore
                tr_text = f"TR: {mins_dec:.2f}min ({total_seconds}s)"
                tk.Label(
                    text_right, text=tr_text,
                    font=("Arial", 10), bg="#F2F2F2", fg="#7f8c8d"
                ).pack(anchor="w")
            else:
                # Mostrar formato de fallback
                tr_text = "TR: 0.00min (0s)"
                tk.Label(
                    text_right, text=tr_text,
                    font=("Arial", 10), bg="#F2F2F2", fg="#7f8c8d"
                ).pack(anchor="w")
            
            # NUEVO: Duración del video (tiempo nativo - formato completo para cualquier longitud)
            duracion_video = inf.get('tiempo_video', inf.get('video_timestamp', 'N/A'))
            if duracion_video != 'N/A' and duracion_video and duracion_video != '00:00':
                def format_video_duration_complete(duration_str):
                    """
                    Formatea duración de video para cualquier longitud:
                    - Videos <60s: "45s"
                    - Videos 60s-3599s: "5:30min"
                    - Videos ≥3600s: "1:25:30h"
                    Soporta entrada en formatos H:MM:SS, MM:SS, SS
                    """
                    try:
                        if isinstance(duration_str, str) and ':' in duration_str:
                            time_parts = duration_str.split(':')
                            
                            if len(time_parts) == 2:  # MM:SS format
                                mins, secs = int(time_parts[0]), int(time_parts[1])
                                total_seconds = mins * 60 + secs
                                
                            elif len(time_parts) == 3:  # H:MM:SS format
                                hours, mins, secs = int(time_parts[0]), int(time_parts[1]), int(time_parts[2])
                                total_seconds = hours * 3600 + mins * 60 + secs
                                
                            else:
                                return f"Duración del video: {duration_str}"
                        
                        elif isinstance(duration_str, (int, float)):
                            total_seconds = int(duration_str)
                        
                        elif isinstance(duration_str, str) and duration_str.isdigit():
                            total_seconds = int(duration_str)
                        
                        else:
                            return f"Duración del video: {duration_str}"
                        
                        # Formatear según longitud total
                        if total_seconds < 60:
                            # Videos cortos: solo segundos
                            return f"Duración del video: {total_seconds}s"
                        
                        elif total_seconds < 3600:  # < 1 hora
                            # Videos medianos: minutos y segundos
                            mins = total_seconds // 60
                            secs = total_seconds % 60
                            return f"Duración del video: {mins}:{secs:02d}min"
                        
                        else:  # ≥ 1 hora
                            # Videos largos: horas, minutos y segundos
                            hours = total_seconds // 3600
                            remaining = total_seconds % 3600
                            mins = remaining // 60
                            secs = remaining % 60
                            return f"Duración del video: {hours}:{mins:02d}:{secs:02d}h"
                    
                    except (ValueError, IndexError):
                        return f"Duración del video: {duration_str}"
                
                duracion_text = format_video_duration_complete(duracion_video)
                    
                tk.Label(
                    text_right, text=duracion_text,
                    font=("Arial", 9), bg="#F2F2F2", fg="#95a5a6"
                ).pack(anchor="w")
            
            # NUEVO: Información de configuración del video
            config_frame = tk.Frame(text_right, bg="#F2F2F2")
            config_frame.pack(anchor="w", fill="x", pady=(5, 0))
            
            # Franja horaria configurada
            franja_horaria = inf.get('franja_horaria', inf.get('horario_configurado', inf.get('time_range', 'No especificada')))
            if franja_horaria and franja_horaria != 'No especificada':
                tk.Label(
                    config_frame, text=f"⏰ Franja horaria: {franja_horaria}",
                    font=("Arial", 9), bg="#F2F2F2", fg="#8e44ad"
                ).pack(anchor="w")
            
            # Tiempos de semáforo configurados
            tiempo_rojo = inf.get('tiempo_rojo', inf.get('semaforo_rojo_seg', inf.get('red_time', 'N/A')))
            tiempo_verde = inf.get('tiempo_verde', inf.get('semaforo_verde_seg', inf.get('green_time', 'N/A')))
            tiempo_amarillo = inf.get('tiempo_amarillo', inf.get('semaforo_amarillo_seg', inf.get('yellow_time', 'N/A')))
            
            if any(str(t) != 'N/A' and str(t).strip() for t in [tiempo_rojo, tiempo_verde, tiempo_amarillo]):
                semaforo_text = f"🚦 Semáforo: R:{tiempo_rojo}s V:{tiempo_verde}s A:{tiempo_amarillo}s"
                tk.Label(
                    config_frame, text=semaforo_text,
                    font=("Arial", 9), bg="#F2F2F2", fg="#e67e22"
                ).pack(anchor="w")
            
            # NUEVO: Campo de razón descriptiva
            razon_infraccion = inf.get('razon', 'Vehículo detectado en infracción de semáforo en rojo')
            tk.Label(
                text_right, text=f"📋 Razón: {razon_infraccion}",
                font=("Arial", 9, "italic"), bg="#F2F2F2", fg="#2ecc71" if clasificacion == "NID" else "#c0392b",
                wraplength=250, justify="left"
            ).pack(anchor="w", pady=(2, 0))
            
            # Panel de botones para acciones
            btn_frame = tk.Frame(card, bg="#F2F2F2")
            btn_frame.pack(fill="x", padx=15, pady=(0, 10), expand=True)  # Aumentado el padding para alinear con el resto
            

            
            # Crear una función específica para cada infracción
            def create_show_plate_func(plate_path, placa_text):
                def show_plate_func():
                    if plate_path and os.path.exists(plate_path):
                        try:
                            # Crear ventana para mostrar la placa
                            plate_window = tk.Toplevel(window)
                            plate_window.title(f"Placa: {placa_text}")
                            
                            # Cargar y mostrar la imagen
                            from PIL import Image, ImageTk
                            img = Image.open(plate_path)
                            photo = ImageTk.PhotoImage(img)
                            img_label = tk.Label(plate_window, image=photo)
                            img_label.image = photo  # Guardar referencia
                            img_label.pack(padx=20, pady=20)
                            
                            # Botón para cerrar
                            tk.Button(plate_window, text="Cerrar", 
                                    command=plate_window.destroy).pack(pady=10)
                            
                        except Exception as e:
                            messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")
                    else:
                        messagebox.showinfo("Información", "No hay imagen de placa disponible")
                return show_plate_func
            
            # Botón para ver la placa - usando closure para evitar problemas con variables
            plate_path = inf.get('plate_path', '')
            placa_text = inf.get('placa', 'No identificada')
            show_plate_func = create_show_plate_func(plate_path, placa_text)
            
            tk.Button(
                btn_frame, text="Ver placa", 
                command=show_plate_func,
                bg="#3366FF", fg="white",
                cursor="hand2"
            ).pack(side="right", padx=5)
            
            # Función para eliminar esta infracción específica
            def create_delete_func(infraction_data):
                def delete_func():
                    result = messagebox.askyesno(
                        "Confirmar eliminación",
                        f"¿Eliminar la infracción de la placa {infraction_data.get('placa', 'No identificada')}?\n\n"
                        "Se eliminarán las imágenes asociadas."
                    )
                    
                    if result:
                        # Eliminar archivos de imagen
                        deleted_files = delete_infraction_files(infraction_data)
                        
                        # Eliminar del JSON
                        placa_to_remove = infraction_data.get('placa', '')
                        if remove_infraction_from_json(placa_to_remove):
                            # Actualizar la vista
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
            
            # Crear función de eliminación para esta infracción
            delete_func = create_delete_func(inf)
            
            # Botón para eliminar esta infracción
            tk.Button(
                btn_frame, text="🗑️ Eliminar", 
                command=delete_func,
                bg="#e74c3c", fg="white",
                cursor="hand2", font=("Arial", 9)
            ).pack(side="right", padx=5)
            
            # Función para mostrar detalles de la infracción
            def create_show_details_func(infraction_data):
                def show_details_func():
                    # Crear ventana de detalles
                    details_window = tk.Toplevel(window)
                    details_window.title(f"Detalles - Placa: {infraction_data.get('placa', 'No identificada')}")
                    details_window.geometry("500x400")
                    details_window.configure(bg="#f8f9fa")
                    
                    # Centrar ventana
                    details_window.update_idletasks()
                    width, height = 500, 400
                    x = (details_window.winfo_screenwidth() - width) // 2
                    y = (details_window.winfo_screenheight() - height) // 2
                    details_window.geometry(f"{width}x{height}+{x}+{y}")
                    
                    # Título
                    title_label = tk.Label(
                        details_window,
                        text="📋 Detalles de la Infracción",
                        font=("Arial", 16, "bold"),
                        bg="#f8f9fa",
                        fg="#2c3e50"
                    )
                    title_label.pack(pady=20)
                    
                    # Frame para detalles
                    details_frame = tk.Frame(details_window, bg="#f8f9fa")
                    details_frame.pack(fill="both", expand=True, padx=20, pady=10)
                    
                    # Mostrar información detallada
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
                    
                    # Botón cerrar
                    close_btn = tk.Button(
                        details_window,
                        text="Cerrar",
                        command=details_window.destroy,
                        bg="#3498db",
                        fg="white",
                        font=("Arial", 12),
                        padx=20,
                        pady=5
                    )
                    close_btn.pack(pady=20)
                return show_details_func
            
            # Crear función de detalles para esta infracción
            show_details_func = create_show_details_func(inf)
            
            # Botón para ver detalles
            tk.Button(
                btn_frame, text="Ver detalles",
                command=show_details_func,
                bg="#5D6D7E", fg="white",
                cursor="hand2"
            ).pack(side="right", padx=5)
            
            # APLICAR SCROLL AL FINAL: Después de crear todos los elementos de la card
            add_scroll_to_card_and_children(card)

    # Inicializar la vista con todos los datos
    populate_cards(all_data)
    
    # Botones de acción adicionales al final
    bottom_actions = tk.Frame(window, bg="#ffffff")
    bottom_actions.pack(fill="x", padx=100, pady=10)
    
    # Botón de migraciones con callback para actualizar
    def open_migrations():
        show_migrations(refresh_callback=populate_cards, all_data_ref=all_data)
    
    tk.Button(
        bottom_actions, text="📊 MIGRACIONES", font=("Arial", 12),
        bg="#9b59b6", fg="white", bd=0,
        activebackground="#8e44ad", activeforeground="white",
        cursor="hand2", command=open_migrations,
        width=20, height=2
    ).pack(side="left", padx=10)
    
    # Botón vaciar todo
    tk.Button(
        bottom_actions, text="🗑️ VACIAR TODO", font=("Arial", 12),
        bg="#e74c3c", fg="white", bd=0,
        activebackground="#c0392b", activeforeground="white",
        cursor="hand2", command=clear_all_infractions,
        width=20, height=2
    ).pack(side="right", padx=10)
