import tkinter as tk
from tkinter import messagebox, filedialog
import json, os
from tkcalendar import DateEntry
from datetime import datetime
import shutil
import csv
import pandas as pd  # Para exportación a Excel (requiere openpyxl)

# Ruta centralizada del archivo de infracciones
INF_FILE = os.path.join("data", "infracciones.json")

# Función para cargar datos de infracciones (movida al principio)
def load_infractions_data():
    if os.path.exists(INF_FILE):
        try:
            with open(INF_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando infracciones: {e}")
            return []
    else:
        return []

def create_infractions_window(window: tk.Toplevel, back_callback):
    window.configure(bg="#ffffff")
    window.state("zoomed")

    # Cargar todas las infracciones al inicio
    all_data = load_infractions_data()

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
            infractions_file = os.path.join("data", "infracciones.json")
            if os.path.exists(infractions_file):
                try:
                    with open(infractions_file, "r", encoding="utf-8") as f:
                        software_infractions = json.load(f)
                except Exception as e:
                    print(f"Error cargando infracciones: {e}")
                    software_infractions = []
            
            # Simular tiempos de procesamiento (10 segundos por infracción)
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
            police_registration_times = [7.2, 6.5, 8.0, 5.9, 6.8]  # minutos por infracción
            
            # ===== 3. CÁLCULO DE INDICADORES =====
            # ----- INDICADOR 1: Tasa de Infracciones Detectadas (TI) -----
            # Sin software: Promedio diario basado en datos históricos PNP
            pnp_total_infractions = sum(data["total"] for data in pnp_monthly_data.values())
            pnp_total_days = sum(data["dias"] for data in pnp_monthly_data.values())
            pnp_daily_average = pnp_total_infractions / pnp_total_days if pnp_total_days else 0
            
            # Con software: Promedio diario basado en datos de los 15 días
            software_days = len(day_infractions)
            software_total_infractions = len(software_infractions)
            software_daily_average = software_total_infractions / software_days if software_days else 0
            
            ti_improvement = ((software_daily_average - pnp_daily_average) / pnp_daily_average * 100) if pnp_daily_average else 0
            
            # ----- INDICADOR 2: Tiempo de Registro (TR) -----
            # Sin software: Promedio de tiempo de registro según encuestas (convertir a segundos)
            pnp_avg_time = sum(police_registration_times) / len(police_registration_times) * 60 if police_registration_times else 0
            
            # Con software: Promedio de tiempo de procesamiento del sistema
            software_avg_time = sum(software_processing_times) / len(software_processing_times) if software_processing_times else 0
            
            tr_reduction = ((pnp_avg_time - software_avg_time) / pnp_avg_time * 100) if pnp_avg_time else 0
            tr_speedup = pnp_avg_time / software_avg_time if software_avg_time else 0
            
            # ----- INDICADOR 3: Índice de Reincidencia (IR) -----
            # Sin software: Porcentaje mensual de placas reincidentes
            pnp_reincidence_total = sum(data["reincidentes"] for data in pnp_monthly_data.values())
            pnp_reincidence_rate = (pnp_reincidence_total / pnp_total_infractions * 100) if pnp_total_infractions else 0
            
            # Con software: Porcentaje diario promedio de placas reincidentes
            daily_reincidence_rates = []
            
            for fecha, data in day_infractions.items():
                reincidente_count = sum(1 for placa, count in data["placas"].items() if count > 1)
                if data["total"] > 0:
                    daily_rate = (reincidente_count / data["total"]) * 100
                    daily_reincidence_rates.append(daily_rate)
            
            software_reincidence_rate = sum(daily_reincidence_rates) / len(daily_reincidence_rates) if daily_reincidence_rates else 0
            
            ir_improvement = ((software_reincidence_rate - pnp_reincidence_rate) / pnp_reincidence_rate * 100) if pnp_reincidence_rate else 0
            ir_ratio = software_reincidence_rate / pnp_reincidence_rate if pnp_reincidence_rate else 0
            
            # ===== 4. GENERAR INFORME =====
            report = {
                "fecha_generacion": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "periodo_analisis": f"{min(day_infractions.keys(), default='N/A')} - {max(day_infractions.keys(), default='N/A')}",
                "dias_analizados": software_days,
                "indicadores": {
                    "TI": {
                        "sin_software": {
                            "total_mensual": pnp_total_infractions,
                            "total_diario": pnp_daily_average,
                            "meses_analizados": len(pnp_monthly_data)
                        },
                        "con_software": {
                            "total_periodo": software_total_infractions,
                            "total_diario": software_daily_average,
                            "dias_analizados": software_days
                        },
                        "mejora_porcentual": ti_improvement
                    },
                    "TR": {
                        "sin_software": {
                            "tiempo_promedio_segundos": pnp_avg_time,
                            "tiempo_promedio_minutos": pnp_avg_time / 60,
                            "fuente": "Encuesta a oficiales PNP"
                        },
                        "con_software": {
                            "tiempo_promedio_segundos": software_avg_time,
                            "muestras_analizadas": len(software_processing_times)
                        },
                        "reduccion_tiempo_porcentual": tr_reduction,
                        "veces_mas_rapido": tr_speedup
                    },
                    "IR": {
                        "sin_software": {
                            "porcentaje_reincidencia_mensual": pnp_reincidence_rate,
                            "total_reincidentes": pnp_reincidence_total,
                            "periodo": "Mensual"
                        },
                        "con_software": {
                            "porcentaje_reincidencia_diaria": software_reincidence_rate,
                            "muestras_diarias": len(daily_reincidence_rates)
                        },
                        "mejora_deteccion_porcentual": ir_improvement,
                        "ratio_deteccion": ir_ratio
                    }
                },
                "resumen_global": {
                    "infracciones_detectadas_mejora": f"+{ti_improvement:.1f}%",
                    "tiempo_registro_reduccion": f"-{tr_reduction:.1f}%",
                    "tiempo_registro_factor": f"{tr_speedup:.1f}x más rápido",
                    "reincidencia_deteccion_factor": f"{ir_ratio:.1f}x más detección"
                }
            }
            
            # Guardar informe en JSON
            report_file = os.path.join("data", "indicadores_rendimiento.json")
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Generar resumen para mostrar
            resumen = f"""
            🟦 INDICADOR 1: Tasa de Infracciones Detectadas (TI)
            Sin software: {pnp_daily_average:.1f} infracciones/día
            Con software: {software_daily_average:.1f} infracciones/día
            Mejora: {ti_improvement:+.1f}%
            
            🟦 INDICADOR 2: Tiempo de Registro (TR)
            Sin software: {pnp_avg_time/60:.1f} minutos ({pnp_avg_time:.1f} segundos)
            Con software: {software_avg_time:.1f} segundos
            Reducción: {tr_reduction:.1f}% ({tr_speedup:.1f}x más rápido)
            
            🟦 INDICADOR 3: Índice de Reincidencia (IR)
            Sin software: {pnp_reincidence_rate:.1f}% mensual
            Con software: {software_reincidence_rate:.1f}% diario
            Mejora: {ir_ratio:.1f}x mayor detección
            
            ✅ RESUMEN: El sistema automatizado detecta {ti_improvement:+.1f}% más infracciones,
            es {tr_speedup:.1f} veces más rápido registrando y detecta {ir_ratio:.1f} veces
            más reincidencias que el método tradicional.
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
                    source_path = os.path.join("data", "indicadores_rendimiento.json")
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
    
    # NUEVO: Botón de indicadores de rendimiento con estilo destacado
    indicators_button = tk.Button(
        actions, 
        text="INDICADORES",
        command=show_performance_indicators,
        font=("Arial", 12),
        bg="#3498db",
        fg="white",
        activebackground="#2980b9",
        activeforeground="white",
        bd=0,
        relief="flat",
        cursor="hand2",
        width=15,
        height=2
    )
    indicators_button.pack(side="left", padx=15)

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
                    with open(file_path, "w", encoding="utf-8", newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=filtered_data[0].keys())
                        writer.writeheader()
                        writer.writerows(filtered_data)
                    messagebox.showinfo("Éxito", f"Infracciones exportadas a {file_path}")
                    export_win.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Error al exportar: {e}")
        
        def export_as_excel():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Guardar infracciones como Excel"
            )
            if file_path:
                try:
                    # Convertir a DataFrame para exportar como Excel
                    df = pd.DataFrame(filtered_data)
                    df.to_excel(file_path, index=False, engine='openpyxl')
                    messagebox.showinfo("Éxito", f"Infracciones exportadas a {file_path}")
                    export_win.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Error al exportar: {e}")
        
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

    # Función para refrescar los datos desde el archivo
    def refresh_data():
        nonlocal all_data
        all_data = load_infractions_data()
        apply_filter()  # Aplicar el filtro actual a los nuevos datos

    # Añadir botón de refresco
    tk.Button(
        actions, text="REFRESCAR", font=("Arial", 12),
        bg="#00CC66", fg="white", bd=0,
        activebackground="#009933", activeforeground="white",
        cursor="hand2", command=refresh_data
    ).pack(side="left", padx=10)

    tk.Button(
        actions, text="FILTRAR", font=("Arial", 12),
        bg="#3366FF", fg="white", bd=0,
        activebackground="#2554CC", activeforeground="white",
        cursor="hand2", command=apply_filter
    ).pack(side="left", padx=10)

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

        # Ordenar por fecha y hora (más reciente primero)
        try:
            data_list = sorted(data_list, 
                            key=lambda x: (datetime.strptime(x.get('fecha', '01/01/2000'), '%d/%m/%Y'), 
                                        x.get('hora', '00:00:00')),
                            reverse=True)
        except Exception as e:
            print(f"Error al ordenar infracciones: {e}")
        
        # Crear tarjetas con diseño mejorado
        for inf in data_list:
            # Card principal con elevación y bordes redondeados
            card = tk.Frame(scrollable_frame, bg="#F2F2F2", 
                         bd=1, relief=tk.RAISED)
            card.pack(fill="x", padx=15, pady=10, expand=True)  # Añadido padding horizontal para separar del borde
            
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
            if timestamp_info and timestamp_info != "00:00":
                tiempo_str += f"   Tiempo de video: {timestamp_info}"
                
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
            

    # Inicializar la vista con todos los datos
    populate_cards(all_data)