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
            
            # Botón para ver detalles (pendiente para futura funcionalidad)
            tk.Button(
                btn_frame, text="Ver detalles",
                bg="#5D6D7E", fg="white",
                cursor="hand2"
            ).pack(side="right", padx=5)

    # Inicializar la vista con todos los datos
    populate_cards(all_data)