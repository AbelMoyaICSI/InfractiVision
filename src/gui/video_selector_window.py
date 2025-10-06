"""
Selector Visual de Videos - Interfaz moderna con miniaturas y metadatos completos
Optimizado para rendimiento y experiencia de usuario
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import json
import threading
from PIL import Image, ImageTk

from src.path_helper import resource_path

class VideoSelectorWindow:
    """
    Ventana de selección visual de videos con miniaturas, metadatos y estado de configuración
    """
    
    def __init__(self, parent, video_dir, on_video_selected=None):
        self.parent = parent
        self.video_dir = video_dir
        self.on_video_selected = on_video_selected
        self.selected_video = None
        
        # Cache para miniaturas (optimización)
        self.thumbnail_cache = {}
        self.metadata_cache = {}
        
        # Configurar archivos de configuración
        self.config_files = {
            'polygon': resource_path("config/polygon_config.json"),
            'time_presets': resource_path("config/time_presets.json"),
            'avenue': resource_path("config/avenue_config.json")
        }
        
        # Crear ventana
        self.create_window()
        
        # Cargar videos en hilo separado para no bloquear UI
        self.load_videos_async()
    
    def create_window(self):
        """Crear la ventana principal del selector"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Selector Visual de Videos")
        
        # Configurar icono
        icon_path = resource_path("img/icon.ico")
        if os.path.exists(icon_path):
            self.window.iconbitmap(icon_path)
        self.window.geometry("1200x800")
        self.window.configure(bg='#f0f0f0')
        self.window.resizable(True, True)
        
        # Hacer ventana modal
        self.window.transient(self.parent)
        self.window.grab_set()
        self.center_window()
        
        # Frame principal con scrollbar
        self.main_frame = tk.Frame(self.window, bg='#f0f0f0')
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Título
        title_label = tk.Label(
            self.main_frame,
            text="🎬 Seleccionar Video para Configurar",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))
        
        # Frame con scrollbar para los videos
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='#f0f0f0', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#f0f0f0')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Habilitar scroll con rueda del mouse en toda la ventana
        self._bind_mousewheel_recursive(self.window)
        
        # Frame para botones de acción
        self.button_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        self.button_frame.pack(fill="x", pady=(20, 0))
        
        # Botón cancelar
        cancel_btn = tk.Button(
            self.button_frame,
            text="❌ Cancelar",
            command=self.cancel,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 12),
            padx=20,
            pady=10
        )
        cancel_btn.pack(side="right", padx=5)
        
        # Botón limpiar todo
        clean_all_btn = tk.Button(
            self.button_frame,
            text="🧹 Limpiar Todo",
            command=self.clean_all_configs,
            bg="#e67e22",
            fg="white",
            font=("Arial", 12),
            padx=20,
            pady=10
        )
        clean_all_btn.pack(side="right", padx=5)
        
        # Botón importar video
        import_btn = tk.Button(
            self.button_frame,
            text="📁 Importar Video",
            command=self.import_new_video,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 12),
            padx=20,
            pady=10
        )
        import_btn.pack(side="right", padx=5)
        
        # Botón refrescar
        refresh_btn = tk.Button(
            self.button_frame,
            text="🔄 Actualizar",
            command=self.refresh_videos,
            bg="#3498db",
            fg="white", 
            font=("Arial", 12),
            padx=20,
            pady=10
        )
        refresh_btn.pack(side="right", padx=5)
        
        # Label de estado
        self.status_label = tk.Label(
            self.button_frame,
            text="🔍 Cargando videos...",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        self.status_label.pack(side="left")
    

    def center_window(self):
        """Centrar ventana en la pantalla"""
        self.window.update_idletasks()
        width = 1200
        height = 800
        
        # Obtener dimensiones de la pantalla
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        
        # Calcular posición para centrar
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # Aplicar posición centrada
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def load_videos_async(self):
        """Cargar videos en hilo separado para optimizar rendimiento"""
        def load_thread():
            try:
                videos = self.get_video_files()
                # Actualizar UI en hilo principal
                self.window.after(0, lambda: self.display_videos(videos))
            except Exception as e:
                self.window.after(0, lambda: self.status_label.config(
                    text=f"❌ Error cargando videos: {str(e)}"
                ))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def get_video_files(self):
        """Obtener lista de archivos de video del directorio"""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        videos = []
        
        if os.path.exists(self.video_dir):
            for file in os.listdir(self.video_dir):
                if file.lower().endswith(video_extensions):
                    video_path = os.path.join(self.video_dir, file)
                    videos.append({
                        'filename': file,
                        'path': video_path,
                        'size': os.path.getsize(video_path)
                    })
        
        return sorted(videos, key=lambda x: x['filename'])
    
    def display_videos(self, videos):
        """Mostrar videos en grid visual"""
        if not videos:
            self.status_label.config(text="📁 No se encontraron videos en el directorio")
            no_videos_label = tk.Label(
                self.scrollable_frame,
                text="📂 No hay videos disponibles\n\nAgrega videos a la carpeta 'videos' para comenzar",
                font=("Arial", 14),
                bg='#f0f0f0',
                fg='#7f8c8d',
                pady=50
            )
            no_videos_label.pack(fill="both", expand=True)
            return
        
        self.status_label.config(text=f"📹 {len(videos)} video(s) encontrado(s)")
        
        # Grid de videos (3 columnas)
        columns = 3
        for idx, video in enumerate(videos):
            row = idx // columns
            col = idx % columns
            
            # Crear card para cada video
            self.create_video_card(video, row, col)
    
    def create_video_card(self, video, row, col):
        """Crear tarjeta visual para cada video"""
        # Frame principal de la tarjeta
        card_frame = tk.Frame(
            self.scrollable_frame,
            bg='white',
            relief='ridge',
            borderwidth=2,
            padx=10,
            pady=10
        )
        card_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        
        # Configurar grid weights para responsive design y alineación uniforme
        self.scrollable_frame.grid_columnconfigure(col, weight=1, uniform="cards")
        self.scrollable_frame.grid_rowconfigure(row, weight=1)
        
        # Thumbnail placeholder (se cargará async)
        thumbnail_frame = tk.Frame(card_frame, bg='#ecf0f1', width=200, height=120)
        thumbnail_frame.pack(pady=(0, 10))
        thumbnail_frame.pack_propagate(False)
        
        thumbnail_label = tk.Label(
            thumbnail_frame,
            text="🎬\nCargando...",
            bg='#ecf0f1',
            fg='#7f8c8d',
            font=("Arial", 10)
        )
        thumbnail_label.pack(expand=True)
        
        # Cargar thumbnail en hilo separado
        self.load_thumbnail_async(video, thumbnail_label)
        
        # Nombre del archivo
        name_label = tk.Label(
            card_frame,
            text=video['filename'],
            font=("Arial", 11, "bold"),
            bg='white',
            fg='#2c3e50',
            wraplength=180
        )
        name_label.pack(pady=(0, 5))
        
        # Información del video
        info_frame = tk.Frame(card_frame, bg='white')
        info_frame.pack(fill="x", pady=(0, 10))
        
        # Cargar y mostrar metadatos
        self.load_video_info_async(video, info_frame)
        
        # Estado de configuración
        status_frame = tk.Frame(card_frame, bg='white')
        status_frame.pack(fill="x", pady=(0, 10))
        
        self.load_config_status_async(video, status_frame)
        
        # Botones de acción
        button_frame = tk.Frame(card_frame, bg='white')
        button_frame.pack(fill="x")
        
        # Fila 1: Botones principales - Siempre seleccionables
        main_buttons_frame = tk.Frame(button_frame, bg='white')
        main_buttons_frame.pack(fill="x", pady=(0, 5))
        
        # Obtener estado de configuración
        config_status = self.get_configuration_status(video['filename'])
        total_configured = sum([config_status['polygon'], config_status['semaphore'], config_status['avenue']])
        is_fully_configured = total_configured == 3
        
        # Botón Seleccionar - Siempre habilitado, cambia el texto según configuración
        if is_fully_configured:
            select_btn = tk.Button(
                main_buttons_frame,
                text="✅ Reseleccionar",
                command=lambda v=video: self.select_video(v),
                bg="#27ae60",
                fg="white",
                font=("Arial", 9),
                pady=4
            )
            # Agregar tooltip explicativo
            self.create_tooltip(select_btn, 
                "Video ya configurado.\nClick para seleccionarlo nuevamente y procesarlo.")
        else:
            select_btn = tk.Button(
                main_buttons_frame,
                text="✅ Seleccionar",
                command=lambda v=video: self.select_video(v),
                bg="#27ae60",
                fg="white",
                font=("Arial", 9),
                pady=4
            )
        select_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))
        
        # Botón Configurar - Cambia texto si está completamente configurado
        if is_fully_configured:
            config_btn = tk.Button(
                main_buttons_frame,
                text="✏️ Editar Config",
                command=lambda v=video: self.configure_video(v),
                bg="#e67e22",
                fg="white",
                font=("Arial", 9),
                pady=4
            )
        else:
            config_btn = tk.Button(
                main_buttons_frame,
                text="⚙️ Configurar",
                command=lambda v=video: self.configure_video(v),
                bg="#3498db",
                fg="white",
                font=("Arial", 9),
                pady=4
            )
        config_btn.pack(side="right", fill="x", expand=True, padx=(2, 0))
        
        # Fila 2: Botones de gestión
        manage_buttons_frame = tk.Frame(button_frame, bg='white')
        manage_buttons_frame.pack(fill="x")
        
        clean_btn = tk.Button(
            manage_buttons_frame,
            text="🧹 Limpiar",
            command=lambda v=video: self.clean_video_config(v),
            bg="#f39c12",
            fg="white",
            font=("Arial", 8),
            pady=3
        )
        clean_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))
        
        delete_btn = tk.Button(
            manage_buttons_frame,
            text="🗑️ Eliminar",
            command=lambda v=video: self.delete_video(v),
            bg="#e74c3c",
            fg="white",
            font=("Arial", 8),
            pady=3
        )
        delete_btn.pack(side="right", fill="x", expand=True, padx=(2, 0))
    
    def load_thumbnail_async(self, video, label):
        """Cargar miniatura del video en hilo separado"""
        def load_thumb():
            try:
                if video['filename'] in self.thumbnail_cache:
                    thumb = self.thumbnail_cache[video['filename']]
                else:
                    thumb = self.generate_thumbnail(video['path'])
                    self.thumbnail_cache[video['filename']] = thumb
                
                # Actualizar UI en hilo principal
                self.window.after(0, lambda: label.config(
                    image=thumb,
                    text="",
                    compound="center"
                ))
                # Mantener referencia para evitar garbage collection
                label.image = thumb
                
            except Exception as e:
                self.window.after(0, lambda: label.config(
                    text=f"❌\nError\ncargando",
                    fg="#e74c3c"
                ))
        
        threading.Thread(target=load_thumb, daemon=True).start()
    
    def generate_thumbnail(self, video_path):
        """Generar miniatura del video"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Buscar un frame representativo (no el primero que puede estar negro)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_frame = min(30, frame_count // 4)  # Frame del primer cuarto
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Redimensionar frame
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                
                if aspect_ratio > 1:  # Video horizontal
                    new_width = 180
                    new_height = int(180 / aspect_ratio)
                else:  # Video vertical
                    new_height = 100
                    new_width = int(100 * aspect_ratio)
                
                frame = cv2.resize(frame, (new_width, new_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convertir a PIL y luego a PhotoImage
                pil_image = Image.fromarray(frame)
                return ImageTk.PhotoImage(pil_image)
            
        except Exception as e:
            print(f"Error generando thumbnail para {video_path}: {e}")
        
        # Thumbnail por defecto si hay error
        return self.create_default_thumbnail()
    
    def create_default_thumbnail(self):
        """Crear miniatura por defecto"""
        # Crear imagen simple con PIL
        img = Image.new('RGB', (180, 100), color='#bdc3c7')
        return ImageTk.PhotoImage(img)
    
    def load_video_info_async(self, video, info_frame):
        """Cargar información del video en hilo separado"""
        def load_info():
            try:
                if video['filename'] in self.metadata_cache:
                    metadata = self.metadata_cache[video['filename']]
                else:
                    metadata = self.get_video_metadata(video['path'])
                    self.metadata_cache[video['filename']] = metadata
                
                # Actualizar UI en hilo principal
                self.window.after(0, lambda: self.display_video_info(info_frame, metadata, video))
                
            except Exception as e:
                self.window.after(0, lambda: tk.Label(
                    info_frame,
                    text="❌ Error cargando info",
                    font=("Arial", 8),
                    bg='white',
                    fg="#e74c3c"
                ).pack())
        
        threading.Thread(target=load_info, daemon=True).start()
    
    def get_video_metadata(self, video_path):
        """Obtener metadatos del video"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calcular duración con soporte completo para H:MM:SS
            duration_seconds = frame_count / fps if fps > 0 else 0
            
            if duration_seconds >= 3600:  # ≥ 1 hora
                hours = int(duration_seconds // 3600)
                remaining = duration_seconds % 3600
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                duration_formatted = f"{hours}:{minutes:02d}:{seconds:02d}"
            else:  # < 1 hora
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                duration_formatted = f"{minutes:02d}:{seconds:02d}"
            
            cap.release()
            
            return {
                'duration': duration_formatted,
                'resolution': f"{width}x{height}",
                'fps': f"{fps:.1f}" if fps > 0 else "N/A",
                'frames': frame_count
            }
            
        except Exception as e:
            return {
                'duration': "N/A",
                'resolution': "N/A", 
                'fps': "N/A",
                'frames': 0
            }
    
    def display_video_info(self, info_frame, metadata, video):
        """Mostrar información del video"""
        # Duración
        duration_label = tk.Label(
            info_frame,
            text=f"⏱️ {metadata['duration']}",
            font=("Arial", 9),
            bg='white',
            fg='#34495e'
        )
        duration_label.pack(anchor="w")
        
        # Resolución
        res_label = tk.Label(
            info_frame,
            text=f"📐 {metadata['resolution']}",
            font=("Arial", 9),
            bg='white',
            fg='#34495e'
        )
        res_label.pack(anchor="w")
        
        # Tamaño de archivo
        size_mb = video['size'] / (1024 * 1024)
        size_label = tk.Label(
            info_frame,
            text=f"💾 {size_mb:.1f} MB",
            font=("Arial", 9),
            bg='white',
            fg='#34495e'
        )
        size_label.pack(anchor="w")
    
    def load_config_status_async(self, video, status_frame):
        """Cargar estado de configuración en hilo separado"""
        def load_status():
            try:
                status = self.get_configuration_status(video['filename'])
                # Actualizar UI en hilo principal
                self.window.after(0, lambda: self.display_config_status(status_frame, status))
            except Exception as e:
                self.window.after(0, lambda: tk.Label(
                    status_frame,
                    text="❌ Error estado",
                    font=("Arial", 8),
                    bg='white',
                    fg="#e74c3c"
                ).pack())
        
        threading.Thread(target=load_status, daemon=True).start()
    
    def get_configuration_status(self, filename):
        """Obtener estado de configuración del video"""
        status = {
            'polygon': False,
            'semaphore': False,
            'avenue': False,
            'semaphore_data': None
        }
        
        # Verificar polígono
        try:
            if os.path.exists(self.config_files['polygon']):
                with open(self.config_files['polygon'], 'r', encoding='utf-8') as f:
                    polygon_config = json.load(f)
                    if filename in polygon_config and polygon_config[filename]:
                        status['polygon'] = True
        except:
            pass
        
        # Verificar semáforo
        try:
            if os.path.exists(self.config_files['time_presets']):
                with open(self.config_files['time_presets'], 'r', encoding='utf-8') as f:
                    semaphore_config = json.load(f)
                    if filename in semaphore_config:
                        status['semaphore'] = True
                        status['semaphore_data'] = semaphore_config[filename]
        except:
            pass
        
        # Verificar avenida
        try:
            if os.path.exists(self.config_files['avenue']):
                with open(self.config_files['avenue'], 'r', encoding='utf-8') as f:
                    avenue_config = json.load(f)
                    if filename in avenue_config and avenue_config[filename]:
                        status['avenue'] = True
                        status['avenue_name'] = avenue_config[filename]
        except:
            pass
        
        return status
    
    def display_config_status(self, status_frame, status):
        """Mostrar estado de configuración"""
        # Estado del polígono
        polygon_status = "✅" if status['polygon'] else "❌"
        polygon_label = tk.Label(
            status_frame,
            text=f"{polygon_status} Área restrictiva",
            font=("Arial", 8),
            bg='white',
            fg='#27ae60' if status['polygon'] else '#e74c3c'
        )
        polygon_label.pack(anchor="w")
        
        # Estado del semáforo
        semaphore_status = "✅" if status['semaphore'] else "❌"
        semaphore_text = f"{semaphore_status} Semáforo"
        if status['semaphore'] and status['semaphore_data']:
            data = status['semaphore_data']
            semaphore_text += f" (🟢{data.get('green',0)}s 🟡{data.get('yellow',0)}s 🔴{data.get('red',0)}s)"
        
        semaphore_label = tk.Label(
            status_frame,
            text=semaphore_text,
            font=("Arial", 8),
            bg='white',
            fg='#27ae60' if status['semaphore'] else '#e74c3c',
            wraplength=170
        )
        semaphore_label.pack(anchor="w")
        
        # Estado de la avenida
        avenue_status = "✅" if status['avenue'] else "❌"
        avenue_text = f"{avenue_status} Ubicación"
        if status['avenue']:
            avenue_text += f": {status.get('avenue_name', 'N/A')}"
        
        avenue_label = tk.Label(
            status_frame,
            text=avenue_text,
            font=("Arial", 8),
            bg='white',
            fg='#27ae60' if status['avenue'] else '#e74c3c',
            wraplength=170
        )
        avenue_label.pack(anchor="w")
        
        # Estado general
        total_configured = sum([status['polygon'], status['semaphore'], status['avenue']])
        if total_configured == 3:
            overall_status = "🎯 Completamente configurado"
            color = '#27ae60'
        elif total_configured > 0:
            overall_status = f"⚠️ Parcialmente configurado ({total_configured}/3)"
            color = '#f39c12'
        else:
            overall_status = "❌ Sin configurar"
            color = '#e74c3c'
        
        overall_label = tk.Label(
            status_frame,
            text=overall_status,
            font=("Arial", 8, "bold"),
            bg='white',
            fg=color,
            wraplength=170
        )
        overall_label.pack(anchor="w", pady=(5, 0))
    
    def select_video(self, video):
        """Seleccionar video y cerrar ventana"""
        self.selected_video = video
        if self.on_video_selected:
            self.on_video_selected(video['path'])
        self.window.destroy()
    

    
    def configure_video(self, video):
        """Configurar video completo (polígono + semáforo + ubicación)"""
        try:
            # Cerrar selector y abrir configuración completa
            if self.on_video_selected:
                # Llamar con modo de configuración
                self.window.destroy()
                self.on_video_selected(video['path'], force_config=True)
        except Exception as e:
            messagebox.showerror("Error", f"Error configurando video: {str(e)}", parent=self.window)
    
    def clean_video_config(self, video):
        """Limpiar configuración de video específico"""
        try:
            response = messagebox.askyesno(
                "Confirmar limpieza",
                f"¿Limpiar toda la configuración de '{video['filename']}'?\n\n"
                "Esto eliminará:\n• Área restrictiva (polígono)\n• Tiempos de semáforo\n• Ubicación asignada\n\n"
                "El archivo de video se mantendrá.",
                parent=self.window
            )
            
            if response:
                self._clean_single_video_config(video['filename'])
                messagebox.showinfo("Limpieza completa", f"Configuración de '{video['filename']}' eliminada.", parent=self.window)
                self.refresh_videos()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error limpiando configuración: {str(e)}", parent=self.window)
    
    def delete_video(self, video):
        """Eliminar video y toda su configuración"""
        try:
            response = messagebox.askyesno(
                "Confirmar eliminación",
                f"¿Eliminar '{video['filename']}' completamente?\n\n"
                "Esto eliminará:\n• El archivo de video\n• Toda su configuración\n• Datos de infracciones asociados\n\n"
                "Esta acción NO se puede deshacer.",
                parent=self.window
            )
            
            if response:
                # Eliminar archivo de video
                if os.path.exists(video['path']):
                    os.remove(video['path'])
                
                # Limpiar configuraciones
                self._clean_single_video_config(video['filename'])
                
                # Limpiar datos de infracciones
                self._clean_video_infractions(video['filename'])
                
                messagebox.showinfo("Eliminación completa", f"'{video['filename']}' eliminado completamente.", parent=self.window)
                self.refresh_videos()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error eliminando video: {str(e)}", parent=self.window)
    
    def import_new_video(self):
        """Importar nuevo video desde explorador de archivos"""
        try:
            from tkinter import filedialog
            
            file_path = filedialog.askopenfilename(
                title="Importar nuevo video",
                filetypes=[
                    ("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                    ("Todos los archivos", "*.*")
                ],
                parent=self.window
            )
            
            if file_path:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.video_dir, filename)
                
                # Verificar si ya existe
                if os.path.exists(dest_path):
                    response = messagebox.askyesno(
                        "Video existente",
                        f"El video '{filename}' ya existe.\n¿Reemplazarlo?",
                        parent=self.window
                    )
                    if not response:
                        return
                
                # Copiar archivo
                import shutil
                shutil.copy2(file_path, dest_path)
                
                messagebox.showinfo(
                    "Video importado",
                    f"'{filename}' importado exitosamente.\n\n¿Desea configurarlo ahora?",
                    parent=self.window
                )
                
                # Refrescar lista
                self.refresh_videos()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error importando video: {str(e)}", parent=self.window)
    
    def clean_all_configs(self):
        """Limpiar todas las configuraciones de todos los videos"""
        try:
            response = messagebox.askyesno(
                "Confirmar limpieza total",
                "¿Limpiar TODAS las configuraciones?\n\n"
                "Esto eliminará:\n• Todos los polígonos\n• Todos los tiempos de semáforo\n• Todas las ubicaciones\n• Todos los datos de infracciones\n\n"
                "Los archivos de video se mantendrán.\n\nEsta acción NO se puede deshacer.",
                parent=self.window
            )
            
            if response:
                # Limpiar todos los archivos de configuración
                config_files = [
                    self.config_files['polygon'],
                    self.config_files['time_presets'],
                    self.config_files['avenue'],
                    resource_path("data/infracciones.json"),
                    resource_path("data/indicadores_rendimiento.json")
                ]
                
                for config_file in config_files:
                    if os.path.exists(config_file):
                        with open(config_file, 'w', encoding='utf-8') as f:
                            json.dump({}, f, indent=2)
                
                # Limpiar directorios de salida
                output_dirs = [
                    resource_path("data/output/placas"),
                    resource_path("data/output/autos")
                ]
                
                for output_dir in output_dirs:
                    if os.path.exists(output_dir):
                        import shutil
                        shutil.rmtree(output_dir)
                        os.makedirs(output_dir, exist_ok=True)
                
                messagebox.showinfo("Limpieza completa", "Todas las configuraciones han sido eliminadas.", parent=self.window)
                self.refresh_videos()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error limpiando configuraciones: {str(e)}", parent=self.window)
    
    def _clean_single_video_config(self, filename):
        """Limpiar configuración de un video específico"""
        print(f"🧹 Limpiando configuración para: {filename}")
        # Limpiar de archivos de configuración
        for config_type, config_path in self.config_files.items():
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    if filename in config_data:
                        del config_data[filename]
                        print(f"✅ Eliminada configuración {config_type} para {filename}")
                        
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=2)
                except Exception as e:
                    print(f"⚠️ Error limpiando configuración {config_type}: {e}")
                    pass
    
    def _clean_video_infractions(self, filename):
        """Limpiar datos de infracciones de un video específico"""
        print(f"🧹 Limpiando infracciones para: {filename}")
        try:
            infractions_file = resource_path("data/infracciones.json")
            if os.path.exists(infractions_file):
                with open(infractions_file, 'r', encoding='utf-8') as f:
                    infractions = json.load(f)
                
                original_count = len(infractions)
                
                # Filtrar infracciones que no pertenezcan a este video
                # Nota: Los registros de infracciones no tienen campo 'video', pero pueden tener
                # información del video en otros campos como 'ubicacion' o metadata
                filtered_infractions = []
                removed_count = 0
                
                for inf in infractions:
                    # Verificar si la infracción está relacionada con este video
                    # (por ahora, conservamos todas las infracciones ya que no hay relación directa)
                    # En el futuro se podría agregar un campo 'video_source' a las infracciones
                    should_remove = False
                    
                    # Solo remover si hay evidencia clara de relación con el video
                    # Por ejemplo, si hay timestamps específicos o metadatos del video
                    if inf.get('video_source', '').endswith(filename):
                        should_remove = True
                    
                    if not should_remove:
                        filtered_infractions.append(inf)
                    else:
                        removed_count += 1
                
                print(f"📊 Infracciones: {original_count} total, {removed_count} eliminadas, {len(filtered_infractions)} conservadas")
                
                with open(infractions_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_infractions, f, indent=2)
                    
        except Exception as e:
            print(f"⚠️ Error limpiando infracciones: {e}")
            pass
    
    def refresh_videos(self):
        """Actualizar lista de videos"""
        # Limpiar cache
        self.thumbnail_cache.clear()
        self.metadata_cache.clear()
        
        # Limpiar grid actual
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Recargar videos
        self.status_label.config(text="🔄 Actualizando...")
        self.load_videos_async()
    
    def _bind_mousewheel_recursive(self, widget):
        """Vincular scroll con rueda del mouse recursivamente a todos los widgets"""
        # Vincular eventos de scroll al widget actual
        widget.bind("<MouseWheel>", self._on_mousewheel)
        widget.bind("<Button-4>", self._on_mousewheel)  # Linux scroll up
        widget.bind("<Button-5>", self._on_mousewheel)  # Linux scroll down
        
        # Aplicar recursivamente a todos los widgets hijos
        for child in widget.winfo_children():
            self._bind_mousewheel_recursive(child)
    
    def _on_mousewheel(self, event):
        """Manejar scroll con rueda del mouse"""
        # Verificar si el canvas tiene contenido scrolleable
        if self.canvas.winfo_exists():
            try:
                # Windows y MacOS
                if hasattr(event, 'delta') and event.delta:
                    delta = -1 * (event.delta / 120)
                # Linux
                elif hasattr(event, 'num'):
                    if event.num == 4:
                        delta = -1
                    elif event.num == 5:
                        delta = 1
                    else:
                        delta = 0
                else:
                    delta = 0
                
                # Scroll suave (3 líneas por scroll)
                self.canvas.yview_scroll(int(delta * 3), "units")
            except:
                pass
    
    def cancel(self):
        """Cancelar selección"""
        self.selected_video = None
        self.window.destroy()
    
    def get_selected_video(self):
        """Obtener video seleccionado"""
        return self.selected_video
    
    def create_tooltip(self, widget, text):
        """Crear tooltip para un widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            tooltip.configure(bg="#2c3e50")
            
            label = tk.Label(
                tooltip,
                text=text,
                font=("Arial", 9),
                bg="#2c3e50",
                fg="white",
                padx=8,
                pady=4,
                justify="left"
            )
            label.pack()
            
            # Guardar referencia para poder destruirlo
            widget.tooltip = tooltip
        
        def hide_tooltip(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        # Vincular eventos
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)


# Función de utilidad para uso externo
def show_video_selector(parent, video_dir, on_video_selected=None):
    """
    Mostrar selector de videos y retornar el video seleccionado
    
    Args:
        parent: Ventana padre
        video_dir: Directorio de videos
        on_video_selected: Callback cuando se selecciona un video
    
    Returns:
        Ruta del video seleccionado o None si se cancela
    """
    selector = VideoSelectorWindow(parent, video_dir, on_video_selected)
    parent.wait_window(selector.window)
    return selector.get_selected_video()
