# src/gui/welcome_window.py

import tkinter as tk
from PIL import Image, ImageTk
from src.path_helper import resource_path
from src.gui.manual_window import show_manual

class WelcomeFrame(tk.Frame):
    def __init__(self, master, app_manager):
        super().__init__(master, bg="#273D86")
        self.app_manager = app_manager
        
        # Control de redimensionado
        self._resize_job = None
        self._pil_image_original = None   # Imagen original (sin redimensionar)
        self._bg_image_tk = None          # Última imagen Tk generada
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.create_widgets()
        
        # Cargar imagen inmediatamente (sin after) pero en un hilo seguro
        self.cargar_imagen_inicial()
    
    def destroy(self):
        # Cancelar cualquier redimensionado pendiente
        if self._resize_job:
            self.after_cancel(self._resize_job)
            self._resize_job = None
        # Liberar referencias a imágenes
        self._pil_image_original = None
        self._bg_image_tk = None
        super().destroy()
    
    def cargar_imagen_inicial(self):
        """Carga la imagen original y la redimensiona por primera vez."""
        if not self.winfo_exists():
            return
        bg_path = resource_path("img/welcome_bg.png")
        try:
            # Abrir imagen y limitar su tamaño máximo a 1920x1080 para evitar consumir mucha memoria
            img = Image.open(bg_path)
            # Si la imagen es más grande que Full HD, la reducimos preventivamente
            max_width, max_height = 1920, 1080
            if img.width > max_width or img.height > max_height:
                img.thumbnail((max_width, max_height), Image.LANCZOS)
                print(f"🖼️ Imagen reducida preventivamente a {img.width}x{img.height}")
            self._pil_image_original = img
            print(f"✅ Imagen original cargada desde: {bg_path}")
            # Realizar el primer redimensionado
            self.redimensionar_imagen()
        except Exception as e:
            print(f"❌ Error cargando imagen: {e}")
            self._pil_image_original = None
            if self.winfo_exists() and hasattr(self, 'left_frame'):
                tk.Label(self.left_frame, text="InfractiVision", font=("Arial", 28, "bold"),
                         bg="#273D86", fg="white").pack(expand=True)
    
    def on_left_frame_resize(self, event):
        """Maneja el evento de cambio de tamaño del panel izquierdo con retardo."""
        if self._resize_job:
            return
        # Esperar 200ms para evitar redimensionados excesivos
        self._resize_job = self.after(200, self.redimensionar_imagen)
    
    def redimensionar_imagen(self):
        """Redimensiona la imagen al tamaño actual del panel izquierdo."""
        self._resize_job = None
        if not self.winfo_exists():
            return
        if self._pil_image_original is None:
            return
        ancho = self.left_frame.winfo_width()
        alto = self.left_frame.winfo_height()
        if ancho <= 10 or alto <= 10:
            return
        
        try:
            # Redimensionar usando LANCZOS (alta calidad)
            img_resized = self._pil_image_original.resize((ancho, alto), Image.LANCZOS)
            # Generar PhotoImage
            nueva_imagen = ImageTk.PhotoImage(img_resized)
            # Actualizar label
            if hasattr(self, 'bg_label') and self.bg_label.winfo_exists():
                self.bg_label.config(image=nueva_imagen)
                self.bg_label.image = nueva_imagen   # Guardar referencia
                self.bg_label.place(x=0, y=0, width=ancho, height=alto)
            # Guardar la nueva imagen Tk para que no sea recolectada
            self._bg_image_tk = nueva_imagen
        except Exception as e:
            print(f"Error redimensionando imagen: {e}")
    
    def create_widgets(self):
        # Panel izquierdo (contenedor de la imagen)
        self.left_frame = tk.Frame(self, bg="#273D86")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        # Etiqueta temporal mientras se carga la imagen
        self.temp_label = tk.Label(self.left_frame, text="Cargando...", bg="#273D86", fg="white")
        self.temp_label.pack(expand=True)
        
        # Etiqueta que contendrá la imagen (se crea ahora y se actualizará)
        self.bg_label = tk.Label(self.left_frame, bg="#273D86")
        self.bg_label.place(x=0, y=0)
        
        # Vincular el evento de redimensionado
        self.left_frame.bind("<Configure>", self.on_left_frame_resize)
        
        # Panel derecho (botones y textos)
        right = tk.Frame(self, bg="white")
        right.grid(row=0, column=1, sticky="nsew")
        content = tk.Frame(right, bg="white")
        content.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(content, text="Bienvenido a\nInfractiVision",
                 font=("Arial", 40, "bold"), bg="white", fg="#3366FF",
                 justify="center").pack(pady=(0,10))
        
        tk.Label(content, text="Selecciona la opción para continuar",
                 font=("Arial", 18), bg="white", fg="gray20",
                 justify="center").pack(pady=(0,20))
        
        manual_btn = tk.Button(content, text="Manual de Usuario",
                               command=self.show_manual,
                               bg="#9b59b6", fg="white", font=("Arial", 14, "bold"),
                               padx=30, pady=8, bd=0)
        manual_btn.pack(pady=(0,20))
        
        btns = tk.Frame(content, bg="white")
        btns.pack()
        
        def mk(txt, cmd):
            return tk.Button(btns, text=txt, font=("Arial",16),
                             bg="#3366FF", fg="white",
                             bd=0, padx=20, pady=10,
                             command=cmd)
        
        mk("Foto Rojo", self.app_manager.open_violation_window).pack(side="left", padx=10)
        mk("Gestión de Infracciones", self.app_manager.open_infractions_window).pack(side="left", padx=10)
    
    def show_manual(self):
        show_manual(self.master)