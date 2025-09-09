# src/gui/welcome_window.py

import tkinter as tk
import os
from PIL import Image, ImageTk
from src.path_helper import resource_path
from src.gui.manual_window import show_manual

class WelcomeFrame(tk.Frame):
    def __init__(self, master, app_manager):
        super().__init__(master, bg="#273D86")
        self.app_manager = app_manager
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.create_widgets()

    def create_widgets(self):
        # Panel izquierdo: imagen
        left = tk.Frame(self, bg="#273D86")
        left.grid(row=0, column=0, sticky="nsew")
        
        # Usar resource_path para que funcione en el ejecutable
        bg_path = resource_path("img/welcome_bg.png")
        
        try:
            img_orig = Image.open(bg_path)
            lbl = tk.Label(left)
            lbl.place(relwidth=1, relheight=1)
            def resize(e):
                img = img_orig.resize((e.width, e.height), Image.LANCZOS)
                self._tk = ImageTk.PhotoImage(img)
                lbl.config(image=self._tk)
            left.bind("<Configure>", resize)
            print(f"✅ Imagen de bienvenida cargada desde: {bg_path}")
        except Exception as e:
            print(f"❌ Error cargando imagen de bienvenida: {e}")
            print(f"   Ruta intentada: {bg_path}")
            tk.Label(left, text="[Imagen no disponible]", bg="#273D86", fg="white").pack(expand=True)

        # Panel derecho: títulos + botones
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
        
        # Botón Manual
        manual_btn = tk.Button(content, text="📖 Manual de Usuario", 
                              command=self.show_manual,
                              bg="#9b59b6", fg="white", font=("Arial", 14, "bold"),
                              padx=30, pady=8, cursor="hand2", bd=0)
        manual_btn.pack(pady=(0,20))

        btns = tk.Frame(content, bg="white")
        btns.pack()
        def mk(txt, cmd):
            return tk.Button(btns, text=txt, font=("Arial",16),
                             bg="#3366FF", fg="white",
                             activebackground="#2554FF", bd=0,
                             padx=20, pady=10, cursor="hand2",
                             command=cmd)
        # FOTO ROJO primero a la izquierda
        mk("Foto Rojo", self.app_manager.open_violation_window).pack(side="left", padx=10)
        mk("Gestión de Infracciones", self.app_manager.open_infractions_window).pack(side="left", padx=10)
    
    def show_manual(self):
        """Mostrar manual de usuario"""
        show_manual(self.master)
