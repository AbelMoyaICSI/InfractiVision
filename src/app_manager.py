import tkinter as tk
from src.welcome import WelcomeFrame
from src.foto_rojo_window import create_foto_rojo_content
from src.gestion_infracciones import create_gestion_infracciones_content

class AppManager:
    def __init__(self, root):
        self.root = root
        self.root.state("zoomed")  
        self.current_window = None

    def show_welcome(self):
        if self.current_window is not None:
            self.current_window.destroy()
            self.current_window = None
        self.root.deiconify()     
        self.root.state("zoomed")
        for widget in self.root.winfo_children():
            widget.destroy()
        welcome = WelcomeFrame(self.root, self)
        welcome.pack(fill="both", expand=True)

    def open_foto_rojo_window(self):
        self.root.withdraw()
        foto_window = tk.Toplevel(self.root)
        foto_window.title("InfractiVision - Foto Rojo")
        foto_window.geometry("1280x720")
        foto_window.state("zoomed")
        foto_window.protocol("WM_DELETE_WINDOW", self.show_welcome)
        create_foto_rojo_content(foto_window, self.show_welcome)
        self.current_window = foto_window

    def open_gestion_infracciones_window(self):
        self.root.withdraw()
        gestion_window = tk.Toplevel(self.root)
        gestion_window.title("InfractiVision - Gestión de Infracciones")
        gestion_window.geometry("1280x720")
        gestion_window.state("zoomed")
        gestion_window.protocol("WM_DELETE_WINDOW", self.show_welcome)
        create_gestion_infracciones_content(gestion_window, self.show_welcome)
        self.current_window = gestion_window
