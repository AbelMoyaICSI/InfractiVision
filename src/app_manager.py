import tkinter as tk
from src.gui.welcome_window import WelcomeFrame
from src.gui.red_light_violation_window import create_violation_window
from src.gui.infractions_management_window import create_infractions_window


class AppManager:
    """Centraliza la navegación entre pantallas GUI"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_window = None
        self.root.state("zoomed")

    # ---------- helpers ----------
    def _clear_root(self):
        for w in self.root.winfo_children():
            w.destroy()

    # ---------- public api ----------
    def show_welcome(self):
        if self.current_window is not None:
            self.current_window.destroy()
            self.current_window = None
        self.root.deiconify()
        self.root.state("zoomed")
        self._clear_root()
        welcome = WelcomeFrame(self.root, self)
        welcome.pack(fill="both", expand=True)

    def open_violation_window(self):
        self.root.withdraw()
        win = tk.Toplevel(self.root)
        win.title("InfractiVision – Detección de Placas")
        win.geometry("1280x720")
        win.state("zoomed")
        win.protocol("WM_DELETE_WINDOW", self.show_welcome)
        create_violation_window(win, self.show_welcome)
        self.current_window = win

    def open_infractions_window(self):
        self.root.withdraw()
        win = tk.Toplevel(self.root)
        win.title("InfractiVision – Gestión de Infracciones")
        win.geometry("1280x720")
        win.state("zoomed")
        win.protocol("WM_DELETE_WINDOW", self.show_welcome)
        create_infractions_window(win, self.show_welcome)
        self.current_window = win
