# main.py
import tkinter as tk
from src.gui.app_manager import AppManager

def main():
    root = tk.Tk()
    root.title("InfractiVision")
    root.geometry("1280x720")
    root.state("zoomed")

    # Instanciamos el gestor de la app
    app = AppManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
