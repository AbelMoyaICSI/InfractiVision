import tkinter as tk
from tkinter import messagebox
import json, os

INF_FILE = os.path.join("data", "infracciones.json")

def create_infractions_window(window: tk.Toplevel, back_callback):
    window.state("zoomed")
    window.configure(bg="white")

    tk.Label(window, text="Gestión de Infracciones", font=("Arial", 28, "bold"),
             bg="white", fg="#273D86").pack(pady=20)

    lb = tk.Listbox(window, font=("Arial", 14), width=80, height=22)
    lb.pack(pady=10)
    lb.delete(0, tk.END)

    if os.path.exists(INF_FILE):
        try:
            data = json.load(open(INF_FILE, "r", encoding="utf-8"))
            for inf in data:
                line = f"{inf.get('placa','N/A')} – {inf.get('fecha','')} – {inf.get('tipo','')}"
                lb.insert(tk.END, line)
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando infracciones: {e}")
    else:
        lb.insert(tk.END, "No se encontraron infracciones.")

    btn_f = tk.Frame(window, bg="white")
    btn_f.pack(pady=10)
    tk.Button(btn_f, text="Volver", font=("Arial", 14), command=back_callback).pack(side="left", padx=10)
    tk.Button(btn_f, text="Descargar Evidencia", font=("Arial", 14),
              command=lambda: messagebox.showinfo("Pendiente", "Función en desarrollo")).pack(side="left", padx=10)