import tkinter as tk
from tkinter import messagebox
import json
import os

def create_gestion_infracciones_content(window, back_callback):
    window.state("zoomed")  # Maximiza la ventana
    window.configure(bg="white")
    title = tk.Label(window, text="Gestión de Infracciones", font=("Arial", 28, "bold"), bg="white", fg="#273D86")
    title.pack(pady=20)
    # Listbox para mostrar infracciones
    infra_listbox = tk.Listbox(window, font=("Arial", 14), width=80, height=20)
    infra_listbox.pack(pady=10)
    infr_file = os.path.join("data", "infracciones.json")
    infra_listbox.delete(0, tk.END)
    if os.path.exists(infr_file):
        try:
            with open(infr_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for infr in data:
                line = f"{infr.get('placa','N/A')} - {infr.get('fecha','')} - {infr.get('tipo','')}"
                infra_listbox.insert(tk.END, line)
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar infracciones: {e}")
    else:
        infra_listbox.insert(tk.END, "No se encontraron infracciones.")
    btn_frame = tk.Frame(window, bg="white")
    btn_frame.pack(pady=10)
    back_button = tk.Button(btn_frame, text="Volver a Principal", font=("Arial", 14), command=back_callback)
    back_button.pack(side="left", padx=10)
    download_button = tk.Button(btn_frame, text="Descargar Evidencia", font=("Arial", 14),
                                command=lambda: messagebox.showinfo("Descargar", "Funcionalidad pendiente."))
    download_button.pack(side="left", padx=10)
