import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox
import json  # Para guardar y cargar los presets de tiempos

PRESETS_FILE = "time_presets.json"

class Semaforo:
    """
    Panel de semáforo a la izquierda.
    Ciclo simple: green -> yellow -> red, con tiempos fijos (30, 3, 30) por defecto,
    pero configurable mediante presets asociados a una avenida o lugar (video).
    """
    def __init__(self, parent):
        self.parent = parent
        self.frame = tk.Frame(parent, bg='white')
        self.frame.pack(side="top", fill="both", expand=True)

        self.canvas = tk.Canvas(self.frame, bg='white', highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.info_label = tk.Label(self.frame, text="", font=("Arial", 14), bg='white')
        self.info_label.pack(pady=(0, 10))

        self.btn_tiempos = tk.Button(
            self.frame, text="Configurar Tiempos",
            command=self.gestionar_tiempos, width=20
        )
        self.btn_tiempos.pack(pady=5)

        # Carcasa y luces
        self.housing_rect = self.canvas.create_rectangle(0, 0, 0, 0,
                                                         fill="black", outline="gray", width=4)
        self.red_light = self.canvas.create_oval(0, 0, 0, 0,
                                                 fill="grey", outline="white", width=2)
        self.yellow_light = self.canvas.create_oval(0, 0, 0, 0,
                                                    fill="grey", outline="white", width=2)
        self.green_light = self.canvas.create_oval(0, 0, 0, 0,
                                                   fill="grey", outline="white", width=2)

        self.canvas.bind("<Configure>", self.resize_canvas)

        self.current_state = "green"
        self.cycle_durations = {"green": 30, "yellow": 3, "red": 30}
        self.target_time = time.time() + self.cycle_durations[self.current_state]

        self.show_state()
        self.update_countdown()

    def gestionar_tiempos(self):
        # Ventana modal para configurar, editar y eliminar presets.
        config_window = tk.Toplevel(self.parent)
        config_window.title("Configurar Tiempos - Presets")

        # Etiquetas y Entradas
        tk.Label(config_window, text="Nombre de la Avenida:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        avenue_entry = tk.Entry(config_window, width=30)
        avenue_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(config_window, text="Tiempo Verde (s):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        green_entry = tk.Entry(config_window, width=10)
        green_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(config_window, text="Tiempo Amarillo (s):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        yellow_entry = tk.Entry(config_window, width=10)
        yellow_entry.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(config_window, text="Tiempo Rojo (s):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        red_entry = tk.Entry(config_window, width=10)
        red_entry.grid(row=3, column=1, padx=5, pady=5)

        tk.Label(config_window, text="Presets guardados:").grid(row=4, column=0, columnspan=2, padx=5, pady=(10,0))

        # Frame para listbox + scrollbars
        list_frame = tk.Frame(config_window)
        list_frame.grid(row=5, column=0, columnspan=2, sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        presets_listbox = tk.Listbox(list_frame, width=50)
        presets_listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar_y = tk.Scrollbar(list_frame, orient="vertical", command=presets_listbox.yview)
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        presets_listbox.config(yscrollcommand=scrollbar_y.set)

        # Barra de desplazamiento horizontal
        scrollbar_x = tk.Scrollbar(config_window, orient="horizontal", command=presets_listbox.xview)
        scrollbar_x.grid(row=6, column=0, columnspan=2, sticky="ew")
        presets_listbox.config(xscrollcommand=scrollbar_x.set)

        def load_presets():
            try:
                with open(PRESETS_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return {}

        def update_listbox():
            presets = load_presets()
            presets_listbox.delete(0, tk.END)
            for avenue, times in presets.items():
                verde_val = times.get("green", 30)
                amarillo_val = times.get("yellow", 3)
                rojo_val = times.get("red", 30)
                # Formato: "Avenida: Verde=15, Amarillo=3, Rojo=20"
                line = f"{avenue}: Verde={verde_val}, Amarillo={amarillo_val}, Rojo={rojo_val}"
                presets_listbox.insert(tk.END, line)

        update_listbox()

        def save_preset():
            avenue = avenue_entry.get().strip()
            try:
                green_time = int(green_entry.get().strip())
                yellow_time = int(yellow_entry.get().strip())
                red_time = int(red_entry.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Los tiempos deben ser números enteros.", parent=config_window)
                return
            if not avenue:
                messagebox.showerror("Error", "Debe ingresar el nombre de la avenida.", parent=config_window)
                return
            presets = load_presets()
            presets[avenue] = {"green": green_time, "yellow": yellow_time, "red": red_time}
            with open(PRESETS_FILE, "w") as f:
                json.dump(presets, f, indent=2)
            # No mostramos el popup cada vez, solo si no existía
            update_listbox()
            # Actualizamos si coincide con lo ingresado
            if avenue == avenue_entry.get().strip():
                self.cycle_durations = presets[avenue]
                self.target_time = time.time() + self.cycle_durations[self.current_state]
            config_window.destroy()

        def edit_preset():
            sel = presets_listbox.curselection()
            if not sel:
                messagebox.showwarning("Advertencia", "Seleccione un preset para editar.", parent=config_window)
                return
            selected_str = presets_listbox.get(sel[0])
            # Esperamos formato: "Av. X: Verde=15, Amarillo=3, Rojo=20"
            try:
                # Dividimos por ":" una sola vez
                name_part, times_part = selected_str.split(":", 1)
                name_part = name_part.strip()
                times_part = times_part.strip()  # "Verde=15, Amarillo=3, Rojo=20"
                splitted = times_part.split(",")
                if len(splitted) < 3:
                    raise ValueError("Formato inválido en la línea.")
                verde_val = splitted[0].split("=")[1].strip()
                amarillo_val = splitted[1].split("=")[1].strip()
                rojo_val = splitted[2].split("=")[1].strip()
                avenue_entry.delete(0, tk.END)
                avenue_entry.insert(0, name_part)
                green_entry.delete(0, tk.END)
                green_entry.insert(0, verde_val)
                yellow_entry.delete(0, tk.END)
                yellow_entry.insert(0, amarillo_val)
                red_entry.delete(0, tk.END)
                red_entry.insert(0, rojo_val)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo editar el preset.\n{str(e)}", parent=config_window)

        def delete_preset():
            sel = presets_listbox.curselection()
            if not sel:
                messagebox.showwarning("Advertencia", "Seleccione un preset para eliminar.", parent=config_window)
                return
            selected_str = presets_listbox.get(sel[0])
            try:
                name_part, _ = selected_str.split(":", 1)
                avenue_name = name_part.strip()
            except Exception:
                messagebox.showerror("Error", "No se pudo identificar el preset a eliminar.", parent=config_window)
                return
            answer = messagebox.askyesno("Confirmar", f"¿Eliminar el preset de '{avenue_name}'?", parent=config_window)
            if not answer:
                return
            presets = load_presets()
            if avenue_name in presets:
                del presets[avenue_name]
                with open(PRESETS_FILE, "w") as f:
                    json.dump(presets, f, indent=2)
                update_listbox()
            else:
                messagebox.showwarning("Advertencia", f"No se encontró el preset para '{avenue_name}'.", parent=config_window)

        save_button = tk.Button(config_window, text="Guardar Preset", command=save_preset)
        save_button.grid(row=6, column=0, pady=10)
        edit_button = tk.Button(config_window, text="Editar Preset", command=edit_preset)
        edit_button.grid(row=6, column=1, pady=10)
        delete_button = tk.Button(config_window, text="Eliminar Preset", command=delete_preset)
        delete_button.grid(row=7, column=0, columnspan=2, pady=5)

        config_window.transient(self.parent)
        config_window.grab_set()
        self.parent.wait_window(config_window)

    def show_state(self):
        if self.current_state == "green":
            self.canvas.itemconfig(self.green_light, fill="green")
            self.canvas.itemconfig(self.yellow_light, fill="grey")
            self.canvas.itemconfig(self.red_light, fill="grey")
        elif self.current_state == "yellow":
            self.canvas.itemconfig(self.green_light, fill="grey")
            self.canvas.itemconfig(self.yellow_light, fill="yellow")
            self.canvas.itemconfig(self.red_light, fill="grey")
        else:
            self.canvas.itemconfig(self.green_light, fill="grey")
            self.canvas.itemconfig(self.yellow_light, fill="grey")
            self.canvas.itemconfig(self.red_light, fill="red")

    def update_lights(self):
        self.cycle_durations = self.get_cycle_durations()
        if self.current_state == "green":
            self.current_state = "yellow"
        elif self.current_state == "yellow":
            self.current_state = "red"
        else:
            self.current_state = "green"
        self.target_time = time.time() + self.cycle_durations[self.current_state]
        self.show_state()

    def get_cycle_durations(self):
        return self.cycle_durations

    def update_countdown(self):
        now = time.time()
        diff = self.target_time - now
        if diff < 0:
            self.update_lights()
            diff = self.target_time - time.time()
        seconds = int(diff)
        milliseconds = int((diff - seconds) * 1000)
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.info_label.config(
            text=(f"{formatted_time}\n"
                  f"Estado: {self.current_state.upper()} - "
                  f"Quedan {seconds}s {milliseconds}ms")
        )
        self.frame.after(50, self.update_countdown)

    def get_current_state(self):
        return self.current_state

    def resize_canvas(self, event):
        cw = int(event.width)
        ch = int(event.height)
        margin = 0.1 * min(cw, ch)
        max_width = int(cw - 2 * margin)
        max_height = int(ch - 2 * margin)
        hw_candidate = int(max_height * 0.4)
        housing_width = min(max_width, hw_candidate)
        housing_height = int(housing_width / 0.4)
        hx0 = (cw - housing_width) // 2
        hy0 = (ch - housing_height) // 2
        hx1 = hx0 + housing_width
        hy1 = hy0 + housing_height
        self.canvas.coords(self.housing_rect, hx0, hy0, hx1, hy1)
        section = housing_height // 3
        center_x = (hx0 + hx1) // 2
        centers = [(center_x, hy0 + section // 2 + i * section) for i in range(3)]
        circle_diam = min(int(0.8 * housing_width), int(0.8 * section))
        for light, center in zip([self.red_light, self.yellow_light, self.green_light], centers):
            cx, cy = center
            x0 = cx - circle_diam // 2
            y0 = cy - circle_diam // 2
            x1 = cx + circle_diam // 2
            y1 = cy + circle_diam // 2
            self.canvas.coords(light, x0, y0, x1, y1)
