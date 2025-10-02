# src/core/traffic_signal/semaphore.py

import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox
import json
import os
from src.path_helper import resource_path

PRESETS_FILE = resource_path("config/time_presets.json")

class Semaforo:
    """
    Panel de semáforo:
    Ciclo simple: green -> yellow -> red, configurable mediante presets
    asociados a un nombre de vídeo.
    """

    def __init__(self, parent):
        self.parent = parent
        self.current_video = None
        self.frame = tk.Frame(parent, bg='white')
        self.frame.pack(side="top", fill="both", expand=True)

        # Canvas para semáforo
        self.canvas = tk.Canvas(self.frame, bg='white', highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=5)

        # Label de estado y tiempos
        self.info_label = tk.Label(self.frame, text="Semáforo inactivo", font=("Arial", 14), bg='white')
        self.info_label.pack(pady=(0, 10))

        # Botón para abrir configuración de tiempos
        self.btn_tiempos = tk.Button(
            self.frame, text="Configurar Tiempos",
            command=self.gestionar_tiempos, width=20,
            bg="#3366FF", fg="white", bd=0, activebackground="#3366FF",
            activeforeground="white", pady=8,
        )
        self.btn_tiempos.pack(pady=5)

        # Dibujar carcasa y luces
        self.housing_rect = self.canvas.create_rectangle(0, 0, 0, 0,
                                                        fill="black", outline="gray", width=4)
        self.red_light    = self.canvas.create_oval(0, 0, 0, 0, fill="grey", outline="white", width=2)
        self.yellow_light = self.canvas.create_oval(0, 0, 0, 0, fill="grey", outline="white", width=2)
        self.green_light  = self.canvas.create_oval(0, 0, 0, 0, fill="grey", outline="white", width=2)

        self.canvas.bind("<Configure>", self.resize_canvas)

        # Estado inicial y duraciones por defecto
        self.current_state = "green"
        self.cycle_durations = {"green": 12, "yellow": 2, "red": 10}
        self.target_time = time.time() + self.cycle_durations[self.current_state]
        
        # Por defecto el semáforo está inactivo
        self.active = False
        self.show_inactive_state()

    def activate_semaphore(self):
        """Activa el semáforo cuando se carga un video"""
        self.active = True
        self.show_state()
        self.update_countdown()

    def deactivate_semaphore(self):
        """Desactiva el semáforo cuando no hay video"""
        print("🚦 DEACTIVATING SEMAPHORE - Cancelando todos los timers")
        self.active = False
        
        # CANCELAR TODOS LOS TIMERS PENDIENTES del frame
        try:
            # Intentar cancelar cualquier after pendiente
            # Nota: Tkinter no tiene un método directo para cancelar todos los after
            # pero al establecer active=False, los métodos update_countdown no se ejecutarán más
            pass
        except Exception as e:
            print(f"Error cancelando timers: {e}")
        
        self.show_inactive_state()
        self.info_label.config(text="🚦 Semáforo PAUSADO - Procesamiento completado")
        print("🚦 SEMÁFORO COMPLETAMENTE PAUSADO")

    def show_inactive_state(self):
        """Muestra el semáforo en estado inactivo (todas las luces grises)"""
        for light in [self.red_light, self.yellow_light, self.green_light]:
            self.canvas.itemconfig(light, fill="grey")

    def show_state(self):
        """Actualiza las luces según el estado actual"""
        if not self.active:
            self.show_inactive_state()
            return
            
        colors = {"green": self.green_light,
                "yellow": self.yellow_light,
                "red": self.red_light}
        for state, light in colors.items():
            fill = state if state == self.current_state else "grey"
            self.canvas.itemconfig(light, fill=fill)

    def update_countdown(self):
        """Actualiza el contador de tiempo y cambia el estado cuando es necesario"""
        if not self.active:
            return
            
        now = time.time()
        diff = self.target_time - now
        if diff <= 0:
            self.update_lights()
            diff = self.target_time - time.time()
        secs = int(diff)
        ms = int((diff - secs) * 1000)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.info_label.config(
            text=f"{ts}\nEstado: {self.current_state.upper()} – Quedan {secs}s {ms}ms"
        )
        # SOLO programar siguiente actualización si el semáforo está activo
        if self.active:
            self.frame.after(50, self.update_countdown)

    # --------------------
    # Gestión de presets
    # --------------------
    def load_presets(self):
        if not os.path.exists(PRESETS_FILE):
            return {}
        try:
            with open(PRESETS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def save_presets(self, data):
        os.makedirs(os.path.dirname(PRESETS_FILE), exist_ok=True)
        with open(PRESETS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def gestionar_tiempos(self):
        """
        UI para listar todos los presets (clave = nombre de vídeo)
        y permitir agregar, editar o eliminar.
        """
        win = tk.Toplevel(self.parent)
        win.title("Configurar Tiempos - Vídeos")

        tk.Label(win, text="Vídeos guardados:").grid(row=0, column=0, columnspan=3, pady=(5,0))
        lb = tk.Listbox(win, width=50)
        lb.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        sb = tk.Scrollbar(win, orient="vertical", command=lb.yview)
        sb.grid(row=1, column=3, sticky="ns", pady=5)
        lb.config(yscrollcommand=sb.set)

        def refresh():
            lb.delete(0, tk.END)
            for vid, times in self.load_presets().items():
                g, y, r = times["green"], times["yellow"], times["red"]
                time_slot = times.get("time_slot", "No especificado")
                lb.insert(tk.END, f"{vid} → 🟢{g}s 🟡{y}s 🔴{r}s ⏰{time_slot}")

        refresh()

        tk.Label(win, text="Nombre de vídeo:").grid(row=2, column=0, sticky="e", padx=5)
        entry_vid = tk.Entry(win, width=30)
        entry_vid.grid(row=2, column=1, columnspan=2, padx=5, pady=2)

        # MEJORA: Selectores visuales en lugar de Entry manual
        tk.Label(win, text="Verde (s):").grid(row=3, column=0, sticky="e", padx=5)
        var_g = tk.IntVar(value=30)
        spin_g = tk.Spinbox(win, from_=1, to=300, width=8, textvariable=var_g, 
                           font=("Arial", 10), justify="center", buttonbackground="#4CAF50")
        spin_g.grid(row=3, column=1, sticky="w", padx=2)

        tk.Label(win, text="Amarillo (s):").grid(row=4, column=0, sticky="e", padx=5)
        var_y = tk.IntVar(value=3)
        spin_y = tk.Spinbox(win, from_=1, to=10, width=8, textvariable=var_y,
                           font=("Arial", 10), justify="center", buttonbackground="#FFC107")
        spin_y.grid(row=4, column=1, sticky="w", padx=2)

        tk.Label(win, text="Rojo (s):").grid(row=5, column=0, sticky="e", padx=5)
        var_r = tk.IntVar(value=30)
        spin_r = tk.Spinbox(win, from_=1, to=300, width=8, textvariable=var_r,
                           font=("Arial", 10), justify="center", buttonbackground="#F44336")
        spin_r.grid(row=5, column=1, sticky="w", padx=2)
        
        # MEJORA: Selector de rango horario formato 12h con AM/PM
        tk.Label(win, text="Horario activo:").grid(row=6, column=0, sticky="e", padx=5)
        
        # Frame para selectores de hora
        time_frame = tk.Frame(win)
        time_frame.grid(row=6, column=1, columnspan=2, sticky="w", padx=2)
        
        # Variables para horarios (formato 12 horas)
        var_start_h = tk.IntVar(value=8)
        var_start_m = tk.IntVar(value=0)
        var_start_ampm = tk.StringVar(value="PM")
        var_end_h = tk.IntVar(value=9)
        var_end_m = tk.IntVar(value=0)
        var_end_ampm = tk.StringVar(value="PM")
        
        # Hora inicio formato 12h
        tk.Label(time_frame, text="De ").grid(row=0, column=0)
        spin_start_h = tk.Spinbox(time_frame, from_=1, to=12, width=3, textvariable=var_start_h,
                                 font=("Arial", 10), justify="center")
        spin_start_h.grid(row=0, column=1)
        tk.Label(time_frame, text=":").grid(row=0, column=2)
        spin_start_m = tk.Spinbox(time_frame, from_=0, to=59, width=3, textvariable=var_start_m,
                                 font=("Arial", 10), justify="center", increment=15)
        spin_start_m.grid(row=0, column=3)
        
        # Selector AM/PM para inicio
        combo_start_ampm = tk.OptionMenu(time_frame, var_start_ampm, "AM", "PM")
        combo_start_ampm.config(font=("Arial", 8), width=3)
        combo_start_ampm.grid(row=0, column=4, padx=2)
        
        tk.Label(time_frame, text=" a ").grid(row=0, column=5, padx=3)
        
        # Hora fin formato 12h
        spin_end_h = tk.Spinbox(time_frame, from_=1, to=12, width=3, textvariable=var_end_h,
                               font=("Arial", 10), justify="center")
        spin_end_h.grid(row=0, column=6)
        tk.Label(time_frame, text=":").grid(row=0, column=7)
        spin_end_m = tk.Spinbox(time_frame, from_=0, to=59, width=3, textvariable=var_end_m,
                               font=("Arial", 10), justify="center", increment=15)
        spin_end_m.grid(row=0, column=8)
        
        # Selector AM/PM para fin
        combo_end_ampm = tk.OptionMenu(time_frame, var_end_ampm, "AM", "PM")
        combo_end_ampm.config(font=("Arial", 8), width=3)
        combo_end_ampm.grid(row=0, column=9, padx=2)

        def on_save():
            vid = entry_vid.get().strip()
            if not vid:
                messagebox.showerror("Error", "Debe ingresar el nombre del vídeo.", parent=win)
                return
            
            # MEJORA: Obtener valores de los Spinbox
            g = var_g.get()
            y = var_y.get()
            r = var_r.get()
            
            # MEJORA: Construir time_slot desde selectores 12h y convertir a 24h
            def convert_12h_to_24h(hour_12, ampm):
                """Convierte hora de formato 12h a 24h"""
                if ampm == "AM":
                    if hour_12 == 12:
                        return 0
                    else:
                        return hour_12
                else:  # PM
                    if hour_12 == 12:
                        return 12
                    else:
                        return hour_12 + 12
            
            start_h_24 = convert_12h_to_24h(var_start_h.get(), var_start_ampm.get())
            end_h_24 = convert_12h_to_24h(var_end_h.get(), var_end_ampm.get())
            
            start_time = f"{start_h_24:02d}:{var_start_m.get():02d}"
            end_time = f"{end_h_24:02d}:{var_end_m.get():02d}"
            time_slot = f"{start_time} - {end_time}"
            
            presets = self.load_presets()
            presets[vid] = {
                "green": g, 
                "yellow": y, 
                "red": r,
                "time_slot": time_slot  # NUEVO: Guardar también el horario
            }
            self.save_presets(presets)
            refresh()
            # Si editamos el preset activo, actualizar ciclo
            if vid == self.current_video:
                self.cycle_durations = presets[vid]
                self.target_time = time.time() + self.cycle_durations[self.current_state]
            messagebox.showinfo("Éxito", f"Configuración guardada para '{vid}':\n• Verde: {g}s\n• Amarillo: {y}s\n• Rojo: {r}s\n• Horario: {time_slot}", parent=win)

        def on_edit():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Advertencia", "Seleccione un ítem para editar.", parent=win)
                return
            line = lb.get(sel[0])
            vid, _ = line.split(" → ",1)
            times = self.load_presets().get(vid, {})
            
            # Cargar nombre del video
            entry_vid.delete(0, tk.END); entry_vid.insert(0, vid)
            
            # MEJORA: Cargar valores en Spinbox
            var_g.set(times.get("green", 30))
            var_y.set(times.get("yellow", 3))
            var_r.set(times.get("red", 30))
            
            # MEJORA: Cargar horarios y convertir de 24h a 12h
            time_slot = times.get("time_slot", "20:00 - 21:00")
            try:
                start_str, end_str = time_slot.split(" - ")
                start_h_24, start_m = map(int, start_str.split(":"))
                end_h_24, end_m = map(int, end_str.split(":"))
                
                # Convertir hora de inicio de 24h a 12h
                if start_h_24 == 0:
                    var_start_h.set(12)
                    var_start_ampm.set("AM")
                elif start_h_24 < 12:
                    var_start_h.set(start_h_24)
                    var_start_ampm.set("AM")
                elif start_h_24 == 12:
                    var_start_h.set(12)
                    var_start_ampm.set("PM")
                else:
                    var_start_h.set(start_h_24 - 12)
                    var_start_ampm.set("PM")
                
                # Convertir hora final de 24h a 12h
                if end_h_24 == 0:
                    var_end_h.set(12)
                    var_end_ampm.set("AM")
                elif end_h_24 < 12:
                    var_end_h.set(end_h_24)
                    var_end_ampm.set("AM")
                elif end_h_24 == 12:
                    var_end_h.set(12)
                    var_end_ampm.set("PM")
                else:
                    var_end_h.set(end_h_24 - 12)
                    var_end_ampm.set("PM")
                
                var_start_m.set(start_m)
                var_end_m.set(end_m)
            except:
                # Si hay error en formato, usar valores por defecto (8PM - 9PM)
                var_start_h.set(8)
                var_start_m.set(0)
                var_start_ampm.set("PM")
                var_end_h.set(9)
                var_end_m.set(0)
                var_end_ampm.set("PM")

        def on_delete():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Advertencia", "Seleccione un ítem para eliminar.", parent=win)
                return
            line = lb.get(sel[0])
            vid = line.split(" → ",1)[0]
            if messagebox.askyesno("Confirmar", f"Eliminar preset para '{vid}'?", parent=win):
                presets = self.load_presets()
                presets.pop(vid, None)
                self.save_presets(presets)
                refresh()

        tk.Button(win, text="Guardar", command=on_save, bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).grid(row=7, column=0, pady=15, padx=5)
        tk.Button(win, text="Cargar edición", command=on_edit, bg="#2196F3", fg="white", font=("Arial", 10)).grid(row=7, column=1, pady=15, padx=5)
        tk.Button(win, text="Eliminar", command=on_delete, bg="#F44336", fg="white", font=("Arial", 10)).grid(row=7, column=2, pady=15, padx=5)

        win.transient(self.parent)
        win.grab_set()
        self.parent.wait_window(win)

    # --------------------
    # Ciclo de semáforo
    # --------------------
    def show_state(self):
        colors = {"green": self.green_light,
                  "yellow": self.yellow_light,
                  "red": self.red_light}
        for state, light in colors.items():
            fill = state if state == self.current_state else "grey"
            self.canvas.itemconfig(light, fill=fill)

    def update_lights(self):
        nxt = {"green":"yellow", "yellow":"red", "red":"green"}
        self.current_state = nxt[self.current_state]
        self.target_time = time.time() + self.cycle_durations[self.current_state]
        self.show_state()

    def update_countdown_legacy(self):
        """MÉTODO DUPLICADO - CORREGIDO PARA VERIFICAR ESTADO ACTIVO"""
        if not self.active:
            return
            
        now = time.time()
        diff = self.target_time - now
        if diff <= 0:
            self.update_lights()
            diff = self.target_time - time.time()
        secs = int(diff)
        ms = int((diff - secs) * 1000)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.info_label.config(
            text=f"{ts}\nEstado: {self.current_state.upper()} – Quedan {secs}s {ms}ms"
        )
        # SOLO programar siguiente actualización si el semáforo está activo
        if self.active:
            self.frame.after(50, self.update_countdown_legacy)

    def get_current_state(self):
        return self.current_state

    def resize_canvas(self, event):
        cw, ch = event.width, event.height
        margin = 0.1 * min(cw, ch)
        max_w, max_h = int(cw - 2*margin), int(ch - 2*margin)
        hw = min(max_w, int(max_h*0.4))
        hh = int(hw/0.4)
        x0, y0 = (cw-hw)//2, (ch-hh)//2
        self.canvas.coords(self.housing_rect, x0, y0, x0+hw, y0+hh)
        sec = hh // 3
        cx = x0 + hw//2
        diam = min(int(0.8*hw), int(0.8*sec))
        for i, light in enumerate([self.red_light, self.yellow_light, self.green_light]):
            cy = y0 + sec//2 + i*sec
            self.canvas.coords(light,
                               cx-diam//2, cy-diam//2,
                               cx+diam//2, cy+diam//2)
