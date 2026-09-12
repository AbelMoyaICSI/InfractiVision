# src/gui/app_manager.py

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from src.gui.welcome_window import WelcomeFrame
from src.gui.red_light_violation_window import create_violation_window
from src.gui.infractions_management_window import create_infractions_window

class AppManager:
    """Centraliza la navegación entre pantallas GUI en una única ventana,
       sin intentar maximizar automáticamente ni restaurar estados problemáticos."""

    def __init__(self, root: tk.Tk, user_id: str = None, device_id: str = None,
                 process_frame_uc=None, traffic_light_state=None):
        self.user_id = user_id
        self.device_id = device_id
        self.process_frame_uc = process_frame_uc
        self.traffic_light_state = traffic_light_state

        self.root = root
        self.root.title("InfractiVision")
        # No se intenta maximizar automáticamente (evita segfault en Linux)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.show_welcome()

    def _clear_root(self):
        """Destruye todos los widgets en root y fuerza actualización."""
        for w in self.root.winfo_children():
            w.destroy()
        self.root.update_idletasks()

    def _on_closing(self):
        """Maneja el cierre de la aplicación principal."""
        self.root.quit()
        self.root.destroy()

    def show_welcome(self):
        """Pantalla de bienvenida."""
        self._clear_root()
        self.root.title("InfractiVision – Principal")
        frm = WelcomeFrame(self.root, self)
        frm.pack(fill="both", expand=True)

    def open_violation_window(self):
        """Pantalla de Foto Rojo con precarga bloqueante de modelos IA.

        Muestra un spinner, calienta YOLO/LPRNet en un hilo worker (Tk nunca
        se toca desde el worker) y recién entonces crea la ventana. Así el
        análisis posterior no se congela cargando `torch.load` a mitad del
        video. Si la precarga crítica falla, ofrece Reintentar/Volver.
        """
        self._clear_root()
        self.root.title("InfractiVision – Foto Rojo")

        loading = tk.Frame(self.root, bg="white")
        loading.pack(fill="both", expand=True)
        box = tk.Frame(loading, bg="white")
        box.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(box, text="Cargando modelos de IA...",
                 font=("Arial", 18, "bold"), bg="white", fg="#273D86").pack(pady=(0, 6))
        status_var = tk.StringVar(value="Preparando...")
        tk.Label(box, textvariable=status_var,
                 font=("Arial", 12), bg="white", fg="gray30").pack(pady=(0, 12))
        bar = ttk.Progressbar(box, mode="indeterminate", length=320)
        bar.pack()
        try:
            bar.start(12)
        except tk.TclError:
            pass

        msg_q: queue.Queue = queue.Queue()
        done: queue.Queue = queue.Queue()

        def _work() -> None:
            try:
                from src.application.services.model_preloader import preload_foto_rojo_models
                result = preload_foto_rojo_models(progress=msg_q.put_nowait)
                done.put_nowait(("ok", result))
            except Exception as e:  # noqa: BLE001 - se reporta a la UI
                done.put_nowait(("error", str(e)))

        threading.Thread(target=_work, daemon=True, name="foto-rojo-preload").start()
        self._poll_foto_rojo_preload(loading, status_var, msg_q, done)

    def _poll_foto_rojo_preload(self, loading, status_var, msg_q, done) -> None:
        """[HILO DE TK] Drena progreso y espera el fin de la precarga."""
        try:
            if not loading.winfo_exists():
                return
        except tk.TclError:
            return
        try:
            while True:
                status_var.set(str(msg_q.get_nowait()))
        except queue.Empty:
            pass
        try:
            kind, payload = done.get_nowait()
        except queue.Empty:
            try:
                self.root.after(60, lambda: self._poll_foto_rojo_preload(loading, status_var, msg_q, done))
            except tk.TclError:
                pass
            return
        if kind == "error":
            self._show_preload_error(loading, f"No se pudo completar la precarga: {payload}")
            return
        errors = payload.get("errors", [])
        critical_missing = (
            payload.get("vehicle_detector") is None
            or payload.get("plate_detector") is None
        )
        if critical_missing:
            detail = "; ".join(errors) if errors else "modelos no disponibles"
            self._show_preload_error(loading, f"Modelos críticos sin cargar ({detail}).")
            return
        try:
            loading.destroy()
        except tk.TclError:
            pass
        self._clear_root()
        self.root.title("InfractiVision – Foto Rojo")
        create_violation_window(
            self.root,
            self.show_welcome,
            process_frame_uc=self.process_frame_uc,
            traffic_light_state=self.traffic_light_state,
            preloaded=payload,
        )
        if not payload.get("plate_token_ok"):
            try:
                messagebox.showwarning(
                    "Plate Recognizer sin token",
                    "No hay PLATE_RECOGNIZER_API_TOKEN.\n"
                    "En vivo solo se detectan vehículos y placas; la lectura "
                    "se hará en la revisión final cuando configure el token.",
                    parent=self.root,
                )
            except tk.TclError:
                pass

    def _show_preload_error(self, loading, message: str) -> None:
        """Reemplaza el spinner por un error con Reintentar/Volver."""
        try:
            for w in loading.winfo_children():
                w.destroy()
        except tk.TclError:
            return
        tk.Label(loading, text="No se pudieron cargar los modelos",
                 font=("Arial", 16, "bold"), bg="white", fg="#c0392b").pack(pady=(40, 8))
        tk.Label(loading, text=message, font=("Arial", 11),
                 bg="white", fg="gray30", wraplength=560, justify="center").pack(pady=(0, 16))
        btns = tk.Frame(loading, bg="white")
        btns.pack()
        tk.Button(btns, text="Reintentar", font=("Arial", 12, "bold"),
                  bg="#3366FF", fg="white", bd=0, padx=20, pady=8,
                  command=self.open_violation_window).pack(side="left", padx=8)
        tk.Button(btns, text="Volver", font=("Arial", 12),
                  bg="#95a5a6", fg="white", bd=0, padx=20, pady=8,
                  command=self.show_welcome).pack(side="left", padx=8)

    def open_infractions_window(self):
        """Pantalla de Gestión de Infracciones."""
        self._clear_root()
        self.root.title("InfractiVision – Gestión de Infracciones")
        create_infractions_window(self.root, self.show_welcome)