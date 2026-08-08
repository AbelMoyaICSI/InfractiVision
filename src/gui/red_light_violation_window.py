# src/gui/red_light_violation_window.py

import tkinter as tk
from src.core.utils.timestamp import TimestampUpdater
from src.core.traffic_signal.semaphore import Semaforo
from src.core.video.videoplayer_opencv import VideoPlayerOpenCV


def create_violation_window(container: tk.Widget, back_callback,
                            process_frame_uc=None, traffic_light_state=None):
    """Crea la pantalla de Foto Rojo.

    Los argumentos `process_frame_uc` y `traffic_light_state` son la nueva
    seam Clean Architecture y son **opcionales**: si vienen `None` la GUI
    funciona idéntica a antes.
    """
    left = tk.Frame(container, bg="white", width=260)
    left.pack(side="left", fill="y", expand=False)
    sem = Semaforo(left)


    center = tk.Frame(container, bg="black")
    center.pack(side="left", fill="both", expand=True)

    ts_label = tk.Label(center, text="", bg="black", fg="white")
    ts_updater = TimestampUpdater(ts_label, container)

    VideoPlayerOpenCV(
        parent=center,
        timestamp_updater=ts_updater,
        timestamp_label=ts_label,
        semaforo=sem,
        process_frame_uc=process_frame_uc,
        traffic_light_state=traffic_light_state,
    )

    tk.Button(container, text="Volver", font=("Arial", 12), padx=16, command=back_callback, bg="#3366FF", fg="white", bd=0, activebackground="#3366FF", activeforeground="white").place(x=10, y=10)
