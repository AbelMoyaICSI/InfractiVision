"""Ventana para la detección de placas – re‑exporta el video player."""
import tkinter as tk
from src.core.utils.timestamp import TimestampUpdater
from src.core.traffic_signal.semaphore import Semaforo
from src.core.video.videoplayer_opencv import VideoPlayerOpenCV


def create_violation_window(window: tk.Toplevel, back_callback):
    window.state("zoomed")

    # ----- Panel izquierdo → semáforo virtual -----
    left = tk.Frame(window, bg="white", width=260)
    left.pack(side="left", fill="y", expand=False)
    sem = Semaforo(left)

    # ----- Panel central → video + lista de placas -----
    center = tk.Frame(window, bg="black")
    center.pack(side="left", fill="both", expand=True)

    ts_label = tk.Label(center, text="", bg="black", fg="white")
    ts_updater = TimestampUpdater(ts_label, window)

    VideoPlayerOpenCV(parent=center,
                      timestamp_updater=ts_updater,
                      timestamp_label=ts_label,
                      semaforo=sem)

    tk.Button(window, text="Volver", command=back_callback).place(x=10, y=10)