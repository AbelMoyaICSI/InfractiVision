import tkinter as tk
from foto_rojo.semaforo import Semaforo
from foto_rojo.timestamp_updater import TimestampUpdater
from foto_rojo.videoplayer_opencv import VideoPlayerOpenCV

def create_foto_rojo_content(window, back_callback):
    window.state("zoomed")  # Asegura que la ventana esté maximizada
    # Panel izquierdo: Semáforo
    left_frame = tk.Frame(window, bg="white", width=250)
    left_frame.pack(side="left", fill="y", expand=False)
    semaforo = Semaforo(left_frame)
    # Panel central: Video y panel de placas
    center_frame = tk.Frame(window, bg="black")
    center_frame.pack(side="left", fill="both", expand=True)
    timestamp_label = tk.Label(center_frame, text="", bg="black", fg="white")
    timestamp_updater = TimestampUpdater(timestamp_label, window)
    VideoPlayerOpenCV(
        parent=center_frame,
        timestamp_updater=timestamp_updater,
        timestamp_label=timestamp_label,
        semaforo=semaforo
    )
    # Botón para volver a la ventana principal: llama al callback recibido
    back_button = tk.Button(window, text="Volver a Principal", command=back_callback)
    back_button.place(x=10, y=10)
