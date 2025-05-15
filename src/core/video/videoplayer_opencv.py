# src/video/videoplayer_opencv.py

import cv2
import threading
import time
import queue
import tkinter as tk
import json
import os
import numpy as np
import torch
import psutil

from tkinter import messagebox, simpledialog, Toplevel, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

from src.core.processing.plate_processing import process_plate

# Archivos de configuración
POLYGON_CONFIG_FILE = "config/polygon_config.json"
AVENUE_CONFIG_FILE  = "config/avenue_config.json"
PRESETS_FILE        = "config/time_presets.json"

class VideoPlayerOpenCV:
    def __init__(self, parent, timestamp_updater, timestamp_label, semaforo):
        self.parent            = parent
        self.timestamp_updater = timestamp_updater
        self.timestamp_label   = timestamp_label
        self.semaforo          = semaforo

        # Directorio de vídeos
        self.video_dir = os.path.join(os.getcwd(), "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        # Contenedor principal
        self.frame = tk.Frame(parent, bg='black')
        self.frame.pack(fill="both", expand=True)

        # Ventana de progreso (texto %)
        self.progress_win   = None
        self.progress_label = None
        self.progress_total = 0
        self.progress_done  = 0

        # Botonera inferior
        self.btn_frame = tk.Frame(self.frame, bg="black")
        self.btn_frame.pack(side="bottom", pady=12, anchor="w")

        btn_style = {
            "font": ("Arial", 12),
            "bg": "#4F4F4F",
            "fg": "white",
            "activebackground": "#6E6E6E",
            "activeforeground": "white",
            "bd": 0,
            "relief": "flat",
            "cursor": "hand2",
            "width": 14,
            "anchor": "center",
            "justify": "center"
        }

        self.load_button = tk.Button(
            self.btn_frame, text="CARGAR\nVIDEO",
            command=self.select_video,
            **btn_style
        )
        self.load_button.pack(side="left", padx=6)

        self.save_poly_button = tk.Button(
            self.btn_frame, text="GUARDAR\nÁREA",
            command=self.save_polygon,
            **btn_style
        )
        self.save_poly_button.pack(side="left", padx=6)

        self.delete_poly_button = tk.Button(
            self.btn_frame, text="BORRAR\nÁREA",
            command=self.delete_polygon,
            **btn_style
        )
        self.delete_poly_button.pack(side="left", padx=6)

        self.btn_gestion_polys = tk.Button(
            self.btn_frame, text="GESTIONAR\nÁREAS",
            command=self.gestionar_poligonos,
            **btn_style
        )
        self.btn_gestion_polys.pack(side="left", padx=6)

        self.btn_gestion_camaras = tk.Button(
            self.btn_frame, text="GESTIONAR\nCÁMARAS",
            command=self.gestionar_camaras,
            **btn_style
        )
        self.btn_gestion_camaras.pack(side="left", padx=6)

        # Panel vídeo + lateral
        self.video_panel_container = tk.Frame(self.frame, bg='black')
        self.video_panel_container.pack(side="top", fill="both", expand=True)

        self.video_frame = tk.Frame(
            self.video_panel_container, bg='black',
            width=640, height=360
        )
        self.video_frame.pack(side="left", fill="both", expand=True)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(
            self.video_frame, bg="black", bd=0, highlightthickness=0
        )
        self.video_label.pack(fill="both", expand=True)

        self.plates_frame = tk.Frame(
            self.video_panel_container, bg="gray", width=220
        )
        self.plates_frame.pack(side="right", fill="y")
        self.plates_frame.pack_propagate(False)

        self.plates_title = tk.Label(
            self.plates_frame, text="Placas Detectadas",
            bg="gray", fg="white", font=("Arial",16,"bold")
        )
        self.plates_title.pack(pady=10)

        self.plates_canvas = tk.Canvas(
            self.plates_frame, bg="gray", highlightthickness=0
        )
        self.plates_canvas.pack(side="left", fill="both", expand=True)

        self.plates_scrollbar = tk.Scrollbar(
            self.plates_frame, orient="vertical",
            command=self.plates_canvas.yview,
            bg="gray", troughcolor="gray", bd=0
        )
        self.plates_scrollbar.pack(side="right", fill="y")
        self.plates_canvas.configure(yscrollcommand=self.plates_scrollbar.set)

        self.plates_inner_frame = tk.Frame(self.plates_canvas, bg="gray")
        self.plates_inner_frame.bind("<Configure>", self._on_plates_inner_configure)
        self.plates_canvas.create_window((0,0), window=self.plates_inner_frame, anchor="nw")
        self.detected_plates_widgets = []

        # Timestamp y avenida
        self.timestamp_label.config(font=("Arial",30,"bold"), bg="black", fg="yellow")
        self.timestamp_label.place(in_=self.video_label, x=50, y=10)

        self.current_avenue = None
        self.avenue_label = tk.Label(
            self.video_frame, text="", font=("Arial",20,"bold"),
            bg="black", fg="white", wraplength=300
        )
        self.avenue_label.place(relx=0.5, y=80, anchor="n")

        # Info CPU/FPS/RAM
        self.info_label = tk.Label(self.video_frame, text="...", bg="black", fg="white", font=("Arial",11,"bold"))
        self.info_label.place(relx=0.98, y=10, anchor="ne")

        # Estado
        self.cap                = None
        self.running            = False
        self.orig_w, self.orig_h= None, None
        self.polygon_points     = []
        self.have_polygon       = False
        self.current_video_path = None

        # Cola de OCR
        self.plate_queue   = queue.Queue(maxsize=1)
        self.plate_running = True
        self.plate_thread  = threading.Thread(target=self.plate_loop, daemon=True)
        self.plate_thread.start()

        # Métricas
        self.last_time = time.time()
        self.fps_calc  = 0.0
        self.using_gpu = torch.cuda.is_available()

        cv2.setUseOptimized(True)
        try: cv2.setNumThreads(4)
        except: pass

        self.video_label.bind("<Button-1>", self.on_mouse_click_polygon)

    def _on_plates_inner_configure(self, event):
        self.plates_canvas.configure(scrollregion=self.plates_canvas.bbox("all"))

    def show_progress_window(self, total_steps):
        if self.progress_win:
            return
        self.progress_total = total_steps
        self.progress_done  = 0

        self.progress_win = Toplevel(self.parent)
        self.progress_win.title("Procesando video")
        self.progress_win.resizable(False, False)
        self.progress_win.transient(self.parent)
        self.progress_win.grab_set()
        self.progress_win.protocol("WM_DELETE_WINDOW", self._on_progress_close)

        # Centrar ventana en pantalla
        w, h = 300, 100
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        x = (sw - w)//2
        y = (sh - h)//2
        self.progress_win.geometry(f"{w}x{h}+{x}+{y}")

        self.progress_label = tk.Label(
            self.progress_win,
            text="Procesando video 0%",
            font=("Arial", 12)
        )
        self.progress_label.pack(expand=True, pady=20)
        self.progress_win.update_idletasks()

    def _on_progress_close(self):
        if messagebox.askyesno("Cancelar", "¿Desea cancelar el procesamiento?", parent=self.progress_win):
            self.hide_progress_window()

    def hide_progress_window(self):
        if self.progress_win:
            self.progress_win.destroy()
        self.progress_win   = None
        self.progress_label = None

    def select_video(self):
        file = filedialog.askopenfilename(
            title="Seleccionar vídeo",
            filetypes=[("Vídeos","*.mp4 *.avi *.mov *.mkv"),("Todos","*.*")]
        )
        if not file:
            return

        cap = cv2.VideoCapture(file)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        cap.release()

        self.show_progress_window(total)

        for i in range(total+1):
            time.sleep(0.01)
            if not self.progress_label:
                break
            pct = int(i * 100 / total)
            self.progress_label.config(text=f"Procesando video {pct}%")
            self.progress_win.update_idletasks()

        self.hide_progress_window()

        fname = os.path.basename(file)
        dest  = os.path.join(self.video_dir, fname)

        if self.current_video_path == dest:
            messagebox.showinfo("Información","Este vídeo ya está cargado.", parent=self.parent)
            return

        if not os.path.exists(dest):
            import shutil
            shutil.copy2(file, dest)

        cap_tmp = cv2.VideoCapture(dest)
        cap_tmp.set(cv2.CAP_PROP_BUFFERSIZE,1)
        ret, _ = cap_tmp.read()
        cap_tmp.release()
        if not ret:
            messagebox.showerror("Error","No se pudo leer el vídeo.", parent=self.parent)
            return

        self.stop_video()
        self.load_video(dest)

    def _load_video_async(self, path):
        cap_tmp = cv2.VideoCapture(path)
        cap_tmp.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap_tmp.read()
        cap_tmp.release()
        if not ret:
            self.parent.after(0, lambda: messagebox.showerror("Error", "No se pudo leer el vídeo."))
            return
        self.parent.after(0, lambda: self._finish_loading_video(path, frame))

    def _finish_loading_video(self, path, first_frame):
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.current_video_path = path
        h, w = first_frame.shape[:2]
        self.orig_h, self.orig_w = h, w
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_fps = max(self.cap.get(cv2.CAP_PROP_FPS), 30)
        self.running = True
        self.load_polygon_for_video()
        self.clear_detected_plates()
        self.semaforo.current_state = "green"
        self.semaforo.show_state()
        ave   = self.get_avenue_for_video(path)
        times = self.get_time_preset_for_video(path)
        if ave is None or times is None:
            self.first_time_setup(path)
        else:
            self.current_avenue = ave
            self.avenue_label.config(text=ave)
            self.cycle_durations = times
            self.target_time     = time.time() + times[self.semaforo.get_current_state()]
        if not self.timestamp_updater.running:
            self.timestamp_updater.start_timestamp()
        self.update_frames()

    def load_video(self, path):
        threading.Thread(
            target=self._load_video_async,
            args=(path,),
            daemon=True
        ).start()

    def stop_video(self):
        self.running = False
        if hasattr(self, "_after_id") and self._after_id:
            self.parent.after_cancel(self._after_id)
            self._after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None

    def plate_loop(self):
        while self.plate_running:
            try:
                frame, roi = self.plate_queue.get(timeout=1)
            except queue.Empty:
                continue
            bbox, plate_sr, ocr_text = process_plate(roi)
            if ocr_text:
                stamp = int(time.time())
                fname = f"plate_{ocr_text}_{stamp}.jpg"
                os.makedirs("data/output", exist_ok=True)
                cv2.imwrite(os.path.join("data/output", fname), plate_sr)
                self._safe_add_plate_to_panel(plate_sr, ocr_text)
            self.plate_queue.task_done()

    def update_frames(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._after_id = self.parent.after(int(1000/30), self.update_frames)
            return
        if self.polygon_points:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts  = np.array(self.polygon_points, np.int32).reshape(-1,1,2)
            cv2.fillPoly(mask, [pts], 255)
            roi = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            roi = frame.copy()
        if self.semaforo.get_current_state() == "red":
            if not self.plate_queue.full():
                self.plate_queue.put((frame.copy(), roi))
        bgr_img = self.resize_and_letterbox(frame)
        if self.polygon_points:
            self.draw_polygon_on_np(bgr_img)
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            alpha   = 0.9
            inst_fps = 1.0 / dt
            self.fps_calc = alpha * self.fps_calc + (1 - alpha) * inst_fps
        proc   = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / (1024*1024)
        dev    = "GPU" if self.using_gpu else "CPU"
        info   = f"{dev} | FPS: {self.fps_calc:.1f} | RAM: {mem_mb:.1f}MB"
        self.info_label.config(text=info)
        self.info_label.lift()
        rgb   = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        self.timestamp_label.lift()
        self.avenue_label.lift()
        self._after_id = self.parent.after(10, self.update_frames)

    def resize_and_letterbox(self, frame_bgr):
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl < 2 or hlbl < 2:
            return frame_bgr
        h_ori, w_ori = frame_bgr.shape[:2]
        scale = min(wlbl / w_ori, hlbl / h_ori, 1.0)
        new_w = int(w_ori * scale)
        new_h = int(h_ori * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas  = np.zeros((hlbl, wlbl, 3), dtype=np.uint8)
        off_x   = (wlbl - new_w) // 2
        off_y   = (hlbl - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas

    def clear_detected_plates(self):
        for item in self.detected_plates_widgets:
            item[0].destroy()
        self.detected_plates_widgets.clear()

    def gestionar_poligonos(self):
        w = tk.Toplevel(self.parent)
        w.title("Áreas Guardadas")
        lb = tk.Listbox(w, width=80)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w, command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)
        presets = {}
        if os.path.exists(POLYGON_CONFIG_FILE):
            try:
                with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                    presets = json.load(f)
            except:
                presets = {}
        for video_path, points in presets.items():
            lb.insert(tk.END, f"{video_path} → {points}")
        tk.Button(w, text="Cerrar", command=w.destroy).pack(pady=5)
        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)

    def delete_polygon(self):
        if not self.current_video_path or not self.polygon_points:
            messagebox.showwarning("Advertencia","No hay área.")
            return
        if not messagebox.askyesno("Confirmar","¿Borrar área?"):
            return
        try:
            with open(POLYGON_CONFIG_FILE,"r",encoding="utf-8") as f:
                presets = json.load(f)
            presets.pop(self.current_video_path,None)
            with open(POLYGON_CONFIG_FILE,"w",encoding="utf-8") as f:
                json.dump(presets,f,indent=2)
            self.have_polygon = False
            self.polygon_points = []
            messagebox.showinfo("Éxito","Área eliminada.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_polygon_for_video(self):
        self.have_polygon = False
        self.polygon_points = []
        if not self.current_video_path or not os.path.exists(POLYGON_CONFIG_FILE):
            return
        try:
            with open(POLYGON_CONFIG_FILE,"r",encoding="utf-8") as f:
                presets = json.load(f)
            if self.current_video_path in presets:
                self.polygon_points = presets[self.current_video_path]
                self.have_polygon = True
        except:
            pass

    def save_polygon(self):
        if not self.cap or not self.current_video_path:
            messagebox.showerror("Error","No hay vídeo cargado.")
            return
        if len(self.polygon_points) < 3:
            messagebox.showwarning("Advertencia","Al menos 3 vértices.")
            return
        self.have_polygon = True
        presets = {}
        if os.path.exists(POLYGON_CONFIG_FILE):
            try:
                with open(POLYGON_CONFIG_FILE,"r",encoding="utf-8") as f:
                    presets = json.load(f)
            except:
                pass
        presets[self.current_video_path] = self.polygon_points
        with open(POLYGON_CONFIG_FILE,"w",encoding="utf-8") as f:
            json.dump(presets,f,indent=2)
        messagebox.showinfo("Éxito","Área guardada.")

    def gestionar_camaras(self):
        w = tk.Toplevel(self.parent)
        w.title("Gestionar Cámaras (videos)")
        lb = tk.Listbox(w, width=60)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w, command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)
        for f in sorted(os.listdir(self.video_dir)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                lb.insert(tk.END, f)
        btn_frame = tk.Frame(w)
        btn_frame.pack(fill="x", pady=5)

        def on_cargar():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Advertencia", "Seleccione un vídeo.")
                return
            fn   = lb.get(sel[0])
            path = os.path.join(self.video_dir, fn)
            if path == self.current_video_path:
                messagebox.showinfo("Información", "Este vídeo ya está cargado.", parent=w)
                return
            w.destroy()
            self.stop_video()
            self.clear_detected_plates()
            self.semaforo.current_state = "green"
            self.semaforo.show_state()
            main_win = self.parent.winfo_toplevel()
            main_win.deiconify()
            self.load_video(path)

        tk.Button(btn_frame, text="Cargar", width=10, command=on_cargar).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Borrar", width=10, command=lambda: self._cam_del(lb)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cerrar", width=10, command=w.destroy).pack(side="left", padx=5)

        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)

    def _cam_del(self, lb):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia","Seleccione un vídeo para borrar.")
            return
        fn = lb.get(sel[0])
        path = os.path.join(self.video_dir, fn)
        if not messagebox.askyesno("Confirmar",f"¿Borrar '{fn}'?"):
            return
        if path == self.current_video_path:
            self.stop_video()
            self.clear_detected_plates()
            self.video_label.config(image="")
            self.current_video_path = None
        try:
            os.remove(path)
            messagebox.showinfo("Info",f"'{fn}' borrado.")
            lb.delete(sel[0])
        except Exception as e:
            messagebox.showerror("Error", str(e))
