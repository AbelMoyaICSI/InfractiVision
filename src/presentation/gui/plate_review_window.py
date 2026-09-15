"""Scrollable sequential review of the best plate crop per infractor."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from src.domain.entities.plate_evidence import PlateEvidence
from src.infrastructure.ocr.cloud_plate_readers import PlateRecognizerSnapshotReader
from src.infrastructure.reports import ReportRepository

# Intento de reutilizar la utilidad compartida de centrado. Si el módulo
# `src.gui.infractions_management_window` no está disponible (ej. falta
# `tkcalendar` en el entorno), se usa la implementación local de respaldo.
try:
    from src.gui.infractions_management_window import center_toplevel as _shared_center_toplevel
except Exception:
    _shared_center_toplevel = None


def center_toplevel(win, width, height):
    """Centra un `Toplevel` respecto a la pantalla principal del monitor.

    Uso: `center_toplevel(mi_ventana, 900, 700)` tras crear el Toplevel.
    Es segura: nunca lanza excepción aunque la ventana esté destruida.
    Misma firma/comportamiento que la utilidad de
    `src/gui/infractions_management_window.py`.
    """
    if _shared_center_toplevel is not None:
        try:
            return _shared_center_toplevel(win, width, height)
        except Exception as exc:
            print(f"⚠️ Falló utilidad compartida de centrado, uso local: {exc}")
    try:
        try:
            win.update_idletasks()
        except Exception:
            pass
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = (sw - int(width)) // 2
        y = (sh - int(height)) // 2
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        win.geometry(f"{int(width)}x{int(height)}")
        win.geometry("+{}+{}".format(x, y))
        return (x, y)
    except Exception as e:
        print(f"⚠️ No se pudo centrar ventana emergente: {e}")
        return None


def compute_zoomed_size(orig_w: int, orig_h: int, zoom: float) -> tuple[int, int]:
    """Calcula el tamaño en píxeles para un factor de zoom (lógica pura, testeable)."""
    zoom = max(EvidenceZoomDialog.MIN_ZOOM, min(EvidenceZoomDialog.MAX_ZOOM, zoom))
    return max(1, int(round(orig_w * zoom))), max(1, int(round(orig_h * zoom)))


class EvidenceZoomDialog:
    """Ventana emergente independiente (Toplevel) para examinar una evidencia en alta resolución.

    Equivalente Tkinter al QDialog solicitado: muestra la imagen original
    centrada, con botones Acercar/Alejar/Restablecer/Ajustar y zoom con la
    rueda del mouse. Se puede abrir/cerrar tantas veces como se quiera sin
    afectar el flujo de la ventana principal (no modal, sin grab_set).
    """

    MIN_ZOOM = 0.2
    MAX_ZOOM = 8.0
    ZOOM_STEP = 1.25

    def __init__(self, parent, image_path: str | Path, title: str = "Evidencia", on_close=None):
        self.parent = parent
        self.image_path = Path(image_path)
        self.title = title
        self.on_close = on_close
        self.zoom = 1.0
        self._photo = None
        self._canvas_image_id = None

        try:
            self.original = Image.open(self.image_path).copy()
        except Exception as exc:
            messagebox.showerror(
                "Imagen no disponible",
                f"No se pudo cargar la evidencia en alta resolución:\n{self.image_path}\n{exc}",
                parent=parent,
            )
            raise

        self.window = tk.Toplevel(parent)
        self.window.title(f"Lupa — {title}")
        # [RESPALDO centrado] Código anterior (no borrar): abría en esquina/descentrado.
        # self.window.geometry("900x700")
        center_toplevel(self.window, 900, 700)
        # transient: se mantiene sobre la ventana de revisión sin bloquearla
        # (sin grab_set para no interrumpir el flujo de validación/OCR).
        try:
            self.window.transient(parent)
        except tk.TclError:
            pass
        self.window.protocol("WM_DELETE_WINDOW", self._on_close_destroy)
        self._build()
        self._render()
        self._center_view()

    # -- UI -----------------------------------------------------------------
    def _build(self):
        toolbar = ttk.Frame(self.window)
        toolbar.pack(fill="x", padx=10, pady=8)
        ttk.Button(toolbar, text="＋ Acercar", command=self.zoom_in).pack(side="left")
        ttk.Button(toolbar, text="－ Alejar", command=self.zoom_out).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Restablecer (100%)", command=self.zoom_reset).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Ajustar a ventana", command=self.zoom_fit).pack(side="left", padx=(6, 0))
        self.zoom_label = ttk.Label(toolbar, text="100%")
        self.zoom_label.pack(side="left", padx=(12, 0))
        ow, oh = self.original.size
        ttk.Label(toolbar, text=f"Original: {ow}x{oh}px").pack(side="right")
        ttk.Label(
            self.window,
            text="Rueda del mouse o botones para examinar vehículo y placa. Esc para cerrar.",
        ).pack(pady=(0, 4))

        body = ttk.Frame(self.window)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.canvas = tk.Canvas(body, highlightthickness=1, highlightbackground="#ccc", bg="#1e1e1e")
        hbar = ttk.Scrollbar(body, orient="horizontal", command=self.canvas.xview)
        vbar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        # Zoom con rueda solo sobre este canvas (bind local: no rompe el scroll global).
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.canvas.bind("<Configure>", lambda _e: self._center_view())
        self.window.bind("<Escape>", lambda _e: self._on_close_destroy())
        self.window.bind("<plus>", lambda _e: self.zoom_in())
        self.window.bind("<minus>", lambda _e: self.zoom_out())
        self.window.bind("<KP_Add>", lambda _e: self.zoom_in())
        self.window.bind("<KP_Subtract>", lambda _e: self.zoom_out())
        self.window.bind("0", lambda _e: self.zoom_reset())

    # -- Zoom ----------------------------------------------------------------
    def _apply_zoom(self, new_zoom: float):
        self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        self._render()

    def zoom_in(self):
        self._apply_zoom(self.zoom * self.ZOOM_STEP)

    def zoom_out(self):
        self._apply_zoom(self.zoom / self.ZOOM_STEP)

    def zoom_reset(self):
        self._apply_zoom(1.0)

    def zoom_fit(self):
        try:
            self.window.update_idletasks()
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 50:
                cw = 800
            if ch < 50:
                ch = 550
            ow, oh = self.original.size
            fit = min(cw / max(1, ow), ch / max(1, oh))
            self._apply_zoom(fit)
        except (tk.TclError, ZeroDivisionError):
            pass

    def _on_wheel(self, event):
        try:
            if getattr(event, "num", None) == 4:
                self.zoom_in()
            elif getattr(event, "num", None) == 5:
                self.zoom_out()
            elif getattr(event, "delta", 0) > 0:
                self.zoom_in()
            elif getattr(event, "delta", 0) < 0:
                self.zoom_out()
            else:
                return
        except tk.TclError:
            pass
        return "break"

    def _render(self):
        ow, oh = self.original.size
        nw, nh = compute_zoomed_size(ow, oh, self.zoom)
        try:
            resized = self.original.resize((nw, nh), Image.LANCZOS)
        except Exception:
            resized = self.original.resize((nw, nh))
        self._photo = ImageTk.PhotoImage(resized)
        try:
            self.canvas.delete("all")
            cw = self.canvas.winfo_width() or 800
            ch = self.canvas.winfo_height() or 600
            # Centrada: si la imagen es menor que el viewport queda al centro;
            # si es mayor, el scrollregion permite recorrerla.
            x = max(cw // 2, nw // 2)
            y = max(ch // 2, nh // 2)
            self._canvas_image_id = self.canvas.create_image(x, y, anchor="center", image=self._photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.zoom_label.config(text=f"{int(round(self.zoom * 100))}% ({nw}x{nh})")
        except tk.TclError:
            pass

    def _center_view(self):
        """Recentra la imagen cuando la ventana cambia de tamaño."""
        try:
            if self._canvas_image_id is None or not self.canvas.winfo_exists():
                return
            bbox = self.canvas.bbox("all")
            if not bbox:
                return
            iw = bbox[2] - bbox[0]
            ih = bbox[3] - bbox[1]
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            x = max(cw // 2, iw // 2)
            y = max(ch // 2, ih // 2)
            self.canvas.coords(self._canvas_image_id, x, y)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except tk.TclError:
            pass

    def _on_close_destroy(self):
        cb = self.on_close
        try:
            self.window.destroy()
        except tk.TclError:
            pass
        finally:
            self._photo = None
            if cb:
                try:
                    cb(self)
                except Exception:
                    pass


class PlateReviewWindow:
    """Show the best crop per infractor and run Plate Recognizer one by one."""

    def __init__(self, parent, evidences: list[PlateEvidence], output_dir: str | Path, on_complete=None):
        self.parent = parent
        self.evidences = evidences
        self.output_dir = Path(output_dir)
        self.on_complete = on_complete
        self.current_index = 0
        self.processing = False
        self.images: list[ImageTk.PhotoImage] = []
        self.rows: list[dict] = []
        self._zoom_dialogs: list[EvidenceZoomDialog] = []
        self.reader = PlateRecognizerSnapshotReader()
        try:
            from src.infrastructure.ocr.cloud_plate_readers import has_plate_recognizer_token
            self._plate_token_ok = bool(has_plate_recognizer_token())
        except Exception:
            self._plate_token_ok = True  # sin bloqueo: el error saldrá por evidencia
        # Tkinter NO es thread-safe: el worker de OCR nunca toca widgets.
        # Publica resultados en esta cola y un poller (hilo de Tk) los drena.
        self._results_queue: queue.Queue[tuple[int, str, float, str]] = queue.Queue()

        self.window = tk.Toplevel(parent)
        self.window.title("Validación secuencial de placas")
        # [RESPALDO centrado] Código anterior (no borrar): abría en esquina/descentrado.
        # self.window.geometry("1050x760")
        center_toplevel(self.window, 1050, 760)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        if not getattr(self, "_plate_token_ok", True):
            self.status.config(
                text="Sin token Plate Recognizer: configure el token para leer las placas."
            )
        self._render_all()
        self.window.after(150, self._process_next)
        self.window.after(50, self._poll_results)

    def _build(self):
        ttk.Label(
            self.window,
            text="Mejores frames de carros infractores",
            font=("Arial", 16, "bold"),
        ).pack(pady=(10, 2))
        ttk.Label(
            self.window,
            text="Plate Recognizer procesa cada vehículo secuencialmente.",
        ).pack(pady=(0, 8))
        self.status = ttk.Label(self.window, text="Preparando reconocimiento...")
        self.status.pack()

        container = ttk.Frame(self.window)
        container.pack(fill="both", expand=True, padx=12, pady=8)
        self.canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)
        self.scroll_frame.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self._scroll_frame_id = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._bind_mousewheel()

        controls = ttk.Frame(self.window)
        controls.pack(fill="x", padx=12, pady=10)
        ttk.Button(controls, text="Reintentar actual", command=self._retry_current).pack(side="left")
        ttk.Button(controls, text="Completado", command=self._complete).pack(side="right", padx=(0, 8))
        ttk.Button(controls, text="Exportar validados", command=self._export).pack(side="right")

    def _on_canvas_configure(self, event):
        """Mantiene el frame interno al ancho del canvas al redimensionar."""
        try:
            self.canvas.itemconfig(self._scroll_frame_id, width=event.width)
        except tk.TclError:
            pass

    def _bind_mousewheel(self):
        """Activa el scroll con rueda solo mientras el cursor está sobre la lista."""
        self.canvas.bind("<Enter>", lambda _e: self._bind_mousewheel_global(), add="+")
        self.scroll_frame.bind("<Enter>", lambda _e: self._bind_mousewheel_global(), add="+")
        self.canvas.bind("<Leave>", lambda _e: self._unbind_mousewheel(), add="+")
        self.scroll_frame.bind("<Leave>", lambda _e: self._unbind_mousewheel(), add="+")
        # Por si el cursor ya está sobre la lista al abrir.
        self._bind_mousewheel_global()

    def _bind_mousewheel_global(self):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-5>", self._on_mousewheel, add="+")

    def _unbind_mousewheel(self):
        try:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        except tk.TclError:
            pass

    def _on_mousewheel(self, event):
        """Desplaza el canvas con la rueda (Windows/macOS `delta`, Linux `Button-4/5`)."""
        try:
            if not self.canvas.winfo_exists():
                return
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            elif getattr(event, "delta", 0):
                delta = -1 * (event.delta / 120)
            else:
                return
            self.canvas.yview_scroll(int(delta * 3), "units")
        except tk.TclError:
            pass
        return "break"

    def _render_all(self):
        self.images.clear()
        self.rows.clear()
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        if not self.evidences:
            ttk.Label(self.scroll_frame, text="No hay evidencias de infractores.").pack(pady=30)
            self.status.config(text="0 evidencias válidas")
            return

        for position, evidence in enumerate(self.evidences, 1):
            row = ttk.LabelFrame(
                self.scroll_frame,
                text=f"Evidencia {position}/{len(self.evidences)} | Track ID {evidence.track_id}",
                padding=8,
            )
            row.pack(fill="x", pady=5, padx=4)

            try:
                image = Image.open(evidence.crop_path)
                image.thumbnail((300, 110))
                photo = ImageTk.PhotoImage(image.copy())
                self.images.append(photo)
                thumb_box = ttk.Frame(row)
                thumb_box.pack(side="left", padx=(0, 12))
                thumb_label = ttk.Label(thumb_box, image=photo, cursor="hand2")
                thumb_label.pack()
                thumb_label.bind(
                    "<Button-1>",
                    lambda _e, p=evidence.crop_path, t=f"Track {evidence.track_id}": self._open_zoom(p, t),
                )
                ttk.Button(
                    thumb_box,
                    text="🔍 Lupa con zoom",
                    command=lambda p=evidence.crop_path, t=f"Track {evidence.track_id}": self._open_zoom(p, t),
                ).pack(pady=(4, 0))
            except Exception:
                ttk.Label(row, text="Crop no disponible").pack(side="left", padx=(0, 12))

            info = ttk.Frame(row)
            info.pack(side="left", fill="x", expand=True)
            ttk.Label(
                info,
                text=(
                    f"Vehículo: {evidence.vehicle_class} | Frame: {evidence.frame_index} | "
                    f"Tiempo: {evidence.timestamp_seconds:.2f}s | Calidad: {evidence.quality_score:.2f}"
                ),
            ).pack(anchor="w")
            if (getattr(evidence, "metadata", None) or {}).get("full_car"):
                ttk.Label(
                    info,
                    text="🚗 VEHÍCULO COMPLETO — la API buscará la placa (validar con lupa)",
                    foreground="#b9770e",
                ).pack(anchor="w", pady=(4, 0))
            ttk.Label(info, text="Resultado Plate Recognizer:").pack(anchor="w", pady=(8, 0))
            text_var = tk.StringVar(value=evidence.plate_text)
            entry = ttk.Entry(info, textvariable=text_var, width=28)
            entry.pack(anchor="w", pady=3)
            confidence = ttk.Label(info, text="Pendiente")
            confidence.pack(anchor="w")
            validated = tk.BooleanVar(value=evidence.validated)
            check = ttk.Checkbutton(row, text="Validar", variable=validated, state="disabled")
            check.pack(side="right", padx=8)
            self.rows.append({
                "evidence": evidence,
                "text": text_var,
                "confidence": confidence,
                "validated": validated,
                "check": check,
                "entry": entry,
            })

    def _open_zoom(self, crop_path: str | Path, title: str = "Evidencia"):
        """Abre una lupa con zoom en ventana independiente y reutilizable.

        Cada clic crea un `EvidenceZoomDialog` nuevo (no modal); al cerrarse
        solo se destruye ese Toplevel y se libera de la lista, sin afectar el
        flujo de OCR/validación. Puede abrirse/cerrarse ilimitadas veces.
        """
        try:
            dialog = EvidenceZoomDialog(
                self.window,
                crop_path,
                title=title,
                on_close=lambda dlg: self._zoom_dialogs.remove(dlg) if dlg in self._zoom_dialogs else None,
            )
        except Exception:
            return  # EvidenceZoomDialog ya mostró el error al usuario
        self._zoom_dialogs.append(dialog)
        try:
            dialog.window.focus_set()
        except tk.TclError:
            pass

    def _process_next(self):
        if self.processing or self.current_index >= len(self.rows):
            if self.current_index >= len(self.rows) and self.rows:
                self.status.config(text="Reconocimiento terminado. Revise y valide los resultados.")
            return
        self.processing = True
        row = self.rows[self.current_index]
        evidence: PlateEvidence = row["evidence"]
        self.status.config(text=f"Reconociendo {self.current_index + 1}/{len(self.rows)} (espera anti-límite activa)...")

        def work():
            idx = self.current_index
            # Wall-clock de inferencia de ESTA placa: t0 justo antes de
            # procesar su crop, t1 al terminar (para el TR individual).
            _t0 = time.time()
            try:
                text, confidence = self.reader.read(evidence.crop_path)
                error = ""
            except Exception as exc:
                text, confidence, error = "", 0.0, str(exc)
            finally:
                try:
                    _meta = evidence.metadata
                    if _meta is None:
                        evidence.metadata = _meta = {}
                    _meta["ocr_seconds"] = round(time.time() - _t0, 4)
                except Exception:
                    pass
            # NUNCA llamar Tk desde este hilo: encolar y dejar que el poller
            # (hilo principal) aplique _show_result.
            self._results_queue.put((idx, text, confidence, error))

        threading.Thread(target=work, daemon=True).start()

    def _poll_results(self):
        """[HILO DE TK] Drena la cola de resultados del worker de OCR."""
        try:
            while True:
                index, text, confidence, error = self._results_queue.get_nowait()
                self._show_result(index, text, confidence, error)
        except queue.Empty:
            pass
        try:
            self.window.after(50, self._poll_results)
        except tk.TclError:
            pass

    def _show_result(self, index: int, text: str, confidence: float, error: str):
        if index >= len(self.rows):
            return
        row = self.rows[index]
        evidence: PlateEvidence = row["evidence"]
        previous = (evidence.plate_text or "").strip().upper()
        meta = getattr(evidence, "metadata", None) or {}
        # Fallback YOLO: el crop es carro completo. El texto OCR solo sirve
        # como clave de dedup de NIE: se conserva pero queda como NIE
        # (sin auto-validar). El operador aun puede tildar manual a NID.
        is_fallback = bool(meta.get("fallback_by_quality") or meta.get("dedup_eligible") or meta.get("full_car"))
        if text:
            evidence.plate_text = text
            evidence.ocr_confidence = confidence
            evidence.ocr_method = "plate_recognizer"
            row["text"].set(text)
            if is_fallback:
                try:
                    evidence.metadata["dedup_key"] = text.strip().upper()
                except Exception:
                    pass
                row["confidence"].config(
                    text=f"Confianza: {confidence:.2f} — YOLO no localizó, queda NIE (solo clave dedup)"
                )
                # Queda NIE por defecto: check habilitado para promocion manual.
                evidence.validated = False
                row["validated"].set(False)
                row["check"].state(["!disabled"])
            else:
                row["confidence"].config(text=f"Confianza: {confidence:.2f}")
                # Auto-validar por defecto: Plate Recognizer sí detectó placa.
                # El usuario aún puede desmarcar manualmente antes de Completar/Exportar.
                evidence.validated = True
                row["validated"].set(True)
                row["check"].state(["!disabled"])
        elif previous:
            # Fallback: la API falló/offline o no vio placa; se conserva el
            # texto previo (si lo hay) en vez de vaciarlo.
            evidence.ocr_method = "previo"
            row["text"].set(previous)
            row["confidence"].config(
                text=f"API sin resultado ({error or 'placa no reconocida'}). Se conserva: {previous}"
            )
            row["check"].state(["!disabled"])
        else:
            evidence.plate_text = ""
            evidence.ocr_confidence = 0.0
            evidence.ocr_method = "plate_recognizer"
            evidence.validated = False
            row["text"].set("")
            row["confidence"].config(text=f"Sin resultado: {error or 'placa no reconocida'}")
            row["validated"].set(False)
            row["check"].state(["disabled"])
        self.processing = False
        self.current_index = index + 1
        self.window.after(50, self._process_next)

    def _retry_current(self):
        if not self.rows or self.processing:
            return
        self.current_index = min(self.current_index, len(self.rows) - 1)
        self._process_next()

    def _apply_review_values(self):
        for row in self.rows:
            evidence: PlateEvidence = row["evidence"]
            evidence.plate_text = row["text"].get().strip().upper()
            evidence.validated = bool(row["validated"].get()) and bool(evidence.plate_text)

    def _export(self):
        self._apply_review_values()
        self._notify_complete()
        valid = [evidence for evidence in self.evidences if evidence.validated and evidence.plate_text]
        if not valid:
            messagebox.showwarning("Sin resultados", "Valide al menos una placa reconocida.", parent=self.window)
            return
        json_path, csv_path = ReportRepository().export_validated(self.output_dir, valid)
        messagebox.showinfo("Reporte exportado", f"JSON: {json_path}\nCSV: {csv_path}", parent=self.window)

    def _close_zoom_dialogs(self):
        for dialog in list(getattr(self, "_zoom_dialogs", [])):
            try:
                dialog.window.destroy()
            except Exception:
                pass
        try:
            self._zoom_dialogs.clear()
        except Exception:
            pass

    def _complete(self):
        """Botón 'Completado': aplica la validación, notifica al llamador
        (que dispara la migración) y cierra la ventana."""
        self._unbind_mousewheel()
        self._close_zoom_dialogs()
        self._apply_review_values()
        self._notify_complete()
        try:
            self.window.destroy()
        except Exception:
            pass

    def _on_close(self):
        """Al cerrar la ventana sincroniza la validación marcada (sin exportar)."""
        self._unbind_mousewheel()
        self._close_zoom_dialogs()
        self._apply_review_values()
        self._notify_complete()
        try:
            self.window.destroy()
        except Exception:
            pass

    def _notify_complete(self):
        """Notifica al llamador con TODOS los evidences ya mutados (NID/NIE)."""
        if self.on_complete:
            try:
                self.on_complete(self.evidences)
            except Exception as exc:
                print(f"⚠️ Error en callback de validación: {exc}")
