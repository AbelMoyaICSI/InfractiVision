"""Scrollable sequential review of the best plate crop per infractor."""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from src.domain.entities.plate_evidence import PlateEvidence
from src.infrastructure.ocr.cloud_plate_readers import PlateRecognizerSnapshotReader
from src.infrastructure.reports import ReportRepository


class PlateReviewWindow:
    """Show every valid evidence crop and run Plate Recognizer one by one."""

    def __init__(self, parent, evidences: list[PlateEvidence], output_dir: str | Path, on_complete=None):
        self.parent = parent
        self.evidences = evidences
        self.output_dir = Path(output_dir)
        self.on_complete = on_complete
        self.current_index = 0
        self.processing = False
        self.images: list[ImageTk.PhotoImage] = []
        self.rows: list[dict] = []
        self.reader = PlateRecognizerSnapshotReader()

        self.window = tk.Toplevel(parent)
        self.window.title("Validación secuencial de placas")
        self.window.geometry("1050x760")
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self._render_all()
        self.window.after(150, self._process_next)

    def _build(self):
        ttk.Label(
            self.window,
            text="Mejores frames de infracciones con placa detectada",
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
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        controls = ttk.Frame(self.window)
        controls.pack(fill="x", padx=12, pady=10)
        ttk.Button(controls, text="Reintentar actual", command=self._retry_current).pack(side="left")
        ttk.Button(controls, text="Exportar validados", command=self._export).pack(side="right")

    def _render_all(self):
        self.images.clear()
        self.rows.clear()
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        if not self.evidences:
            ttk.Label(self.scroll_frame, text="No hay evidencias con placa detectada.").pack(pady=30)
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
                ttk.Label(row, image=photo).pack(side="left", padx=(0, 12))
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
            try:
                text, confidence = self.reader.read(evidence.crop_path)
                error = ""
            except Exception as exc:
                text, confidence, error = "", 0.0, str(exc)
            self.window.after(0, lambda: self._show_result(self.current_index, text, confidence, error))

        threading.Thread(target=work, daemon=True).start()

    def _show_result(self, index: int, text: str, confidence: float, error: str):
        if index >= len(self.rows):
            return
        row = self.rows[index]
        evidence: PlateEvidence = row["evidence"]
        evidence.plate_text = text
        evidence.ocr_confidence = confidence
        evidence.ocr_method = "plate_recognizer"
        row["text"].set(text)
        if text:
            row["confidence"].config(text=f"Confianza: {confidence:.2f}")
            row["check"].state(["!disabled"])
        else:
            row["confidence"].config(text=f"Sin resultado: {error or 'placa no reconocida'}")
            row["check"].state(["disabled"])
        self.processing = False
        self.current_index = index + 1
        self.window.after(50, self._process_next)

    def _retry_current(self):
        if not self.rows:
            return
        self.current_index = min(self.current_index, len(self.rows) - 1)
        self.processing = False
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

    def _on_close(self):
        """Al cerrar la ventana sincroniza la validación marcada (sin exportar)."""
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
