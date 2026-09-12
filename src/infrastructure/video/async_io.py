"""Fase 3: I/O asíncrono para no bloquear el hilo de inferencia.

- AsyncCropWriter: imwrite (35ms medidos) fuera del hilo crítico vía pool.
- ThreadedVideoWriter: VideoWriter(mp4v) con cola; intenta fourcc H264/avc1
  (NVENC/QuickSync si el FFmpeg lo trae) con fallback a mp4v.

Ambos con fallback síncrono si algo falla: nunca deben romper el pipeline.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


class AsyncCropWriter:
    """Guarda crops JPEG en background. `flush()` al final del video."""

    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="crop-w")
        self._futures: list = []
        self._lock = threading.Lock()

    def save(self, path: str | Path, img: np.ndarray) -> None:
        try:
            fut = self._pool.submit(cv2.imwrite, str(path), img)
            with self._lock:
                self._futures.append(fut)
                # Evita crecer sin cota en videos largos: limpia completados.
                if len(self._futures) > 64:
                    self._futures = [f for f in self._futures if not f.done()]
        except Exception:
            try:
                cv2.imwrite(str(path), img)
            except Exception:
                pass

    def flush(self, timeout: float | None = None) -> None:
        with self._lock:
            futs = list(self._futures)
            self._futures.clear()
        for f in futs:
            try:
                f.result(timeout=timeout)
            except Exception:
                pass

    def shutdown(self) -> None:
        try:
            self.flush()
        finally:
            self._pool.shutdown(wait=True, cancel_futures=False)


class ThreadedVideoWriter:
    """VideoWriter con cola para que `write()` no bloquee inferencia+CPU.

    Intenta H264 (NVENC si FFmpeg lo soporta) y cae a mp4v.
    """

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path = str(path)
        self._writer = self._open(self.path, fps, size)
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._dropped = 0
        self._t = threading.Thread(target=self._loop, daemon=True, name="video-w")
        self._t.start()

    @staticmethod
    def _open(path: str, fps: float, size: tuple[int, int]):
        # Fase 3: intenta codecs con aceleración HW antes de mp4v CPU.
        for tag in ("avc1", "H264", "mp4v"):
            try:
                fourcc = cv2.VideoWriter_fourcc(*tag)
                w = cv2.VideoWriter(path, fourcc, fps, size)
                if w.isOpened():
                    return w
                try:
                    w.release()
                except Exception:
                    pass
            except Exception:
                continue
        # Último intento mp4v directo.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, fps, size)

    @property
    def is_opened(self) -> bool:
        try:
            return self._writer is not None and self._writer.isOpened()
        except Exception:
            return False

    def write(self, frame: np.ndarray) -> None:
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            # Si el encoder va lento, dropea el frame más viejo (preview, no evidencia).
            try:
                self._q.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            try:
                frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self._writer is not None:
                    self._writer.write(frame)
            except Exception:
                pass
            finally:
                self._q.task_done()

    def release(self) -> int:
        try:
            self._q.join()
        except Exception:
            pass
        self._stop.set()
        self._t.join(timeout=5.0)
        try:
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
        return self._dropped
