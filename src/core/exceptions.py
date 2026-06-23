"""Jerarquía de excepciones del dominio InfractiVision.

Las capas superiores capturan estas excepciones de negocio y las transforman
en mensajes para el usuario; las capas de infraestructura las re-lanzan
desde sus errores técnicos (cv2, torch, sqlite, etc.).
"""
from __future__ import annotations


class InfractiVisionError(Exception):
    """Raíz de toda excepción del sistema."""


# ─── Domain ────────────────────────────────────────────────────────────────
class DomainError(InfractiVisionError):
    pass


class InvalidViolationError(DomainError):
    """Se intentó construir una infracción inválida (estado/ datos faltantes)."""


# ─── Infrastructure ───────────────────────────────────────────────────────
class InfrastructureError(InfractiVisionError):
    pass


class DetectorError(InfrastructureError):
    """Falla en el detector (YOLO, placa, semáforo)."""


class OCRError(InfrastructureError):
    """Falla en el lector OCR (LPRNet / EasyOCR / PaddleOCR)."""


class TrackerError(InfrastructureError):
    """Falla en el tracker DeepSORT."""


class VideoSourceError(InfrastructureError):
    """No se pudo abrir / leer la fuente de video."""


class RepositoryError(InfrastructureError):
    """Falla persistiendo o leyendo evidencia en la base de datos."""


# ─── Application ──────────────────────────────────────────────────────────
class UseCaseError(InfractiVisionError):
    pass
