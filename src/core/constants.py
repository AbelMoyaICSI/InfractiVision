"""Constantes globales del sistema InfractiVision.

Solo VALORES, sin lógica. No depende de ninguna otra capa.
"""
from __future__ import annotations

# === COCO classes utilizadas por YOLOv8 ===
# 2: car, 5: bus, 7: truck (omitimos 0:person, 3:motorcycle por requisito del cliente)
VEHICLE_CLASS_IDS: tuple[int, ...] = (2, 5, 7)
VEHICLE_CLASS_NAMES: dict[int, str] = {2: "Carro", 5: "Bus", 7: "Camion"}

# === Estados del semáforo ===
TRAFFIC_LIGHT_GREEN = "green"
TRAFFIC_LIGHT_YELLOW = "yellow"
TRAFFIC_LIGHT_RED = "red"

# === Tipos de infracción ===
VIOLATION_RED_LIGHT = "RED_LIGHT_CROSSED"

# === Default thresholds (overridable desde config/settings.py) ===
DEFAULT_DETECTION_CONF = 0.30
DEFAULT_OCR_MIN_CONF = 0.55
DEFAULT_TRACKER_MAX_AGE = 30

# === Regiones SIIV (resumen ─ ver recognizer para detalle completo) ===
SIIV_REGION_TRUJILLO = "T"
