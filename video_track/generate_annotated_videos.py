import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
GROUND_TRUTH_PATH = PROJECT_ROOT / "tests" / "verdad.test.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

CODEC = "mp4v"
POLYGON_ALPHA = 0.35
POLYGON_THICKNESS = 2

SEMAPHORE_COLORS = {
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
}


def load_ground_truth():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("videos_verdad", [])


def get_semaphore_color(entry, frame_idx, fps):
    green = entry.get("green", 0)
    yellow = entry.get("yellow", 0)
    red = entry.get("red", 0)
    cycle = green + yellow + red

    if cycle == 0 or fps <= 0:
        return SEMAPHORE_COLORS["green"]

    elapsed = (frame_idx / fps) % cycle
    if elapsed < green:
        return SEMAPHORE_COLORS["green"]
    elif elapsed < green + yellow:
        return SEMAPHORE_COLORS["yellow"]
    else:
        return SEMAPHORE_COLORS["red"]


def draw_polygon_overlay(frame, polygon_points, color):
    overlay = frame.copy()
    pts = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=POLYGON_THICKNESS)
    return cv2.addWeighted(overlay, POLYGON_ALPHA, frame, 1 - POLYGON_ALPHA, 0)


def draw_info_text(frame, entry, frame_idx, total_frames, fps, current_color):
    name = entry["path_name"]
    green = entry.get("green", 0)
    yellow = entry.get("yellow", 0)
    red = entry.get("red", 0)
    infracciones = entry.get("infraccione", 0)

    if fps > 0:
        elapsed = frame_idx / fps
        cycle = green + yellow + red
        cycle_info = f" | Ciclo actual: {elapsed:.1f}s"
        if cycle > 0:
            cycle_info += f" (ciclo de {cycle}s)"
    else:
        cycle_info = ""

    semaforo_label = "VERDE" if current_color == SEMAPHORE_COLORS["green"] else \
                     "AMARILLO" if current_color == SEMAPHORE_COLORS["yellow"] else "ROJO"

    lines = [
        f"Video: {name}",
        f"Verde: {green}s | Amarillo: {yellow}s | Rojo: {red}s",
        f"Semaforo actual: {semaforo_label}{cycle_info}",
        f"Infracciones esperadas: {infracciones}",
        f"Frame: {frame_idx}/{total_frames}",
    ]

    y0 = 30
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, y0 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)


def sanitize_filename(name):
    return Path(name).stem


def process_video(entry, total, current):
    video_name = entry["path_name"]
    video_path = VIDEOS_DIR / video_name
    polygon_points = entry.get("polygon", [])

    if not polygon_points or len(polygon_points) < 3:
        print(f"  [{current}/{total}] {video_name} -> SKIPPED (sin poligono valido)")
        return

    if not video_path.exists():
        print(f"  [{current}/{total}] {video_name} -> SKIPPED (archivo no encontrado: {video_path})")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [{current}/{total}] {video_name} -> SKIPPED (no se pudo abrir)")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_name = f"{sanitize_filename(video_name)}_annotated.mp4"
    output_path = OUTPUT_DIR / output_name

    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        color = get_semaphore_color(entry, frame_idx, fps)
        annotated = draw_polygon_overlay(frame, polygon_points, color)
        draw_info_text(annotated, entry, frame_idx, total_frames, fps, color)

        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"  [{current}/{total}] {video_name} -> {output_name} ({frame_idx} frames)")


def main():
    entries = load_ground_truth()

    if not entries:
        print("No se encontraron videos en verdad.test.json")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(entries)
    print(f"Procesando {total} videos...\n")

    for i, entry in enumerate(entries, start=1):
        process_video(entry, total, i)

    print(f"\nListo. Videos anotados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
