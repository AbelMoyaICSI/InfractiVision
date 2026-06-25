"""
semaforo_log.py — log de transiciones del semáforo para todos los videos
de ``tests/verdad.test.json``.

Lee el JSON de calibración, itera las 11 entradas válidas (con
``green/yellow/red`` no todos cero) y escribe las transiciones
determinísticas de cada video a ``data/logs/semaforo.log``.

El campo ``time`` del JSON se usa como duración del video (no se abre
el archivo de video). La separación entre videos se marca con
``── [i/N] nombre ──``.

Uso:
    python scripts/adapter/semaforo_log.py
    python scripts/adapter/semaforo_log.py --quiet
    python scripts/adapter/semaforo_log.py --output-dir /tmp/audit
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adapter.video_semaphore import VideoSemaphore  # noqa: E402

VERDAD_JSON = PROJECT_ROOT / "tests" / "verdad.test.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "logs"
DEFAULT_LOG_NAME = "semaforo.log"


def format_timestamp(seconds: float) -> str:
    """75.0 → ``"01:15"``, 3661.0 → ``"01:01:01"``. Negativos se clampan a 0."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def compute_transitions(
    sem: VideoSemaphore, duration_s: float
) -> List[Tuple[float, str]]:
    """Retorna ``[(t_segundos, nuevo_estado), ...]`` para cada cambio en ``[0, duration]``.

    Cada ciclo aporta 3 transiciones: ``red→green`` (inicio de ciclo),
    ``green→yellow`` y ``yellow→red``. El loop se detiene cuando el
    siguiente ciclo arrancaría más allá de la duración.
    """
    if duration_s <= 0:
        return []
    transitions: List[Tuple[float, str]] = []
    n = 0
    while True:
        cycle_start = sem.offset + n * sem.total
        for delta, new_state in (
            (0, "green"),
            (sem.green, "yellow"),
            (sem.green + sem.yellow, "red"),
        ):
            t = cycle_start + delta
            if 0 <= t <= duration_s:
                transitions.append((t, new_state))
        n += 1
        if cycle_start + sem.total > duration_s:
            break
    return transitions


def iterate_all_videos(json_path: Path) -> List[dict]:
    """Lee ``videos_verdad`` y retorna entradas válidas (g/y/r no todos cero).

    Cada entrada retornada es ``{path_name, time, green, yellow, red}``.
    Entradas con ``green=yellow=red=0`` se filtran (calibración inválida).
    Lanza ``OSError`` si el archivo no existe, ``json.JSONDecodeError``
    si está mal formado.
    """
    if not json_path.exists():
        raise OSError(f"No existe {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    result: List[dict] = []
    for entry in data.get("videos_verdad", []):
        g = int(entry.get("green", 0))
        y = int(entry.get("yellow", 0))
        r = int(entry.get("red", 0))
        if g > 0 or y > 0 or r > 0:
            result.append({
                "path_name": entry.get("path_name", "?"),
                "time": int(entry.get("time", 0)),
                "green": g,
                "yellow": y,
                "red": r,
            })
    return result


def setup_logger(log_path: Path, quiet: bool = False) -> logging.Logger:
    """Logger que escribe a archivo UTF-8 + opcional ``StreamHandler`` a stdout."""
    logger = logging.getLogger("semaforo_log")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8", errors="replace")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if not quiet:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for semaforo_log.py."""
    parser = argparse.ArgumentParser(
        description=(
            "Log de transiciones del semáforo para todos los videos "
            "de verdad.test.json (modelo determinístico)."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directorio del log (default: data/logs/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Solo escribir al log, sin stdout",
    )
    return parser


def _run_batch_mode(logger: logging.Logger, log_path: Path) -> int:
    """Itera todos los videos del JSON, escribe transiciones a log_path."""
    try:
        videos = iterate_all_videos(VERDAD_JSON)
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"No se pudo leer {VERDAD_JSON}: {e}")
        return 1

    logger.info("=" * 60)
    logger.info("=== START Semaforo Log (batch completo) ===")
    logger.info(f"Source: {VERDAD_JSON}")
    logger.info(f"Videos a procesar: {len(videos)}")
    logger.info("=" * 60)

    total_transitions = 0
    final_states = {"green": 0, "yellow": 0, "red": 0}

    for i, video in enumerate(videos, 1):
        logger.info("")
        logger.info(f"── [{i}/{len(videos)}] {video['path_name']} ──")
        cfg = {
            "green": video["green"],
            "yellow": video["yellow"],
            "red": video["red"],
            "start_offset_seconds": 0.0,
        }
        sem = VideoSemaphore(cfg)
        duration = float(video["time"])
        cycle_total = cfg["green"] + cfg["yellow"] + cfg["red"]

        logger.info(
            f"Semáforo: green={cfg['green']} yellow={cfg['yellow']} "
            f"red={cfg['red']} (ciclo={cycle_total}s)"
        )
        logger.info(f"Duración: {format_timestamp(duration)} ({duration:.1f}s)")

        transitions = compute_transitions(sem, duration)
        for t, state in transitions:
            marker = "  (inicio)" if t == 0.0 else ""
            logger.info(f"{format_timestamp(t)}  →  {state.upper():6s}{marker}")

        full_cycles = int(duration // cycle_total) if cycle_total > 0 else 0
        final_state = sem.get_state(duration) if transitions else "n/a"
        final_t = transitions[-1][0] if transitions else 0.0
        logger.info(
            f"Subtotal: {len(transitions)} transiciones, "
            f"{full_cycles} ciclos completos, estado final: {final_state.upper()}"
        )
        total_transitions += len(transitions)
        final_states[final_state] = final_states.get(final_state, 0) + 1

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        f"Total videos: {len(videos)} | "
        f"Transiciones globales: {total_transitions} | "
        f"Estados finales: GREEN={final_states['green']} "
        f"YELLOW={final_states['yellow']} RED={final_states['red']}"
    )
    logger.info(f"Log file: {log_path}")
    logger.info("=== END Semaforo Log ===")
    logger.info("=" * 60)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    log_path = Path(args.output_dir) / DEFAULT_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_path, quiet=args.quiet)

    return _run_batch_mode(logger, log_path)


if __name__ == "__main__":
    sys.exit(main())
