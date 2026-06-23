"""Audio multiplataforma — wrapper sobre `winsound` con fallbacks para Linux/macOS.

Uso:
    from src.core.utils.audio import play_beep, play_sequence

    play_beep(1000, 150)               # 1 kHz durante 150 ms
    play_sequence([(800,150),(1200,200)])

Estrategia por plataforma:
    * Windows: usa `winsound.Beep` (síncrono, mismo comportamiento que antes).
    * Linux:   usa `os.system("paplay ...")` con un archivo /usr/share/sounds o
               imprime BEL (`\\a`) como último recurso silencioso.
    * macOS:   usa `os.system("afplay ...")` o BEL.

El módulo nunca lanza excepciones al exterior: cualquier fallo se silencia y se
registra en stderr una sola vez (para no saturar el log).
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, Tuple

_IS_WINDOWS = sys.platform.startswith("win")
_IS_MAC = sys.platform == "darwin"
_AUDIO_WARNED = False


def _warn_once(msg: str) -> None:
    global _AUDIO_WARNED
    if not _AUDIO_WARNED:
        print(f"[audio] {msg}", file=sys.stderr)
        _AUDIO_WARNED = True


def play_beep(frequency: int = 1000, duration_ms: int = 150) -> None:
    """Reproduce un beep simple sin bloquear la app si falla.

    `frequency` y `duration_ms` se usan tal cual en Windows. En otras
    plataformas se ignoran y se reproduce un beep genérico (o BEL).
    """
    try:
        if _IS_WINDOWS:
            import winsound  # import perezoso: solo en Windows
            winsound.Beep(int(frequency), int(duration_ms))
            return

        # POSIX: intentamos comando del sistema; si no, BEL ASCII.
        if _IS_MAC:
            # /System/Library/Sounds/Tink.aiff existe en macOS por defecto
            ret = os.system("afplay /System/Library/Sounds/Tink.aiff >/dev/null 2>&1 &")
            if ret != 0:
                sys.stdout.write("\a")
                sys.stdout.flush()
            return

        # Linux y otros UNIX: probamos paplay/aplay con sonido del sistema.
        for cmd in (
            "paplay /usr/share/sounds/freedesktop/stereo/bell.oga",
            "aplay -q /usr/share/sounds/alsa/Front_Center.wav",
        ):
            if os.system(cmd + " >/dev/null 2>&1 &") == 0:
                return
        # Último recurso: BEL en terminal (silencioso si no hay TTY).
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception as e:  # noqa: BLE001
        _warn_once(f"beep fallido ({e}); audio deshabilitado")


def play_sequence(notes: Iterable[Tuple[int, int]]) -> None:
    """Reproduce una secuencia de (frecuencia, duración_ms)."""
    for freq, dur in notes:
        play_beep(freq, dur)


__all__ = ["play_beep", "play_sequence"]
