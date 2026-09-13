"""
threads.py — Centralized thread budget for the CLI pipeline.

Why this exists:
  The CLI scripts run on a 4-core / 4-thread CPU (i3-9100F + RTX 5050)
  where OpenCV decode, PyTorch inference and the Python tracker all
  compete for the same physical cores. Without explicit thread caps the
  three runtimes oversubscribe the 4 physical cores and silently thrash,
  starving the GPU feed (decode/UI) while the GPU idles.

This module gives a single place to set the budget, idempotently, so the
caller (usually `main()` of a CLI script) does not have to repeat five
unrelated lines and so the numbers can be tuned empirically in one spot.
"""

from __future__ import annotations

import os


def configure_thread_budget(
    cv_threads: int = 2,
    torch_threads: int = 2,
) -> dict:
    """
    Pin the number of threads used by OpenCV, PyTorch and the BLAS backend.

    Defaults are tuned for i3-9100F 4C/4T + discrete NVIDIA (GTX 1650 Ti /
    RTX 5050), where inference runs on GPU (FP16) and CPU only feeds
    decode + pre/post + Tk:
      - OpenCV: 2 threads (decode + resize, leaves room for Tk/worker)
      - PyTorch: 2 threads (CPU pre/post only; GPU does the matmuls)

    Total torch+cv2 <= 4: no quitarle CPU a la GPU ni al decode.

    On any other layout (more cores, GPU available) the caller can pass
    different values. The function returns the values that were actually
    set so they can be logged.

    Safe to call multiple times: it just overwrites the same env vars /
    torch globals. It does NOT raise if a backend is missing — the goal
    is best-effort configuration, not a hard contract.

    Note on `set_num_interop_threads`:
        It is intentionally NOT called here. That function can only be
        invoked before torch has started its parallel runtime, and any
        subsequent call may abort the process. Callers that need it must
        do so at the very top of their `main()` (before any torch import
        has had a chance to spawn workers). For a single-stream inference
        loop like ours, the intra-op pool (`set_num_threads`) is what
        actually matters.
    """
    applied: dict = {}

    # 1. OpenCV (FFmpeg-backed decode path is multi-threaded internally;
    #    this caps OpenCV's own thread pool on top of that).
    try:
        import cv2

        cv2.setNumThreads(cv_threads)
        applied["cv_threads"] = cv_threads
    except Exception:
        applied["cv_threads"] = None

    # 2. PyTorch (intra-op = the compute pool). This is the lever that
    #    actually controls inference speed on CPU.
    try:
        import torch

        torch.set_num_threads(torch_threads)
        applied["torch_threads"] = torch_threads
    except Exception:
        applied["torch_threads"] = None

    # 3. BLAS / OMP env vars. These must be set BEFORE the BLAS library
    #    loads, so we set them every call but they only take effect on
    #    a fresh process. We still set them so that subprocesses spawned
    #    later (e.g. multiprocessing pools) inherit the right budget.
    for var, value in (("OMP_NUM_THREADS", str(torch_threads)),
                       ("MKL_NUM_THREADS", str(torch_threads))):
        os.environ[var] = str(value)
        applied[var.lower()] = value

    return applied
