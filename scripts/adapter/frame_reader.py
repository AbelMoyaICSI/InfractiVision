"""
frame_reader.py — Single-threaded background frame producer.

Why this exists:
  On a 4-core / 8-thread CPU, every Python thread that does work
  steals cycles from the inference runtime. That makes the design
  trade-off very different from a GPU pipeline, where adding a
  background reader is "free" (the CPU was idle waiting on the GPU).

  Here we still want a background reader — to overlap decode with
  inference — but with very tight constraints:
    - ONE single reader thread (a pool would just fight for cores).
    - A bounded queue (maxsize=2). A larger queue would let the
      reader race ahead of the consumer and silently accumulate
      *stale* frames in RAM, which looks like "fluid" processing
      but is actually showing the user outdated data.
    - If the consumer can't keep up, the reader drops the newest
      frame (the one that would block) and keeps going. This is
      the right trade-off: stale frames are worse than missing
      frames, and the next frame from the video is more useful
      than the dropped one.

  OpenCV's `VideoCapture` already uses FFmpeg's internal thread
  pool to parallelize decode across cores, so adding more Python
  threads on top of that gives no benefit and only causes
  oversubscription.

The reader takes ownership of nothing: the caller still opens and
releases the `cv2.VideoCapture`. The reader just borrows it.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import cv2
import numpy as np


# Sentinel pushed into the queue to signal end-of-stream to the
# consumer. The natural value is `None` because frames are never
# None (they are always np.ndarrays).
_EOF: Optional[np.ndarray] = None


class FrameReader:
    """Reads frames from a `cv2.VideoCapture` in a background thread.

    The reader thread blocks on `cap.read()` (which itself is blocking
    on FFmpeg). When a frame is available it is placed into a bounded
    queue. The consumer pulls frames with `read()`. On EOF the reader
    pushes `None` and exits; the consumer sees `None` from `read()`
    and should break its processing loop.

    The reader does NOT release the `cap` — that is the caller's
    responsibility. `stop()` only signals the reader to exit its
    loop as soon as possible.
    """

    def __init__(self, cap: cv2.VideoCapture, maxsize: int = 2,
                 frame_put_timeout_s: float = 1.0,
                 eof_put_timeout_s: float = 5.0) -> None:
        """
        Args:
            cap: an already-opened cv2.VideoCapture. The reader does
                not own it; the caller must `cap.release()` after
                `stop()`.
            maxsize: upper bound on the queue. Defaults to 2, which
                is enough to overlap decode with one batch of
                inference but small enough to avoid accumulating
                stale frames.
            frame_put_timeout_s: how long the reader blocks waiting
                for queue space to push a regular frame. If the
                consumer cannot drain in that time, the frame is
                dropped and `frames_dropped` is incremented. Default
                1s is generous enough that a normally-paced consumer
                never sees drops.
            eof_put_timeout_s: how long the reader blocks waiting
                for queue space to push the EOF sentinel. If this
                also times out, the reader exits and the consumer's
                `read()` polling loop will notice the dead thread.
        """
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        if frame_put_timeout_s < 0:
            raise ValueError(
                f"frame_put_timeout_s must be >= 0, got {frame_put_timeout_s}")
        if eof_put_timeout_s < 0:
            raise ValueError(
                f"eof_put_timeout_s must be >= 0, got {eof_put_timeout_s}")
        self.cap = cap
        self._q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=maxsize)
        self._frame_put_timeout_s = frame_put_timeout_s
        self._eof_put_timeout_s = eof_put_timeout_s
        self._stopped = False
        self._stop_lock = threading.Lock()
        # frames_dropped counts how many times the consumer was too
        # slow and we had to skip a freshly-decoded frame. Useful for
        # diagnosing pipeline imbalance.
        self.frames_dropped = 0
        # frames_delivered counts how many frames the consumer has
        # actually read. Together with frames_dropped, it accounts
        # for every frame the reader pulled from the cap.
        self.frames_delivered = 0
        # If cap.read() raised in the reader thread, the exception
        # is stored here so the main thread can surface it instead of
        # silently treating the error as EOF.
        self._last_error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="FrameReader")

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> "FrameReader":
        """Start the background reader thread.

        Returns self so the caller can chain:
            reader = FrameReader(cap).start()
        """
        if self._thread.is_alive():
            raise RuntimeError("FrameReader is already started")
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the reader to exit and wait briefly for the thread.

        The thread is a daemon, so even if `timeout` is exceeded the
        process can still exit cleanly. We just give it a chance to
        drain the queue and observe the flag.

        Does NOT release the underlying cv2.VideoCapture.
        """
        with self._stop_lock:
            self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        # Drain any pending frames so the consumer does not see them
        # after a stop() (mostly defensive — callers should also
        # break out of their loop on stop).
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _is_alive_unlocked(self) -> bool:
        """Cheap alive check used inside read()'s polling loop.

        Identical to `is_alive()` but named differently to make it
        obvious in the polling loop that we are reading a flag that
        the reader thread can flip at any time. (A torn read is not
        a correctness issue here — if we miss the alive→dead
        transition by a cycle, the next poll iteration will catch
        it.)
        """
        return self._thread.is_alive()

    # ── consumer API ────────────────────────────────────────────────

    def read(self, poll_interval_s: float = 0.1) -> Optional[np.ndarray]:
        """Block until a frame is available, or return None on EOF.

        Uses a short polling loop instead of `q.get()` with no timeout
        so that if the reader thread dies while the queue is empty
        (e.g. it could not push the EOF sentinel because the consumer
        was too slow, or it crashed on cap.read()), the consumer
        still unblocks and returns None.

        Args:
            poll_interval_s: how often to wake up and check whether
                the reader is still alive. Default 100ms — short
                enough that EOF / stop is noticed quickly, long
                enough that we are not burning CPU spinning.

        Returns:
            - A numpy.ndarray frame (BGR) when the reader is keeping
              up.
            - None when the video has been fully read OR the reader
              was stopped OR the reader died (possibly because
              cap.read() raised). The caller should break its loop
              and check `last_error` if it wants to distinguish
              "clean EOF" from "crash".
        """
        while True:
            try:
                frame = self._q.get(timeout=poll_interval_s)
                if frame is not None:
                    self.frames_delivered += 1
                return frame
            except queue.Empty:
                # No frame in `poll_interval_s`. If the reader is
                # still alive, just retry. If the reader is gone, the
                # queue will never be filled again, so we have hit
                # end-of-stream and the caller should stop.
                if not self._is_alive_unlocked():
                    return None

    def qsize(self) -> int:
        """How many frames are currently buffered. Useful for tests."""
        return self._q.qsize()

    @property
    def last_error(self) -> Optional[BaseException]:
        """The exception that killed the reader thread, if any.

        None while the thread is alive or after a clean EOF. After
        `read()` returns None, the caller can inspect this to decide
        whether the run ended naturally (last_error is None) or the
        decoder crashed (last_error is set).
        """
        return self._last_error

    # ── internal ────────────────────────────────────────────────────

    def _is_stopped(self) -> bool:
        # Cheap snapshot of the flag, with a lock so the writer
        # (stop()) and reader (_run) cannot tear on weakly-ordered
        # platforms. CPython's GIL usually makes this redundant, but
        # being explicit costs nothing.
        with self._stop_lock:
            return self._stopped

    def _run(self) -> None:
        while not self._is_stopped():
            try:
                ret, frame = self.cap.read()
            except Exception as exc:
                # OpenCV can throw a `cv2.error` (or any C++ exception
                # surfacing as a generic Python exception) from a
                # background thread when the cap is in a bad state
                # (already released, codec glitch, etc.). A crashing
                # thread here would print a noisy traceback and
                # leave the consumer hanging; instead we capture the
                # error, signal EOF, and let the pipeline wind down
                # gracefully.
                self._last_error = exc
                break
            # If stop() was called between cap.read() returning and
            # us touching the queue, bail out before pushing.
            if self._is_stopped():
                break
            if not ret:
                self._signal_eof()
                break
            try:
                # Blocking put with timeout: prefer waiting briefly
                # for the consumer to drain over silently dropping
                # a decoded frame. The timeout bounds the worst-case
                # stall if the consumer is genuinely stuck.
                self._q.put(frame, timeout=self._frame_put_timeout_s)
            except queue.Full:
                # The consumer is too slow. Drop the freshly decoded
                # frame and keep going. The alternative — waiting
                # indefinitely — would cause OpenCV's internal
                # buffers to fill up and back-pressure the entire
                # decode path, which is worse for everyone.
                self.frames_dropped += 1
        # Always signal EOF on the way out, even if the loop was
        # broken by stop() or by an exception in cap.read(). Otherwise
        # a consumer blocked in read() would wait forever for a
        # sentinel that will never arrive.
        self._signal_eof()

    def _signal_eof(self) -> None:
        """Push the EOF sentinel, blocking briefly if the queue is full.

        Best-effort: if the consumer is too slow to drain, the sentinel
        put will time out. The consumer's `read()` will detect that
        the reader thread is dead and return None on its own, so the
        sentinel not arriving is not catastrophic — just slower to
        notice.
        """
        try:
            self._q.put(_EOF, timeout=self._eof_put_timeout_s)
        except queue.Full:
            # Consumer is too slow to drain. The reader will exit
            # and the consumer's read() polling loop will notice.
            pass
