"""
test_frame_reader.py — Unit tests for FrameReader.

The reader is hard to test with a real cv2.VideoCapture in unit
tests (file I/O, codec availability, etc.), so we inject a fake
capture that mimics the relevant subset of the API.

Covers:
  - Frames flow from the capture to the consumer.
  - EOF is signaled with None and the reader thread exits.
  - Queue size is bounded by maxsize.
  - When the consumer is slow, frames are dropped and the counter
    increments (the desired back-pressure behavior).
  - stop() is non-blocking and the thread joins.
  - cap.release() is NOT called by the reader.
  - maxsize validation.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pytest

from scripts.adapter.frame_reader import FrameReader


class FakeCapture:
    """Minimal cv2.VideoCapture stand-in for tests.

    Yields a pre-loaded list of frames on each `read()` call, then
    returns `(False, None)` to signal EOF. Records `release()` calls
    so we can assert the reader does NOT release the cap.
    """

    def __init__(self, frames: List[np.ndarray],
                 read_delay_s: float = 0.0) -> None:
        self._frames = list(frames)
        self._idx = 0
        self._read_delay_s = read_delay_s
        self.released = False
        self.read_count = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        self.read_count += 1
        if self._read_delay_s:
            time.sleep(self._read_delay_s)
        if self._idx >= len(self._frames):
            return False, None
        f = self._frames[self._idx]
        self._idx += 1
        return True, f

    def release(self) -> None:
        self.released = True


def _make_frames(n: int, h: int = 4, w: int = 4) -> List[np.ndarray]:
    """Build n distinguishable BGR frames (each with a unique marker)."""
    return [np.full((h, w, 3), fill_value=i % 256, dtype=np.uint8)
            for i in range(n)]


# ── Lifecycle / basic flow ────────────────────────────────────────────


class TestFrameReaderBasic:
    def test_starts_and_delivers_all_frames(self):
        frames = _make_frames(5)
        cap = FakeCapture(frames)
        # Tight eof timeout so the test fails fast if EOF delivery
        # breaks. 0.2s is plenty for a consumer reading back-to-back.
        reader = FrameReader(cap, maxsize=2, eof_put_timeout_s=0.2).start()
        try:
            # Simulate real consumer work: 1ms between reads. This
            # gives the reader time to deliver the EOF sentinel
            # without the consumer's tight loop starving it.
            delivered = []
            for _ in range(5):
                delivered.append(reader.read())
                time.sleep(0.001)
        finally:
            reader.stop()
        # All five frames must arrive, in order, intact.
        for got, expected in zip(delivered, frames):
            assert got is not None
            assert np.array_equal(got, expected)
        assert reader.frames_delivered == 5
        # Some frames may have been dropped at the start because the
        # queue filled before the consumer started reading. That's
        # fine — the test only requires that all CONSUMED frames
        # are correct and in order.

    def test_eof_signals_none_and_exits(self):
        cap = FakeCapture(_make_frames(3))
        reader = FrameReader(cap, eof_put_timeout_s=0.2).start()
        try:
            for _ in range(3):
                f = reader.read()
                assert f is not None
                time.sleep(0.001)  # let the reader produce the sentinel
            eof = reader.read()
        finally:
            reader.stop()
        assert eof is None
        # The reader thread should have exited on its own after EOF.
        # Give it a moment to wind down.
        time.sleep(0.05)
        assert not reader.is_alive()

    def test_does_not_release_capture(self):
        """The caller still owns the cap. Reader must NOT release it."""
        cap = FakeCapture(_make_frames(2))
        reader = FrameReader(cap, eof_put_timeout_s=0.2).start()
        try:
            assert reader.read() is not None
            time.sleep(0.001)
            assert reader.read() is not None
        finally:
            reader.stop()
        assert cap.released is False

    def test_start_twice_raises(self):
        cap = FakeCapture(_make_frames(1))
        reader = FrameReader(cap, eof_put_timeout_s=0.2).start()
        try:
            with pytest.raises(RuntimeError):
                reader.start()
        finally:
            reader.stop()

    def test_maxsize_validation(self):
        cap = FakeCapture([])
        with pytest.raises(ValueError):
            FrameReader(cap, maxsize=0)
        with pytest.raises(ValueError):
            FrameReader(cap, maxsize=-1)

    def test_eof_timeout_validation(self):
        cap = FakeCapture([])
        with pytest.raises(ValueError):
            FrameReader(cap, eof_put_timeout_s=-1)


# ── Back-pressure / drop behavior ────────────────────────────────────


class TestFrameReaderBackPressure:
    def test_drops_frames_when_consumer_is_genuinely_stuck(self):
        """Drops happen only when the consumer is blocked for longer
        than the frame_put_timeout_s. With a short timeout and a
        long consumer sleep, at least one drop must be recorded."""
        frames = _make_frames(20)
        cap = FakeCapture(frames, read_delay_s=0.001)
        # Very short frame_put_timeout so the test fails fast.
        reader = FrameReader(
            cap,
            maxsize=2,
            frame_put_timeout_s=0.1,
            eof_put_timeout_s=0.1,
        ).start()
        try:
            # Consume only the first 2 frames (fills the queue), then
            # sleep WAY longer than frame_put_timeout. The reader
            # will be blocked on the third put and time out, dropping
            # the frame.
            reader.read()
            reader.read()
            time.sleep(0.3)
        finally:
            reader.stop()

        # We consumed 2 frames. The reader tried to put a third and
        # timed out, so at least one drop should be recorded.
        assert reader.frames_delivered == 2
        assert reader.frames_dropped >= 1

    def test_queue_size_stays_bounded(self):
        """Even with a fast producer and a stalled consumer, the
        queue must never exceed maxsize."""
        frames = _make_frames(50)
        cap = FakeCapture(frames, read_delay_s=0.001)
        reader = FrameReader(
            cap,
            maxsize=2,
            frame_put_timeout_s=0.05,
            eof_put_timeout_s=0.1,
        ).start()
        try:
            time.sleep(0.2)  # let the reader run ahead
            assert reader.qsize() <= 2
        finally:
            reader.stop()

    def test_no_drops_when_consumer_keeps_up(self):
        """If the consumer drains the queue as fast as the reader
        fills it, no frames should be dropped and all frames
        delivered."""
        frames = _make_frames(5)
        cap = FakeCapture(frames)
        reader = FrameReader(
            cap,
            maxsize=2,
            frame_put_timeout_s=0.2,
            eof_put_timeout_s=0.2,
        ).start()
        try:
            # Drain one frame at a time with a short pause, mimicking
            # a real consumer doing inference between reads.
            for _ in range(5):
                assert reader.read() is not None
                time.sleep(0.005)
        finally:
            reader.stop()
        assert reader.frames_delivered == 5
        assert reader.frames_dropped == 0


# ── stop() behavior ──────────────────────────────────────────────────


class TestFrameReaderStop:
    def test_stop_unblocks_consumer(self):
        """If the consumer is blocked in read() and the reader thread
        exits, read() must return None within the eof timeout."""
        # Infinite-ish capture that never EOFs on its own.
        class NeverEnding:
            def __init__(self):
                self.i = 0

            def read(self):
                time.sleep(0.01)
                self.i += 1
                return True, np.zeros((2, 2, 3), dtype=np.uint8)

            def release(self):
                pass

        reader = FrameReader(NeverEnding(), maxsize=2,
                             eof_put_timeout_s=0.5).start()
        # Read a few frames first so the reader is "in flight"
        for _ in range(2):
            reader.read()
        reader.stop()
        # After stop, the next read() must unblock within ~0.5s
        # (the reader's eof_put_timeout_s). We allow a generous
        # overall bound so test scheduling jitter does not flake.
        t0 = time.perf_counter()
        result = reader.read()
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"read() took {elapsed:.2f}s after stop()"

    def test_stop_is_idempotent(self):
        cap = FakeCapture(_make_frames(1))
        reader = FrameReader(cap, eof_put_timeout_s=0.2).start()
        reader.stop()
        # Calling stop() a second time must not raise.
        reader.stop()


# ── Thread hygiene ────────────────────────────────────────────────────


class TestFrameReaderThreading:
    def test_thread_is_daemon(self):
        cap = FakeCapture(_make_frames(1))
        reader = FrameReader(cap)
        assert reader._thread.daemon is True

    def test_thread_has_descriptive_name(self):
        cap = FakeCapture(_make_frames(1))
        reader = FrameReader(cap)
        assert reader._thread.name == "FrameReader"


# ── Error handling ────────────────────────────────────────────────────


class ExplodingCapture:
    """A capture whose read() raises after a few successful frames.

    Used to verify the reader captures the exception in `last_error`
    and exits cleanly instead of crashing the daemon thread.
    """

    def __init__(self, frames_before_explode: int, exc: BaseException):
        self._remaining = frames_before_explode
        self._exc = exc

    def read(self):
        if self._remaining > 0:
            self._remaining -= 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        raise self._exc

    def release(self):
        pass


class TestFrameReaderErrorCapture:
    def test_exception_in_cap_read_is_captured(self):
        boom = ExplodingCapture(
            frames_before_explode=2,
            exc=RuntimeError("codec glitch"),
        )
        reader = FrameReader(boom, eof_put_timeout_s=0.2).start()
        try:
            # First two reads succeed.
            assert reader.read() is not None
            time.sleep(0.001)
            assert reader.read() is not None
            # Third read: the thread crashed, so the polling loop
            # returns None once the reader is dead.
            assert reader.read() is None
        finally:
            reader.stop()
        assert isinstance(reader.last_error, RuntimeError)
        assert "codec glitch" in str(reader.last_error)

    def test_no_error_on_clean_run(self):
        cap = FakeCapture(_make_frames(2))
        reader = FrameReader(cap, eof_put_timeout_s=0.2).start()
        try:
            assert reader.read() is not None
            time.sleep(0.001)
            assert reader.read() is not None
            assert reader.read() is None  # clean EOF
        finally:
            reader.stop()
        assert reader.last_error is None

    def test_cpp_exception_also_captured(self):
        """OpenCV can throw C++ exceptions that surface as generic
        Python errors. The reader must still capture and exit."""
        boom = ExplodingCapture(
            frames_before_explode=1,
            exc=Exception("Unknown C++ exception from OpenCV code"),
        )
        reader = FrameReader(boom, eof_put_timeout_s=0.2).start()
        try:
            assert reader.read() is not None
            time.sleep(0.001)
            assert reader.read() is None
        finally:
            reader.stop()
        assert reader.last_error is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
