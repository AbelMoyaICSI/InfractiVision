"""Deterministic frame schedule for official red-light analysis."""
from __future__ import annotations


class TrafficProcessingPlanner:
    def __init__(self, green: float, yellow: float, red: float, fps: float,
                 pre_red_seconds: float = 0.5, green_skip_rate: int = 60):
        self.green = float(green)
        self.yellow = float(yellow)
        self.red = float(red)
        self.fps = max(float(fps), 1.0)
        self.pre_red_seconds = max(float(pre_red_seconds), 0.0)
        self.green_skip_rate = max(int(green_skip_rate), 1)
        self.cycle = self.green + self.yellow + self.red

    def state_at(self, frame_index: int) -> str:
        t = (frame_index / self.fps) % self.cycle
        if t < self.green:
            return "green"
        if t < self.green + self.yellow:
            return "yellow"
        return "red"

    def should_detect(self, frame_index: int) -> bool:
        """Return whether vehicle detection is allowed for this frame."""
        t = (frame_index / self.fps) % self.cycle
        red_start = self.green + self.yellow
        return t >= red_start - self.pre_red_seconds

    def should_display(self, frame_index: int) -> bool:
        """Return whether a frame should be shown/written to the fast preview."""
        return self.should_detect(frame_index) or frame_index % self.green_skip_rate == 0

    def should_process(self, frame_index: int) -> bool:
        """Backward-compatible alias for detection scheduling."""
        return self.should_detect(frame_index)

    def intensity_at(self, frame_index: int) -> str:
        t = (frame_index / self.fps) % self.cycle
        red_start = self.green + self.yellow
        if t >= red_start or t >= red_start - self.pre_red_seconds:
            return "full"
        return "display_only"
