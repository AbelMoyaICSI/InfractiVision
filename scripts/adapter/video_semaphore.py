"""
VideoSemaphore — deterministic traffic light state for CLI video processing.

Replaces the Tk/after-based Semaforo with a pure-function look-up
keyed on video timestamp. No state, no drift, testable.

Usage:
    sem = VideoSemaphore({"green": 30, "yellow": 5, "red": 40})
    assert sem.get_state(0)   == "green"
    assert sem.get_state(30)  == "yellow"
    assert sem.get_state(35)  == "red"
    assert sem.get_state(75)  == "green"
"""


class VideoSemaphore:
    def __init__(self, cycle_config: dict) -> None:
        """
        cycle_config keys:
            green  — seconds of green light (required)
            yellow — seconds of yellow light (required)
            red    — seconds of red light (required)
            start_offset_seconds — offset into video before first cycle (default 0)
        """
        self.green = int(cycle_config["green"])
        self.yellow = int(cycle_config["yellow"])
        self.red = int(cycle_config["red"])
        self.offset = float(cycle_config.get("start_offset_seconds", 0))
        self.total = self.green + self.yellow + self.red

    def get_state(self, video_second: float) -> str:
        """Return 'green' | 'yellow' | 'red' for a given video timestamp (seconds)."""
        t = (video_second - self.offset) % self.total
        if t < 0:
            t += self.total
        if t < self.green:
            return "green"
        t -= self.green
        if t < self.yellow:
            return "yellow"
        return "red"

    def get_cycle_durations(self) -> dict:
        return {"green": self.green, "yellow": self.yellow, "red": self.red}
