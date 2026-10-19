"""Tests for the frame-rate meter of the live demo, using a fake clock instead of a camera."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from realtime import FpsMeter


def _meter(times):
    ticks = iter(times)
    return FpsMeter(clock=lambda: next(ticks))


def test_fps_meter_reports_the_rate_of_a_steady_loop():
    """Frames every 0.1 s give 10 FPS, both smoothed and on average."""
    meter = _meter([0.0, 0.1, 0.2, 0.3])

    rates = [meter.tick() for _ in range(4)]

    assert rates[0] == 0.0
    assert rates[1:] == pytest.approx([10.0, 10.0, 10.0])
    assert meter.average() == pytest.approx(10.0)


def test_fps_meter_smooths_a_slow_frame_and_averages_over_wall_time():
    """One slow frame moves the counter only 10% of the way; the average uses the real elapsed time."""
    meter = _meter([0.0, 0.1, 0.3])

    rates = [meter.tick() for _ in range(3)]

    assert rates[2] == pytest.approx(0.9 * 10.0 + 0.1 * 5.0)
    assert meter.average() == pytest.approx(2 / 0.3)
