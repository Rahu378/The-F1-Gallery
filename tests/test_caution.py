"""Gap and tension behaviour under a safety car.

Regression cover for the bug where gaps were converted from race distance to
seconds with the session median lap time. Under a full-course yellow the field
runs well off that pace, so every gap came out short, the whole field read as
nose to tail, and the tension scorer produced a wall of false maximum-score
battles at the moment the feed should have been on the safety car.

These run without FastF1 or a race cache: the inputs are hand-built grids and
frames, which is also the only way to test a safety car deterministically.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.agents.tension import ATTACK_RANGE, FAST_CLOSE, TensionScorer
from backend.data.source import (
    NEUTRALISED,
    TRACK_STATUS,
    Car,
    Frame,
    field_pace,
    status_series,
)

GREEN_LAP = 90.0


def car(num: str, gap: float, speed: float = 200.0, pos: int = 1,
        tyre_age: int = 10) -> Car:
    c = Car(num=num, code=f"D{num}", team="T", color="#fff")
    c.gap_ahead = gap
    c.speed = speed
    c.pos = pos
    c.tyre_age = tyre_age
    return c


def frame(gaps: list[float], t: float = 0.0, status: str = "green",
          speed: float = 200.0) -> Frame:
    cars = [car(str(i), gap=g, speed=speed, pos=i + 1)
            for i, g in enumerate(gaps)]
    return Frame(t=t, lap=10, cars=cars, total_laps=53,
                 status=status, pace=GREEN_LAP)


# --------------------------------------------------------------- status_series
def test_status_series_is_a_step_function():
    """A status holds until the next message, and the grid takes the last one."""
    times = np.array([0.0, 100.0, 200.0])
    codes = ["1", "4", "1"]          # green, safety car, green
    grid = np.array([50.0, 99.9, 100.0, 150.0, 199.9, 200.0, 250.0])

    out = status_series(times, codes, grid)

    assert list(out) == ["green", "green", "safety_car", "safety_car",
                         "safety_car", "green", "green"]


def test_status_series_before_first_message_falls_back_to_green():
    out = status_series(np.array([100.0]), ["4"], np.array([0.0, 50.0, 100.0]))
    assert list(out) == ["green", "green", "safety_car"]


def test_status_series_handles_unsorted_channel():
    times = np.array([200.0, 0.0, 100.0])
    codes = ["1", "1", "4"]
    out = status_series(times, codes, np.array([50.0, 150.0, 250.0]))
    assert list(out) == ["green", "safety_car", "green"]


def test_status_series_with_no_messages_is_all_green():
    out = status_series(np.array([]), [], np.array([0.0, 1.0, 2.0]))
    assert list(out) == ["green", "green", "green"]


def test_unknown_status_code_does_not_neutralise():
    """A code we cannot identify must not suspend the feed's scoring."""
    out = status_series(np.array([0.0]), ["99"], np.array([1.0]))
    assert out[0] == "green"
    assert out[0] not in NEUTRALISED


def test_vsc_ending_is_still_neutralised():
    """Code 7 is "VSC ending" — overtaking is still forbidden until green."""
    assert TRACK_STATUS["7"] == "vsc"
    assert TRACK_STATUS["7"] in NEUTRALISED
    assert TRACK_STATUS["2"] not in NEUTRALISED   # a local yellow is still racing


# ------------------------------------------------------------------ field_pace
def test_field_pace_recovers_a_steady_lap_time():
    step, n = 0.5, 400
    # 20 cars all lapping in exactly GREEN_LAP seconds.
    t = np.arange(n) * step
    prog = np.tile(t / GREEN_LAP, (20, 1))

    pace = field_pace(prog, step, fallback=GREEN_LAP)

    assert pace.shape == (n,)
    assert np.allclose(pace, GREEN_LAP, rtol=1e-6)


def test_field_pace_tracks_a_slowdown():
    """This is the bug: under caution the field runs slower and the gap
    conversion has to follow it."""
    step, n = 0.5, 800
    t = np.arange(n) * step
    sc_lap = GREEN_LAP * 1.6                     # safety car pace
    rate = np.where(t < 200.0, 1.0 / GREEN_LAP, 1.0 / sc_lap)
    prog = np.tile(np.concatenate([[0.0], np.cumsum(rate[:-1] * step)]), (20, 1))

    pace = field_pace(prog, step, fallback=GREEN_LAP, window_s=20.0)

    assert pace[100] == pytest.approx(GREEN_LAP, rel=0.02)     # green
    assert pace[700] == pytest.approx(sc_lap, rel=0.02)        # under caution
    # A session median would have held GREEN_LAP throughout and understated
    # every gap by this factor.
    assert pace[700] / pace[100] == pytest.approx(1.6, rel=0.05)


def test_field_pace_ignores_stationary_cars():
    """A car parked in the pits must not drag the field's pace toward zero."""
    step, n = 0.5, 400
    t = np.arange(n) * step
    prog = np.tile(t / GREEN_LAP, (20, 1))
    prog[3] = 0.0            # retired on lap 1
    prog[7] = 0.0

    pace = field_pace(prog, step, fallback=GREEN_LAP)

    assert np.allclose(pace, GREEN_LAP, rtol=1e-6)


def test_field_pace_clamps_a_stopped_field_instead_of_dividing_by_zero():
    """A red flag takes every rate to zero; the reciprocal must stay finite."""
    step, n = 0.5, 200
    prog = np.zeros((20, n))

    pace = field_pace(prog, step, fallback=GREEN_LAP, hi_mult=4.0)

    assert np.all(np.isfinite(pace))
    assert np.all(pace <= 4.0 * GREEN_LAP)


def test_field_pace_survives_a_single_sample_grid():
    pace = field_pace(np.zeros((20, 1)), 0.5, fallback=GREEN_LAP)
    assert np.all(pace == GREEN_LAP)


# --------------------------------------------------------------- tension gating
def test_bunched_field_under_safety_car_scores_nothing():
    """The regression. Twenty cars two tenths apart is not twenty battles."""
    gaps = [0.0] + [0.2] * 19
    scorer = TensionScorer()

    assert scorer.score_frame(frame(gaps, status="safety_car")) == []
    assert scorer.score_frame(frame(gaps, status="vsc")) == []
    assert scorer.score_frame(frame(gaps, status="red")) == []

    # Same geometry under green is exactly what the scorer is for.
    assert len(scorer.score_frame(frame(gaps, status="green"))) == 19


def test_local_yellow_keeps_racing():
    """A yellow covers one sector; the rest of the lap is still a race."""
    scorer = TensionScorer()
    out = scorer.score_frame(frame([0.0, 0.4], status="yellow"))
    assert len(out) == 1


def test_caution_clears_stale_closing_rates():
    scorer = TensionScorer()
    scorer.score_frame(frame([0.0, 1.0], t=0.0))
    f = frame([0.0, 0.4], t=1.0, status="safety_car")
    scorer.score_frame(f)
    assert all(c.closing == 0.0 for c in f.cars)


def test_restart_does_not_spike_closing_rate_across_the_seam():
    """Gap history must not survive a caution.

    Under the safety car the field closes to a couple of tenths. If those
    samples are still in the buffer at the restart, the first green frame is
    differenced against them and the closing rate reads as a violent lunge on
    every car in the field.
    """
    scorer = TensionScorer()

    # Racing, a second apart and stable.
    for k in range(6):
        scorer.score_frame(frame([0.0, 1.0], t=k * 0.1))

    # Safety car: bunched up to 0.2s over two seconds.
    for k in range(20):
        scorer.score_frame(frame([0.0, 0.2], t=0.6 + k * 0.1, status="safety_car"))

    # Green flag, and the field spreads back out to a second.
    out = []
    for k in range(6):
        out = scorer.score_frame(frame([0.0, 1.0], t=2.6 + k * 0.1))

    assert out, "should be scoring again once green"
    # Without the history clear this came out strongly negative (dropping back
    # from a caution gap); what matters is that it is not a fabricated spike.
    assert abs(out[0].closing) < FAST_CLOSE


def test_pairings_outside_attack_range_are_still_dropped_under_green():
    scorer = TensionScorer()
    out = scorer.score_frame(frame([0.0, ATTACK_RANGE + 0.5]))
    assert out == []


def test_pit_lane_pair_is_not_a_battle():
    """The existing per-pair speed gate still applies under green."""
    scorer = TensionScorer()
    out = scorer.score_frame(frame([0.0, 0.3], speed=40.0))
    assert out == []
