"""What actually gets pushed to Grafana Cloud's Prometheus.

Regression cover for a counter that carried the running total under whichever
tier was last used, so `sum by (tier) (gallery_cuts_total)` over-counted every
cut made before a tier change — the dashboard's headline panel.

Runs against the synthetic race source, so no cache, no network and no model.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.agents.orchestrator import Orchestrator
from backend.data.source import Frame
from backend.grafana.remote_write import encode


@pytest.fixture
def orch() -> Orchestrator:
    return Orchestrator()


def batch_of(orch: Orchestrator) -> dict:
    """The push batch, keyed by (metric, sorted labels)."""
    return {(name, tuple(sorted(labels.items()))): value
            for name, labels, value in orch._push_batch}


def test_cut_counters_are_per_tier_and_sum_to_the_total(orch):
    """Three cuts on adk then two on heuristic is 3 and 2, never 5 and 5."""
    for tier in ("adk", "adk", "adk", "heuristic", "heuristic"):
        orch._cuts_by_tier[tier] = orch._cuts_by_tier.get(tier, 0) + 1
        orch._cut_count += 1

    frame = next(orch.source.frames(0.1))
    orch._push_metrics(frame, [])
    b = batch_of(orch)

    assert b[("gallery_cuts_total", (("tier", "adk"),))] == 3.0
    assert b[("gallery_cuts_total", (("tier", "heuristic"),))] == 2.0
    total = sum(v for (name, _), v in b.items() if name == "gallery_cuts_total")
    assert total == float(orch._cut_count) == 5.0


def test_no_cut_counter_series_before_the_first_cut(orch):
    """An empty counter is better than a zero on a tier that never ran."""
    frame = next(orch.source.frames(0.1))
    orch._push_metrics(frame, [])
    assert not [k for k in batch_of(orch) if k[0] == "gallery_cuts_total"]


def test_track_status_and_pace_are_pushed(orch):
    frame = next(orch.source.frames(0.1))
    orch._push_metrics(frame, [])
    b = batch_of(orch)

    assert b[("gallery_track_status", (("status", "green"),))] == 1.0
    assert b[("gallery_field_pace_seconds", ())] == pytest.approx(frame.pace)
    assert frame.pace > 0, "a source that reports no pace gives gaps of zero seconds"


def test_battle_series_carry_their_identifying_labels(orch):
    frame = next(orch.source.frames(0.1))
    battles = orch.scorer.score_frame(frame)
    orch._push_metrics(frame, battles)
    b = batch_of(orch)

    pushed = [k for k in b if k[0] == "gallery_battle_tension"]
    assert len(pushed) == min(8, len(battles))
    for _, labels in pushed:
        assert {k for k, _ in labels} == {"position", "ahead", "behind"}


def test_everything_in_the_batch_encodes_to_remote_write(orch):
    """A value Prometheus rejects is indistinguishable from a broken dashboard."""
    frame = next(orch.source.frames(0.1))
    orch._push_metrics(frame, orch.scorer.score_frame(frame))

    series = [({"__name__": name, **labels}, float(value), 1700000000000)
              for name, labels, value in orch._push_batch]
    payload = encode(series)

    assert payload, "nothing encoded"
    for _, value, _ in series:
        assert np.isfinite(value), "NaN or inf reaches the wire as a rejected sample"


def test_position_change_under_caution_is_detected_but_not_counted(orch):
    """Pit shuffles behind a safety car are not passes a camera could catch.

    The detection still has to fire — an operator wants to see the order
    change — but it must stay out of the capture-rate denominator.
    """
    orch.subscribe()                       # a viewer, so accounting is live
    frames = orch.source.frames(0.1)
    orch._detect_overtakes(next(frames))   # seed the position map

    nxt = next(frames)
    nxt.cars[0].pos, nxt.cars[1].pos = nxt.cars[1].pos, nxt.cars[0].pos
    neutralised = Frame(t=nxt.t, lap=nxt.lap, cars=nxt.cars,
                        total_laps=nxt.total_laps, status="safety_car",
                        pace=nxt.pace)

    events = orch._detect_overtakes(neutralised)
    assert events, "the position change itself should still be detected"

    for _ in events:
        orch._count_capture(neutralised, live=False)
    assert orch.ot_total == 0, "caution shuffles must stay out of the denominator"


def test_green_flag_pass_is_counted(orch):
    """The same accounting under green has to actually count."""
    orch.subscribe()
    frame = next(orch.source.frames(0.1))
    assert frame.racing

    orch._count_capture(frame, live=True)
    orch._count_capture(frame, live=False)

    assert (orch.ot_total, orch.ot_caught) == (2, 1)


def test_nothing_is_counted_while_nobody_is_watching(orch):
    """The director idles with no viewers, so it cannot be charged for misses."""
    frame = next(orch.source.frames(0.1))
    assert not orch.watched

    orch._count_capture(frame, live=False)

    assert orch.ot_total == 0
