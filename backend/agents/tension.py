"""Deterministic battle scoring.

Runs at full tick rate (10 Hz) over all 19 adjacent pairings. No LLM here on
purpose: this is a numeric signal, it must be stable, cheap and explainable.
Gemini is only asked the question numbers cannot answer — which of these
stories is worth the cut.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass

from ..data.source import Car, Frame

DRS_RANGE = 1.0        # seconds — inside this, the follower gets DRS
ATTACK_RANGE = 2.2     # seconds — outside this there is no fight to show
HISTORY = 25           # gap samples retained per pairing (~2.5 s at 10 Hz)

# Measured over a full replay: closing rate sits at p50 0.004, p90 0.012,
# p99 0.018 s/s. Normalising by 0.015 makes a genuinely quick closure
# saturate the momentum term instead of leaving it permanently near zero.
FAST_CLOSE = 0.015

# Below this (km/h) a car is on the grid, in the pits or under a stoppage.
RACING_SPEED = 60.0


@dataclass
class Battle:
    ahead: str              # driver code of the defending car
    behind: str             # driver code of the attacking car
    ahead_num: str
    behind_num: str
    position: int           # track position being contested
    gap: float
    closing: float          # seconds per second, positive = catching
    drs: bool
    tyre_delta: int         # follower tyre age advantage, in laps
    score: float
    why: str

    def as_dict(self) -> dict:
        return asdict(self)


class TensionScorer:
    def __init__(self) -> None:
        self._hist: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=HISTORY)
        )

    @staticmethod
    def _position_weight(pos: int) -> float:
        """A fight for the lead is worth more airtime than a fight for P17.

        The floor stays high on purpose: two cars 0.06s apart is the best story
        on the circuit wherever it is happening, and an aggressive weight would
        rank a static 0.8s gap at the front above it.
        """
        if pos <= 1:
            return 1.0
        if pos <= 3:
            return 0.96
        if pos <= 6:
            return 0.90
        if pos <= 10:
            return 0.82   # points positions
        return 0.74

    def score_frame(self, frame: Frame) -> list[Battle]:
        if not frame.racing:
            # Behind a safety car the field closes to a few car lengths at well
            # under racing speed, so every pairing reads as a photo finish — and
            # not one of them is a fight, because overtaking is forbidden. Left
            # alone this produces a wall of false maximum-tension battles at
            # exactly the moment the world feed should be on the safety car.
            #
            # The per-pair RACING_SPEED gate does not catch this: a safety car
            # runs at 120-160 km/h, comfortably above it.
            #
            # History goes with it. Keeping it would difference the first green
            # sample against a caution gap several tenths smaller, and every
            # closing rate on the restart lap would spike through FAST_CLOSE.
            self._hist.clear()
            for c in frame.cars:
                c.closing = 0.0
            return []

        cars: list[Car] = frame.cars
        out: list[Battle] = []

        for i in range(1, len(cars)):
            ahead, behind = cars[i - 1], cars[i]
            key = f"{ahead.num}-{behind.num}"
            gap = behind.gap_ahead
            self._hist[key].append((frame.t, gap))

            hist = self._hist[key]
            closing = 0.0
            if len(hist) >= 4:
                (t0, g0), (t1, g1) = hist[0], hist[-1]
                if t1 > t0:
                    closing = (g0 - g1) / (t1 - t0)
            behind.closing = closing

            if gap > ATTACK_RANGE:
                continue
            # Two cars nose to tail in the pit lane, on the grid, or under a red
            # flag are not a battle. Require both to actually be racing.
            if min(ahead.speed, behind.speed) < RACING_SPEED:
                continue

            proximity = 1.0 - (gap / ATTACK_RANGE)              # 0..1
            momentum = max(0.0, min(1.0, closing / FAST_CLOSE))
            drs = gap <= DRS_RANGE
            tyre_delta = int(ahead.tyre_age - behind.tyre_age)

            score = (
                0.52 * proximity
                + 0.28 * momentum
                + 0.12 * (1.0 if drs else 0.0)
                + 0.08 * max(0.0, min(1.0, tyre_delta / 12.0))
            ) * self._position_weight(behind.pos)

            bits = [f"{gap:.2f}s"]
            if drs:
                bits.append("DRS")
            if closing > 0.006:
                bits.append(f"closing {closing:.3f}s/s")
            elif closing < -0.006:
                bits.append("dropping back")
            if tyre_delta >= 4:
                bits.append(f"tyres {tyre_delta} laps fresher")

            out.append(
                Battle(
                    ahead=ahead.code, behind=behind.code,
                    ahead_num=ahead.num, behind_num=behind.num,
                    position=ahead.pos, gap=round(gap, 3),
                    closing=round(closing, 4), drs=drs, tyre_delta=tyre_delta,
                    score=round(score, 4), why=" · ".join(bits),
                )
            )

        out.sort(key=lambda b: -b.score)
        return out
