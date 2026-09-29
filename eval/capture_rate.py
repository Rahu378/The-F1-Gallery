"""Does the director actually catch the moments?

The point of this project is that a pass which happens off screen is lost. So
the measure that matters is not latency or uptime — it is what fraction of the
overtakes in a race the feed was actually watching when they happened.

This replays a full Grand Prix under several cut policies and counts. No model
calls: the policies compared here are the deterministic ones, because they are
what decides *where the camera is*. The Gemini director chooses which of the
candidate battles is the better story; the candidate set, and therefore the
capture rate, comes from the tension scorer and the cut guard. Measuring those
in isolation keeps the number reproducible and free.

Usage:  .venv/bin/python -m eval.capture_rate [--race monza] [--policies ...]
"""
from __future__ import annotations

import argparse
import asyncio
import random
from dataclasses import dataclass

from backend.agents.cutguard import CutGuard
from backend.agents.tension import TensionScorer
from backend.data.source import build_source


@dataclass
class Result:
    policy: str
    overtakes: int
    captured: int
    cuts: int

    @property
    def rate(self) -> float:
        return self.captured / self.overtakes if self.overtakes else 0.0


def collect_overtakes(race: str, speed: float = 40.0, tick_hz: float = 10.0,
                      laps: int = 0) -> list[tuple]:
    """Every position change in the race, as (tick, carA, carB).

    Used to compute the ceiling: with one camera and a minimum hold, some
    overtakes are simply not catchable — two passes seconds apart at opposite
    ends of the field cannot both be shown. Reporting a raw percentage without
    that context understates the result.
    """
    src = build_source(race)
    dt = 1.0 / tick_hz
    out, last = [], {}
    for i, frame in enumerate(src.frames(dt * speed)):
        if laps and frame.lap > laps:
            break
        # Positions change hands under a safety car too, but those are pit-stop
        # shuffles rather than passes any camera could have caught. The live
        # orchestrator excludes them from its capture rate; this has to match it
        # or the two report different numbers for the same race.
        if not frame.racing:
            last = {c.num: c.pos for c in frame.cars}
            continue
        now = {c.num: c.pos for c in frame.cars}
        for num, pos in now.items():
            was = last.get(num)
            if was is None or pos >= was:
                continue
            lost = [n for n, p in now.items() if last.get(n) == pos and p == was]
            if lost:
                out.append((i, frame.t, num, lost[0]))
        last = now
    return out


def oracle_ceiling(events: list[tuple], min_hold: float = 4.0) -> tuple[int, int]:
    """Most overtakes any single feed could have caught, knowing the future.

    Greedy over time: take an overtake, then skip anything inside the minimum
    hold that does not involve the same cars. This is an upper bound no real
    director can reach, since it requires foreknowledge.
    """
    caught = 0
    last_t = -1e9
    held: set[str] = set()
    for _, t, a, b in events:
        if t - last_t >= min_hold:
            caught += 1
            last_t = t
            held = {a, b}
        elif a in held or b in held:
            caught += 1
    return caught, len(events)


def run(race: str, policy: str, speed: float = 40.0, tick_hz: float = 10.0,
        laps: int = 0) -> Result:
    src = build_source(race)
    scorer = TensionScorer()
    guard = CutGuard(min_hold=4.0, max_hold=25.0)
    rng = random.Random(11)

    dt = 1.0 / tick_hz
    race_dt = dt * speed
    last_pos: dict[str, int] = {}
    overtakes = captured = cuts = 0
    on_air: str | None = None

    for frame in src.frames(race_dt):
        if laps and frame.lap > laps:
            break
        battles = scorer.score_frame(frame)

        # --- did a position change hands, and were we watching it? ---
        # Caution periods are excluded, matching Orchestrator._count_capture:
        # nobody may overtake under a safety car, so an order change there is a
        # pit stop, not a moment a camera missed.
        if not frame.racing:
            last_pos = {c.num: c.pos for c in frame.cars}
            continue
        now = {c.num: c.pos for c in frame.cars}
        for num, pos in now.items():
            was = last_pos.get(num)
            if was is None or pos >= was:
                continue
            lost = [n for n, p in now.items()
                    if last_pos.get(n) == pos and p == was]
            if not lost:
                continue
            overtakes += 1
            if on_air in (num, lost[0]):
                captured += 1
        last_pos = now

        # --- choose a shot ---
        if policy == "leader":
            # What a lazy broadcast does: stay on the front of the race.
            want = frame.cars[0].num
        elif policy == "random":
            want = rng.choice(frame.cars).num
        elif policy == "tension":
            cand = next((b for b in battles if b.score >= 0.55), None)
            live = next((b for b in battles if b.ahead_num == guard.current), None)
            guard.current_score = live.score if live else 0.0
            verdict = guard.evaluate(frame.t, cand, guard.current_score)
            want = cand.ahead_num if (verdict.allowed and cand) else guard.current
        else:
            raise SystemExit(f"unknown policy {policy}")

        if want and want != on_air:
            # Leader and random still respect a minimum hold; without it they
            # would flicker every tick and the comparison would be unfair.
            if policy == "tension" or (frame.t - guard.since) >= 4.0:
                on_air = want
                guard.commit(want, frame.t, 0.0)
                cuts += 1

    return Result(policy, overtakes, captured, cuts)


async def run_agent(race: str, speed: float = 40.0, tick_hz: float = 10.0,
                    cooldown_race_s: float = 30.0, laps: int = 0) -> Result:
    """The same replay, but Gemini picks which candidate goes on air.

    This is the one comparison that gives the model's judgment a ground truth.
    Everything else here measures the deterministic pipeline; this asks whether
    choosing the better *story* also catches more of the racing, or whether the
    two goals pull apart. Either answer is worth knowing — and an honest
    result matters more than a flattering one.
    """
    from backend.agents.director import DirectorAgent

    src = build_source(race)
    scorer = TensionScorer()
    guard = CutGuard(min_hold=4.0, max_hold=25.0)
    director = DirectorAgent()
    await director.warmup()

    dt = 1.0 / tick_hz
    last_pos: dict[str, int] = {}
    overtakes = captured = cuts = 0
    on_air: str | None = None
    last_call = 0.0
    model_calls = 0

    for frame in src.frames(dt * speed):
        if laps and frame.lap > laps:
            break
        battles = scorer.score_frame(frame)

        now = {c.num: c.pos for c in frame.cars}
        for num, pos in now.items():
            was = last_pos.get(num)
            if was is None or pos >= was:
                continue
            lost = [n for n, p in now.items() if last_pos.get(n) == pos and p == was]
            if not lost:
                continue
            overtakes += 1
            if on_air in (num, lost[0]):
                captured += 1
        last_pos = now

        cand = next((b for b in battles if b.score >= 0.55), None)
        live = next((b for b in battles if b.ahead_num == guard.current), None)
        guard.current_score = live.score if live else 0.0
        verdict = guard.evaluate(frame.t, cand, guard.current_score)
        if not (verdict.allowed and cand):
            continue
        if frame.t - last_call < cooldown_race_s:
            continue
        last_call = frame.t

        d = await director.decide(battles, frame.lap, frame.total_laps, guard.current)
        if d.tier in ("adk", "genai"):
            model_calls += 1
        chosen = next((b for b in battles if b.ahead_num == d.cut_to), cand)
        if chosen.ahead_num != on_air:
            on_air = chosen.ahead_num
            guard.commit(on_air, frame.t, chosen.score)
            cuts += 1
        if on_air:
            director.note_airtime(on_air, dt * speed)

    r = Result("agent", overtakes, captured, cuts)
    r.model_calls = model_calls  # type: ignore[attr-defined]
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", default="monza")
    ap.add_argument("--policies", nargs="+",
                    default=["tension", "leader", "random"])
    ap.add_argument("--laps", type=int, default=0,
                    help="limit to the first N laps (keeps agent runs affordable)")
    ap.add_argument("--agent", action="store_true",
                    help="also run the Gemini director (costs model calls)")
    args = ap.parse_args()

    events = collect_overtakes(args.race, laps=args.laps)
    ceil_n, total = oracle_ceiling(events)
    ceiling = ceil_n / total if total else 0.0

    window = f"first {args.laps} laps" if args.laps else "full race"
    print(f"\n  {args.race} — {window}, one pass per policy")
    print(f"  {total} position changes · one camera · 4s minimum hold\n")
    print(f"  {'policy':10s} {'captured':>9s} {'rate':>7s} {'of ceiling':>11s} {'cuts':>6s}")
    print("  " + "-" * 50)
    for pol in args.policies:
        r = run(args.race, pol, laps=args.laps)
        share = r.rate / ceiling if ceiling else 0.0
        print(f"  {r.policy:10s} {r.captured:>9d} {r.rate*100:>6.1f}% "
              f"{share*100:>10.1f}% {r.cuts:>6d}")
    if args.agent:
        r = asyncio.run(run_agent(args.race, laps=args.laps))
        share = r.rate / ceiling if ceiling else 0.0
        calls = getattr(r, "model_calls", 0)
        print(f"  {'agent':10s} {r.captured:>9d} {r.rate*100:>6.1f}% "
              f"{share*100:>10.1f}% {r.cuts:>6d}   ({calls} model calls)")
    print(f"  {'oracle':10s} {ceil_n:>9d} {ceiling*100:>6.1f}% {100.0:>10.1f}%       —")
    print("\n  oracle knows every pass in advance; no live director can reach it.\n")


if __name__ == "__main__":
    main()
