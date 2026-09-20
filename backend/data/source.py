"""Race sources.

`FastF1Source` replays a real Grand Prix from official timing + position
telemetry. `SyntheticSource` is a deterministic fallback so the pipeline runs
with no network and no FastF1 cache — the agent stack is identical either way.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..config import CACHE_DIR, settings
from .geometry import Centerline, synthetic_circuit, unwrap_progress

log = logging.getLogger("gallery.data")

# The F1 track-status feed, as FastF1 exposes it: single-character codes on a
# timestamped channel. 3 is undocumented and rare; treat it as a yellow rather
# than neutralising the field on a code we cannot identify.
TRACK_STATUS = {
    "1": "green",
    "2": "yellow",
    "3": "yellow",
    "4": "safety_car",
    "5": "red",
    "6": "vsc",
    "7": "vsc",       # "VSC ending" — still neutralised until the green flag
}

# Periods where overtaking is forbidden and the field is running to a delta.
NEUTRALISED = frozenset({"safety_car", "vsc", "red"})


def status_series(times: np.ndarray, codes: list[str], grid: np.ndarray,
                  default: str = "green") -> np.ndarray:
    """Project a timestamped track-status channel onto a uniform time grid.

    The feed is a step function — a status holds until the next message — so
    each grid sample takes the last status published at or before it. Samples
    before the first message fall back to `default`.
    """
    out = np.full(len(grid), default, dtype=object)
    if len(times) == 0:
        return out
    order = np.argsort(times)
    t_sorted = np.asarray(times, dtype=float)[order]
    c_sorted = [codes[i] for i in order]
    idx = np.searchsorted(t_sorted, grid, side="right") - 1
    for k, i in enumerate(idx):
        if i >= 0:
            out[k] = TRACK_STATUS.get(str(c_sorted[i]).strip(), default)
    return out


def field_pace(prog: np.ndarray, step: float, fallback: float,
               window_s: float = 20.0, lo_mult: float = 0.5,
               hi_mult: float = 4.0) -> np.ndarray:
    """Seconds per lap the field is actually running, per grid sample.

    Gaps are a difference in race distance, and turning that into seconds needs
    a seconds-per-lap. Using the session median for that is wrong the moment the
    field stops running at session pace: behind a safety car a lap takes half
    again as long, so a median-derived gap understates the real one by that
    factor and the whole field reads as nose to tail.

    The rate is taken over a window rather than between adjacent samples, since
    a 0.5 s difference of interpolated lap counts is mostly noise. Cars that are
    not moving — pits, grid, stopped on track — are dropped before the median so
    one retirement cannot drag the field's pace toward zero.
    """
    prog = np.atleast_2d(np.asarray(prog, dtype=float))
    n = prog.shape[1]
    if n < 2:
        return np.full(max(n, 1), fallback)

    w = max(1, int(round(window_s / step)))
    idx = np.arange(n)
    hi = np.minimum(n - 1, np.maximum(idx, w))
    lo = np.maximum(0, hi - w)
    span = np.maximum(step, (hi - lo) * step)

    rate = (prog[:, hi] - prog[:, lo]) / span          # laps per second, per car
    rate = np.where(rate > 1e-5, rate, np.nan)         # parked cars out

    with warnings.catch_warnings():
        # A stopped field is an all-NaN column by construction, not a surprise.
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(rate, axis=0)
    spl = np.where(np.isfinite(med) & (med > 0), 1.0 / np.where(med > 0, med, 1.0), np.nan)
    # A red flag drives the rate to zero and the reciprocal to infinity; a
    # single-sample glitch can do the opposite. Clamp to a band around the
    # green-flag baseline and fall back to it where there is no signal at all.
    spl = np.clip(spl, lo_mult * fallback, hi_mult * fallback)
    return np.where(np.isfinite(spl), spl, fallback)

# Fallback identity colours when a source has none (kept distinct, not team-accurate).
_PALETTE = [
    "#00E676", "#A24BFF", "#FFD93D", "#E8112D", "#3AA6FF",
    "#FF7A3D", "#00D5C8", "#FF5CA8", "#9AE66E", "#C9A227",
]


@dataclass
class Car:
    num: str
    code: str
    team: str
    color: str
    x: float = 0.0
    y: float = 0.0
    progress: float = 0.0
    pos: int = 0
    gap_ahead: float = 0.0      # seconds to the car in front
    closing: float = 0.0        # seconds/second — positive means catching
    tyre: str = "MEDIUM"
    tyre_age: int = 0
    speed: float = 0.0
    z: float = 0.0


@dataclass
class Frame:
    t: float
    lap: int
    cars: list[Car]
    total_laps: int = 0
    status: str = "green"   # green | yellow | safety_car | vsc | red
    pace: float = 0.0       # seconds per lap the field is currently running

    @property
    def racing(self) -> bool:
        """Is the field racing, or neutralised behind a safety car or a flag?

        Overtaking is forbidden under a safety car, a VSC and a red flag, so
        nothing in those periods is a battle no matter how small the gaps get.
        A local yellow only covers one sector and the rest of the lap is still
        green, so it stays racing.
        """
        return self.status not in NEUTRALISED


@dataclass
class RaceMeta:
    name: str = "Synthetic Circuit"
    year: int = 0
    session: str = "R"
    outline: list = field(default_factory=list)
    total_laps: int = 0
    source: str = "synthetic"
    corners: list = field(default_factory=list)   # [{n, x, y}] normalised
    drs_zones: list = field(default_factory=list) # [[start_frac, end_frac]]
    bounds: list = field(default_factory=list)    # [minx, miny, maxx, maxy]
    elevation_m: float = 0.0                     # real elevation change, metres


class SyntheticSource:
    """Deterministic 20-car race with organic convergence and overtakes."""

    def __init__(self, seed: int = 7, total_laps: int = 53):
        self.rng = np.random.default_rng(seed)
        self.cl = Centerline(synthetic_circuit())
        self.total_laps = total_laps
        self.meta = RaceMeta(
            name="Synthetic Circuit",
            outline=self.cl.outline(),
            total_laps=total_laps,
            source="synthetic",
            bounds=self.cl.bounds,
            elevation_m=round(self.cl.elev_range_m, 1),
        )
        nums = ["1", "11", "16", "55", "44", "63", "4", "81", "14", "18",
                "10", "31", "23", "22", "77", "24", "20", "27", "2", "3"]
        codes = ["VER", "PER", "LEC", "SAI", "HAM", "RUS", "NOR", "PIA", "ALO", "STR",
                 "GAS", "OCO", "ALB", "TSU", "BOT", "ZHO", "MAG", "HUL", "SAR", "RIC"]
        teams = ["Alpha", "Alpha", "Rossa", "Rossa", "Silver", "Silver", "Papaya", "Papaya",
                 "Verde", "Verde", "Bleu", "Bleu", "Navy", "Navy", "Cinza", "Cinza",
                 "Aco", "Aco", "Navy", "Bleu"]

        self.cars: list[Car] = []
        self.base_pace: list[float] = []
        lap_time = 82.0
        prog = 0.0
        for i in range(20):
            prog -= self.rng.uniform(0.55, 2.6) / lap_time  # realistic starting gaps
            c = Car(
                num=nums[i], code=codes[i], team=teams[i],
                color=_PALETTE[i % len(_PALETTE)],
                progress=prog,
                tyre=["SOFT", "MEDIUM", "HARD"][i % 3],
                tyre_age=int(self.rng.integers(4, 20)),
            )
            self.cars.append(c)
            # Slight pace spread — this is what makes battles form on their own.
            self.base_pace.append(lap_time + self.rng.normal(0.0, 0.45))
        self.phase = self.rng.uniform(0, 6.28, 20)
        self.t = 0.0

    def frames(self, dt) -> Iterator[Frame]:
        L = self.cl.length
        while True:
            step = dt() if callable(dt) else dt
            self.t += step
            for i, c in enumerate(self.cars):
                deg = 0.010 * max(0, c.tyre_age + self.t / 90.0 - 12)
                lap_time = self.base_pace[i] + deg + 0.35 * np.sin(self.t / 26.0 + self.phase[i])
                c.progress += step / lap_time
                c.speed = float(L / lap_time * 3.6 * (0.72 + 0.28 * np.sin(c.progress * 6.283 * 3)))
            yield self._assemble()

    def _assemble(self) -> Frame:
        order = sorted(self.cars, key=lambda c: -c.progress)
        lap = max(1, int(order[0].progress) + 1) if order[0].progress > 0 else 1
        for i, c in enumerate(order):
            c.pos = i + 1
            arc = (c.progress % 1.0) * self.cl.length
            sx, sy = self.cl.point_at(arc)
            c.x, c.y = sx, sy
            c.z = self.cl.elevation_at(arc)
            if i == 0:
                c.gap_ahead = 0.0
            else:
                ahead = order[i - 1]
                pace = self.base_pace[self.cars.index(c)]
                c.gap_ahead = max(0.0, (ahead.progress - c.progress) * pace)
        # The synthetic race has no safety cars, so it is green throughout and
        # its pace is the field's baseline. Stated rather than left to default,
        # so downstream code never has to special-case the source.
        return Frame(t=self.t, lap=min(lap, self.total_laps),
                     cars=list(order), total_laps=self.total_laps,
                     status="green", pace=float(np.median(self.base_pace)))


# FastF1 position coordinates are in tenths of a metre. Monza measures 57,347
# in raw units against a real 5,793 m lap — a ratio of 9.9. Gaps are unaffected
# because they come from progress times lap time, but any speed derived from
# the centerline is out by 10x without this.
UNITS_PER_METRE = 10.0


class FastF1Source:
    """Replays a real Grand Prix from official position + timing telemetry."""

    def __init__(self, year: int, event: str, session_id: str = "R", step: float = 0.5):
        import fastf1

        fastf1.Cache.enable_cache(str(CACHE_DIR))
        log.info("loading %s %s %s from FastF1…", year, event, session_id)
        ses = fastf1.get_session(year, event, session_id)
        ses.load(laps=True, telemetry=True, weather=False, messages=False)
        self.step = step

        pos = ses.pos_data
        results = ses.results
        drivers = [d for d in pos.keys() if d in set(results["DriverNumber"].astype(str))]
        if len(drivers) < 6:
            raise RuntimeError("insufficient position telemetry")

        # --- circuit centerline ---
        # This has to be exactly one clean lap. A raw slice of position data
        # spans in-laps, out-laps and the pit lane, which produces a centerline
        # that doubles back on itself — and since race order and every gap are
        # derived from arc-length along this line, a bad centerline corrupts
        # the entire pipeline. The fastest lap's telemetry is one clean circuit.
        self.cl = self._build_centerline(ses, pos, drivers)

        # --- common time grid ---
        starts, ends = [], []
        for d in drivers:
            df = pos[d]
            starts.append(df["SessionTime"].iloc[0].total_seconds())
            ends.append(df["SessionTime"].iloc[-1].total_seconds())
        t0, t1 = max(starts), min(ends)

        laps = ses.laps
        # Session telemetry begins well before the race does. Everything up to
        # lights-out is the grid and the formation lap — the whole field sits at
        # the same track position, every gap reads 0.00s and the tension scorer
        # sees twenty simultaneous photo finishes. Start at lights-out instead.
        try:
            race_start = float(laps["LapStartTime"].min().total_seconds())
            if t0 < race_start < t1:
                t0 = race_start
        except Exception:  # noqa: BLE001
            pass

        self.grid = np.arange(t0, t1, step)

        # Baseline pace, used as the fallback and as the band the live pace is
        # clamped into. Taken from green-flag laps only: a session median that
        # includes safety-car laps is pulled slow by them, which is the same
        # error this is meant to correct, just smaller.
        self.median_lap = self._green_median_lap(laps)

        self.cars: list[Car] = []
        self.prog: dict[str, np.ndarray] = {}
        self.arc: dict[str, np.ndarray] = {}
        for i, d in enumerate(drivers):
            df = pos[d]
            ts = df["SessionTime"].dt.total_seconds().to_numpy()
            gx = np.interp(self.grid, ts, df["X"].to_numpy())
            gy = np.interp(self.grid, ts, df["Y"].to_numpy())
            # Time-ordered path, so project with continuity rather than
            # nearest-point — see Centerline.project_path.
            s = self.cl.project_path(gx, gy)
            self.arc[d] = (s % self.cl.length) / self.cl.length   # map position
            self.prog[d] = self._progress(d, laps, s)             # race distance

            row = results[results["DriverNumber"].astype(str) == d]
            code = str(row["Abbreviation"].iloc[0]) if len(row) else d
            team = str(row["TeamName"].iloc[0]) if len(row) else "—"
            tc = str(row["TeamColor"].iloc[0]) if len(row) else ""
            color = f"#{tc}" if tc and not tc.startswith("#") else (tc or _PALETTE[i % len(_PALETTE)])
            self.cars.append(Car(num=d, code=code, team=team, color=color))

        # --- track status and live field pace ---
        # Both are grid-aligned lookups, so the hot loop reads an index rather
        # than recomputing anything per frame.
        self.status = self._status_grid(ses)
        self.pace = field_pace(
            np.vstack([self.prog[c.num] for c in self.cars]),
            self.step, fallback=self.median_lap,
        )

        # --- tyre state per driver per lap ---
        self.tyres: dict[str, list[tuple[float, str, int]]] = {}
        for d in drivers:
            dl = laps[laps["DriverNumber"].astype(str) == d]
            seq = []
            for _, r in dl.iterrows():
                st = r["LapStartTime"]
                if st != st:
                    continue
                comp = str(r.get("Compound", "") or "UNKNOWN")
                age = r.get("TyreLife", 0)
                seq.append((st.total_seconds(), comp, int(age) if age == age else 0))
            self.tyres[d] = seq

        total_laps = int(laps["LapNumber"].max()) if len(laps) else 0
        corners, drs = self._track_features(ses)
        self.meta = RaceMeta(
            name=str(ses.event["EventName"]),
            year=year,
            session=session_id,
            outline=self.cl.outline(),
            total_laps=total_laps,
            source="fastf1",
            corners=corners,
            drs_zones=drs,
            bounds=self.cl.bounds,
            elevation_m=round(self.cl.elev_range_m, 1),
        )
        self.i = 0
        log.info("loaded %s — %d drivers, %d frames", self.meta.name, len(self.cars), len(self.grid))

    def _progress(self, driver: str, laps, s: np.ndarray) -> np.ndarray:
        """Race distance in laps, from official timing alone.

        Geometry is the wrong instrument for this. Counting laps by watching
        arc-length wrap works while the field is bunched, as at Monza, and
        fails once it spreads: one missed crossing puts a car a whole lap out,
        and at Spa by lap three the actual leaders were being ranked behind
        backmarkers. Mixing the two — official lap plus geometric fraction —
        is worse again, because the centerline origin and the timing line are
        not the same point, so the two disagree by a sliver every lap and the
        error alternates sign.

        Timing knows exactly when each lap began. Interpolating between lap
        starts gives fractional race distance directly: monotonic by
        construction, exact at every boundary, and immune to anything the
        projection does. Geometry keeps the job it is good at — putting the
        car in the right place on the map.
        """
        dl = laps[laps["DriverNumber"].astype(str) == driver]
        starts, numbers = [], []
        for _, r in dl.iterrows():
            st = r["LapStartTime"]
            if st != st:
                continue
            starts.append(st.total_seconds())
            numbers.append(float(r["LapNumber"]) - 1.0)

        if len(starts) < 2:
            return unwrap_progress(s, self.cl.length)

        order = np.argsort(starts)
        starts_a = np.asarray(starts)[order]
        numbers_a = np.asarray(numbers)[order]

        # Extend the final lap so a car still running at the end of the grid
        # keeps advancing instead of flat-lining on its last recorded start.
        if len(starts_a) >= 2:
            last_len = starts_a[-1] - starts_a[-2]
            starts_a = np.append(starts_a, starts_a[-1] + max(1.0, last_len))
            numbers_a = np.append(numbers_a, numbers_a[-1] + 1.0)

        return np.interp(self.grid, starts_a, numbers_a)

    @staticmethod
    def _green_median_lap(laps) -> float:
        """Median lap time over green-flag laps only.

        `TrackStatus` on a lap is the concatenation of every status seen during
        it, so a clean lap is exactly "1" and anything else saw a flag.
        """
        try:
            secs = laps["LapTime"].dt.total_seconds()
            green = laps["TrackStatus"].astype(str).str.strip() == "1"
            med = secs[green].median()
            if med == med and med > 0:
                return float(med)
        except Exception:  # noqa: BLE001 — older FastF1 without TrackStatus
            pass
        med = laps["LapTime"].dt.total_seconds().median()
        return float(med) if med == med else 90.0

    def _status_grid(self, ses) -> np.ndarray:
        """Track status per grid sample, from the official status channel.

        Inferring a safety car from pace would work, but timing publishes it
        directly and a derived signal cannot be better than the one it is
        derived from. If the channel is missing the race is treated as green
        throughout, which is the old behaviour rather than a new failure.
        """
        try:
            ts = ses.track_status
            times = ts["Time"].dt.total_seconds().to_numpy()
            codes = [str(c) for c in ts["Status"].tolist()]
            grid = status_series(times, codes, self.grid)
            flagged = int(np.sum(grid != "green"))
            if flagged:
                log.info("track status: %d of %d frames neutralised or flagged",
                         flagged, len(grid))
            return grid
        except Exception as exc:  # noqa: BLE001
            log.warning("no track status channel (%s) — assuming green throughout", exc)
            return np.full(len(self.grid), "green", dtype=object)

    @staticmethod
    def _build_centerline(ses, pos, drivers) -> Centerline:
        """One clean lap of circuit geometry."""
        try:
            rot = 0.0
            try:
                rot = float(ses.get_circuit_info().rotation or 0.0)
            except Exception:  # noqa: BLE001
                pass
            fastest = ses.laps.pick_fastest()
            if fastest is not None:
                tel = fastest.get_pos_data()
                cols = ["X", "Y", "Z"] if "Z" in tel.columns else ["X", "Y"]
                xy = tel[cols].to_numpy()
                if len(xy) > 120:
                    log.info("centerline from fastest lap (%d samples, rotation %.0f)",
                             len(xy), rot)
                    return Centerline(xy, rotation=rot)
        except Exception as exc:  # noqa: BLE001
            log.warning("fastest-lap centerline failed (%s)", exc)

        # Fallback: one mid-race lap of a driver who ran the full distance.
        for d in sorted(drivers, key=lambda k: -len(pos[k])):
            try:
                dl = ses.laps[ses.laps["DriverNumber"].astype(str) == d]
                mid = dl.iloc[len(dl) // 2]
                mp = mid.get_pos_data()
                cols = ["X", "Y", "Z"] if "Z" in mp.columns else ["X", "Y"]
                xy = mp[cols].to_numpy()
                if len(xy) > 120:
                    log.info("centerline from car %s mid-race lap", d)
                    return Centerline(xy)
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError("could not build a circuit centerline")

    def _track_features(self, ses) -> tuple[list, list]:
        """Corner markers and DRS zones, both derived from session data.

        Corners come straight from the circuit info. DRS zones are not
        published anywhere, so they are recovered from car telemetry: the DRS
        channel reads 10, 12 or 14 while the flap is open, and mapping those
        samples onto the centerline shows where on the lap that happens.
        """
        corners: list = []
        try:
            ci = ses.get_circuit_info()
            s = self.cl.project(ci.corners["X"].to_numpy(), ci.corners["Y"].to_numpy())
            for arc, num in zip(s, ci.corners["Number"].to_numpy()):
                x, y = self.cl.point_at(float(arc))
                corners.append({"n": int(num), "x": round(x, 5), "y": round(y, 5)})
        except Exception as exc:  # noqa: BLE001
            log.debug("corner markers unavailable: %s", exc)

        drs: list = []
        try:
            import numpy as _np
            open_frac: list[float] = []
            picked = 0
            for _, lap in ses.laps.iterrows():
                if picked >= 6:
                    break
                try:
                    car = lap.get_car_data()
                    pos = lap.get_pos_data()
                except Exception:  # noqa: BLE001
                    continue
                if "DRS" not in car or len(pos) < 50:
                    continue
                picked += 1
                on = car[car["DRS"].isin([10, 12, 14])]
                if not len(on):
                    continue
                # align by session time, then project onto the centerline
                pt = pos["SessionTime"].dt.total_seconds().to_numpy()
                ot = on["SessionTime"].dt.total_seconds().to_numpy()
                gx = _np.interp(ot, pt, pos["X"].to_numpy())
                gy = _np.interp(ot, pt, pos["Y"].to_numpy())
                arcs = self.cl.project(gx, gy) / self.cl.length
                open_frac.extend(float(a) for a in arcs)

            if open_frac:
                # cluster the fractions into contiguous zones
                open_frac.sort()
                start = prev = open_frac[0]
                for f in open_frac[1:]:
                    if f - prev > 0.03:          # gap => new zone
                        if prev - start > 0.01:
                            drs.append([round(start, 4), round(prev, 4)])
                        start = f
                    prev = f
                if prev - start > 0.01:
                    drs.append([round(start, 4), round(prev, 4)])
        except Exception as exc:  # noqa: BLE001
            log.debug("DRS zones unavailable: %s", exc)

        log.info("track features: %d corners, %d DRS zones", len(corners), len(drs))
        return corners, drs

    def _tyre_at(self, d: str, t: float) -> tuple[str, int]:
        seq = self.tyres.get(d, [])
        cur = ("MEDIUM", 0)
        for st, comp, age in seq:
            if st <= t:
                cur = (comp, age)
            else:
                break
        return cur

    def reset(self) -> None:
        """Rewind to lights-out so the replay can loop."""
        self.i = 0

    def seek_lap(self, lap: int) -> bool:
        """Move the cursor to the first frame on the given lap."""
        target = max(1, int(lap))
        lead = max(self.prog.values(), key=lambda a: a[-1])
        idx = int(np.searchsorted(lead, target - 1))
        if 0 <= idx < len(self.grid):
            self.i = idx
            return True
        return False

    def frames(self, dt) -> Iterator[Frame]:
        while self.i < len(self.grid):
            stride = max(1, int(round((dt() if callable(dt) else dt) / self.step)))
            t = float(self.grid[self.i])
            pace = float(self.pace[self.i])
            status = str(self.status[self.i])
            back = max(0, self.i - stride)
            span = max(1e-6, (self.i - back) * self.step)
            for c in self.cars:
                p = float(self.prog[c.num][self.i])
                # Speed straight off the progress derivative — used to tell a
                # racing car from one parked in the pits or on the grid.
                c.speed = float(
                    (p - float(self.prog[c.num][back])) / span
                    * (self.cl.length / UNITS_PER_METRE) * 3.6
                )
                c.progress = p
                arc = float(self.arc[c.num][self.i]) * self.cl.length
                sx, sy = self.cl.point_at(arc)
                c.x, c.y = sx, sy
                c.z = self.cl.elevation_at(arc)
                c.tyre, c.tyre_age = self._tyre_at(c.num, t)
            order = sorted(self.cars, key=lambda c: -c.progress)
            for k, c in enumerate(order):
                c.pos = k + 1
                # Seconds per lap comes from what the field is running now, not
                # from the session median — see field_pace.
                c.gap_ahead = 0.0 if k == 0 else max(
                    0.0, (order[k - 1].progress - c.progress) * pace
                )
            lap = max(1, int(order[0].progress) + 1)
            yield Frame(t=t, lap=min(lap, self.meta.total_laps or lap),
                        cars=list(order), total_laps=self.meta.total_laps,
                        status=status, pace=pace)
            self.i += stride


# Races shipped in the image. Keep in step with scripts/prefetch.py — anything
# listed here that is not in the cache will try to reach the F1 API at runtime,
# which is not reachable from Cloud Run.
CATALOGUE = [
    {"id": "monza", "year": 2023, "event": "Italian Grand Prix",
     "label": "Monza", "note": "flat-out, two DRS zones"},
    {"id": "spa", "year": 2023, "event": "Belgian Grand Prix",
     "label": "Spa", "note": "longest lap on the calendar"},
    {"id": "silverstone", "year": 2023, "event": "British Grand Prix",
     "label": "Silverstone", "note": "fast and flowing"},
    {"id": "barcelona", "year": 2023, "event": "Spanish Grand Prix",
     "label": "Barcelona", "note": "conventional permanent circuit"},
]


def catalogue() -> list[dict]:
    return [dict(r) for r in CATALOGUE]


def build_source(race_id: str | None = None):
    """Real race if we can get it, deterministic synthetic if we can't."""
    year, event, session = settings.race_year, settings.race_event, settings.race_session
    if race_id:
        picked = next((r for r in CATALOGUE if r["id"] == race_id), None)
        if picked:
            year, event, session = picked["year"], picked["event"], "R"
    try:
        src = FastF1Source(year, event, session)
        log.info("source: FastF1 — %s", src.meta.name)
        return src
    except Exception as exc:  # noqa: BLE001 — any failure must degrade, not crash
        log.warning("FastF1 unavailable (%s); using synthetic source", exc)
        return SyntheticSource()
