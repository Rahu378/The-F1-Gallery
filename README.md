# The F1 Gallery:

Agent-directed world feed for a Formula 1 broadcast.

**Live app** — https://the-gallery-742484896712.us-central1.run.app

**Dashboard** — https://bluehalibut1967.grafana.net/public-dashboards/d86453f15cce4974a052692a6374e6c4 (no login)

Built for the Agentic Cinema hackathon, Grafana track.

<img width="1098" height="701" alt="Screenshot 2026-09-20 at 12 00 20 AM" src="https://github.com/user-attachments/assets/05bc23a0-97f8-46cb-ae50-9f988f447fb7" />

https://youtu.be/n4R7Ut1NK1U?si=x9oJ-bqeCo-EYklm

## The problem

An F1 broadcast has a camera on every car and a hundred more around the
circuit, and sends one of them to the world feed. One person picks. When they
hold on the leader in clean air, a pass for P8 happens off screen and there is
no second take.

This scores every contested position continuously and cuts to a fight before
it resolves.

## How it works

```
FastF1 telemetry -> tension scorer -> director -> cut guard -> feed
                     deterministic   Gemini/ADK  deterministic
                          |               |            |
                          +---------------+------------+
                                          |
                            Grafana: annotations, alerts, metrics
```

| Stage | Kind | Does |
|---|---|---|
| Tension scorer | plain Python, 10 Hz | Scores all 19 adjacent pairings on gap, closing rate, DRS range, tyre delta |
| Director | Gemini 2.5 Flash via ADK | Picks which fight to show and writes the commentary line. Calls `recent_airtime` to spread coverage |
| Cut guard | plain Python | Minimum hold, staleness forcing, a margin before leaving a live story |

Scoring 19 pairings ten times a second is arithmetic and runs without a model.
The director is asked only what the arithmetic cannot answer — which of these
is worth showing. It sits behind the scorer and the guard, so it is called a
few times a minute rather than in the hot loop.

Every decision reports which tier produced it. `adk` and `genai` are model
decisions; `heuristic` and `timeout` are deterministic fallbacks. The UI
colours them differently and the logs label them, so a fallback is never shown
as though the model chose.

## Running locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env
docker compose up -d              # Grafana + Prometheus, optional
.venv/bin/uvicorn backend.main:app --port 8042
```

First start pulls the race from FastF1 and caches it (~30 s). Without network
it falls back to a deterministic synthetic race and says so in the header.

| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_GENAI_USE_VERTEXAI` | `0` | `1` for Vertex; then no API key is needed |
| `GOOGLE_CLOUD_PROJECT` | — | Required for Vertex |
| `GOOGLE_API_KEY` | — | AI Studio path only |
| `DIRECTOR_MODEL` | `gemini-2.5-flash` | Vertex and AI Studio expose different IDs |
| `DIRECTOR_COOLDOWN_S` | `13` | Wall seconds between director calls |
| `DIRECTOR_TIMEOUT_S` | `4` | Deadline before falling back |
| `GRAFANA_URL` / `GRAFANA_TOKEN` | — | Any Grafana, local or Cloud |
| `RACE_YEAR` / `RACE_EVENT` | `2023` / `Italian Grand Prix` | Use full event names — "Spa" fuzzy-matches to the *Spanish* Grand Prix |
| `REPLAY_SPEED` | `8.0` | Race seconds per wall second |

## Spoken commentary

Two voices, as on a real broadcast. The director writes both: a lead
play-by-play line and a shorter reaction from the colour commentator beside
them. Gemini's multi-speaker text-to-speech reads them with different voices —
Charon low and steady for the lead, Puck brighter for the reaction. One voice
reading both halves sounds like a press release.

The prompt forbids inventing anything. The director is given position, gap,
closing rate, DRS, tyre age and lap, and nothing else — so it may not say a
driver "has been quick all weekend", because it does not know that and a real
broadcast would be wrong to say it. It caught itself doing exactly that before
the constraint went in.

### Moments

A completed overtake is the one event in this system that is not a prediction,
so it is detected explicitly by watching for positions changing hands rather
than inferred from a score crossing a threshold. When a pass completes on a
battle the feed was actually watching, the call is re-read with the delivery
direction changed, and a crowd swell plays underneath.

The crowd is synthesised in the browser from filtered noise with a fast attack
and a long tail — a real crowd recording is someone's copyright and this repo
is public. It fires only on a pass the feed was on. Cheering every cut would
make the cheer mean nothing.

Off by default and generated on demand. TTS is another model call and quota is
the tightest constraint here, so nothing is synthesised until a client asks;
lines are cached, since the same sentence never needs paying for twice. Vertex
returns raw 16-bit PCM at 24 kHz, which browsers will not play, so it is
wrapped in a WAV header server-side rather than shipping a decoder.

## Passed over

Beside every cut, the panel shows the battles the director declined and what
they scored. The live contested list keeps moving, so this is frozen at the
moment of the decision — otherwise there is no way to see the choice that was
actually made.

This is deliberately not a multi-view. The premise is that one camera goes out
and something has to choose; showing four battles at once dissolves the problem
the project exists to solve. Showing the rejected options keeps the single feed
and makes the reasoning legible instead.

## Two audiences

**Broadcast** is what the feed puts out: the picture, the shot on air, and
both commentary voices. Everything else is gone.

**Control** is the operator surface — timing tower, contested positions,
passed-over, agent decisions, the metric strip. Dense on purpose, because it
is for someone running the feed rather than watching it.

Same agent, same decisions, two ways in. A studio head asking "what does this
actually put out?" should not have to read a timing tower to find out.

## Two views

The map has a **2D** and a **3D** toggle, bottom right. 2D is the default and
the operating view — top-down is objectively the better instrument for reading
gaps and order, which is why every timing screen uses it.

3D exists to show the one thing 2D cannot: elevation. The Z channel in
position telemetry gives Spa a 102 m climb through Eau Rouge and Raidillon,
and Monza 12 m across the entire lap. Both figures are measured, not styled.
Selecting a driver drops the camera to track level and chases them, so the
terrain is seen from the side rather than flattened.

Three.js r160 is vendored (MIT) and loaded on demand — it is 1.2 MB and most
viewers never open 3D. Glow is done with additive sprites scaled against
camera distance rather than a post-processing pass, which keeps the dependency
to three core.

## Circuits

Four 2023 races ship in the image and can be switched at runtime from the
replay controls in the footer: **Monza** (flat-out, two DRS zones), **Spa**
(longest lap on the calendar), **Silverstone** (fast and flowing) and
**Barcelona**. Corner markers come from circuit info; DRS zones are recovered
from the DRS channel in car telemetry, since the zones are not published
anywhere — Monza resolves to two, Spa to three, which is correct.

They are baked in rather than fetched on demand because Cloud Build egress
cannot reach the F1 timing API. Adding more means extending `RACES` in
[`scripts/prefetch.py`](scripts/prefetch.py) and `CATALOGUE` in
[`backend/data/source.py`](backend/data/source.py), then re-running prefetch
locally before deploying. Each race is roughly 110 MB of cache.

## Deploying

```bash
gcloud auth login && gcloud config set project YOUR_PROJECT
./scripts/deploy-cloudrun.sh
```

Three flags matter and none are optional:

- `--min-instances 1` — the race loop has to keep running between requests
- `--no-cpu-throttling` — Cloud Run parks CPU outside request handling, which
  stops the 10 Hz loop dead the moment a request finishes
- `--timeout 3600` — keeps the WebSocket open

The image carries the race (`scripts/prefetch.py`), so a cold container starts
in about 9 s and does not depend on the F1 timing API being reachable.

## Notes from building it

Things that cost time, kept here because none of them are obvious and all of
them fail quietly.

**Timing owns lap count; geometry owns position.** Deriving race order by
watching arc-length wrap around the centerline works while the field is
bunched and fails once it spreads. At Monza everyone stays in a slipstream
train, so it looked correct for weeks. At Spa the field is a lap wide by lap
three, one crossing gets missed, and the actual leaders rank behind
backmarkers — Albon "leading" a race Verstappen won. Mixing the two, official
lap plus geometric fraction, is worse again: the centerline origin and the
timing line are different points, so they disagree by a sliver every lap and
the error alternates sign. Interpolating between official lap start times
gives fractional race distance directly — monotonic by construction, exact at
every boundary. Geometry keeps the job it is good at, which is drawing the car
in the right place.

**The centerline has to be exactly one lap.** Race order and every gap come
from arc-length along a circuit centerline. Building it from a raw slice of
position data pulls in in-laps, out-laps and the pit lane, so the line doubles
back on itself and every downstream gap is wrong. Symptom was Albon leading
Monza by 141 seconds. Use the fastest lap's telemetry.

**Start the replay at lights-out, not session start.** Session telemetry
begins on the grid, where the whole field sits at the same track position.
Every gap reads 0.00 s and the scorer sees twenty simultaneous photo finishes
scoring 0.61.

**Measure before tuning.** The momentum term was normalised by 0.28 s/s.
Actual closing rates across a full replay: p50 0.004, p90 0.012, p99 0.018.
The term contributed nothing until the divisor became 0.015.

**A race is finite.** The replay ran out after ~8 minutes, the loop returned,
and the WebSocket went silent while HTTP stayed healthy. Anyone opening the
page after that saw a frozen screen. Found by connecting six clients and
watching all six receive the opening message and then hang.

**A session median lap time is the wrong conversion for a gap.** Gaps are a
difference in race distance, and turning that into seconds needs a
seconds-per-lap. The session median works until the field stops running at
session pace: behind a safety car a lap takes half again as long, so every gap
came out short by that factor and twenty cars reported two tenths apart. The
per-pair racing-speed gate does not catch it either — a safety car runs at
120–160 km/h, well above the 60 km/h floor that screens out the pits. So the
scorer saw nineteen simultaneous photo finishes at exactly the moment the world
feed should have been on the safety car. Pace is now measured over a rolling
window from the field's own progress, and the track-status channel suspends
scoring outright while the field is neutralised. The history buffer is dropped
with it, or the first green-flag sample is differenced against a caution gap
and every closing rate spikes on the restart lap.

**Gemini 2.5 thinks by default, and it dominates latency.** Measured on
Vertex: 3.1–7.5 s per call, against 0.7–0.9 s with `thinking_budget=0`. The
director picks from a ranked shortlist, so the thinking budget buys nothing.

**Model IDs differ between AI Studio and Vertex.** The `-latest` aliases do
not exist on Vertex at all.

**Free-tier AI Studio caps at 20 requests a day** on the newest flash model,
and function calling makes one director decision cost two or three requests.
That is about two minutes of running before everything silently falls back.

**Vertex 429s carry no numbers.** Just "resource exhausted, try again later" —
dynamic shared quota, regional capacity rather than a fixed ceiling. A fixed
30 s backoff threw away most of a race for a transient blip; now 6–10 s with
jitter.

**The first ADK call pays 27 s of warm-up.** Left alone that lands on whoever
opens the page first. A throwaway decision now fires at startup.

**Cloud Build cannot reach the F1 timing API.** Position data comes back empty
there, so regenerating the race inside the build produced an empty cache and
the container served the synthetic race while reporting healthy.

**`gcloud run deploy --source` reads `.gcloudignore`,** and with no such file
derives one from `.gitignore` — which excluded `cache/`. The 101 MB never left
the machine and `COPY` failed on a directory that exists locally.

**A fresh GCP project gives the compute service account no build roles.** The
first deploy dies on `storage.objects.get denied` against a bucket Cloud Run
had just created itself, which reads like a Cloud Run bug rather than IAM.

**A Grafana Cloud stack ships several `prometheus` datasources.** One of them,
`grafanacloud-usage`, holds billing telemetry. Binding to the first type match
gives a dashboard that queries the wrong store, renders empty, and logs
nothing.

## State

Working and verified against the live deployment:

- FastF1 loads the 2023 Italian Grand Prix, 20 drivers, 9,718 replay frames
- Race order matches history: Sainz leads from pole, Verstappen 0.097 s behind,
  Verstappen through around lap 15
- Director on Vertex via ADK, 1.6–2.9 s per decision, roughly 72% of cuts
  model-authored over a three-minute sample, the rest falling back cleanly
- Six simultaneous WebSocket viewers, 97 frames each over 10 s, no spread
- Deployed WebSocket verified over `wss`, 118 frames in 12 s
- Grafana Cloud: dashboard provisioned by the app, alert rule installed, cut
  annotations landing on the timeline, public sharing on

Sample of unedited director output:

```
Verstappen is right on the gearbox of leader Carlos Sainz!
Norris under immense pressure from Hamilton in the fight for eighth.
Incredible wheel-to-wheel action between the two McLaren teammates!
```

Nothing in the prompt says Piastri and Norris drive for the same team.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Deterministic and offline — no cache, no network, no model calls. They cover
the track-status projection, the rolling pace estimate, and what the scorer does
when the field is neutralised, which is the one situation a replay of a real
race cannot be relied on to reach on demand.

## Does it actually work?

The claim is that a pass which happens off screen is lost, so the measure that
matters is what fraction of the position changes in a race the feed was
watching when they happened. That is countable, so it is counted.

```bash
.venv/bin/python -m eval.capture_rate --race spa
```

Four full races, 735 green-flag position changes, one camera, a four-second
minimum hold:

| Circuit | Passes | This pipeline | Follow the leader | Random | Of ceiling |
|---|---|---|---|---|---|
| Monza | 160 | **41.9%** | 3.1% | 15.0% | 46.5% |
| Spa | 214 | **50.0%** | 1.4% | 14.5% | 55.4% |
| Silverstone | 83 | **56.6%** | 2.4% | 6.0% | 60.3% |
| Barcelona | 278 | **42.4%** | 0.0% | 12.2% | 48.6% |
| **Pooled** | **735** | **46.1%** | **1.4%** | **12.8%** | **51.5%** |

The denominator counts green-flag passes only. Positions change hands under a
safety car as well, but those are pit-stop shuffles rather than moments a camera
could have caught, and counting them adds misses that were never catchable.
Silverstone is the race this matters on: 15.8% of it ran under a safety car or
VSC, and 29 of its 112 order changes happened there. Including them reported
43.8%; excluding them reports 56.6%, the highest of the four.

The ceiling column matters. One camera cannot be in two places, so an oracle
that knows every pass in advance still only reaches about 90%. Against what is
physically catchable, this captures roughly half — with no knowledge of the
future. Following the leader, which is what an inattentive broadcast does,
catches almost nothing.

### What this does not measure

The model's *selection* quality. `--agent` runs the real director, but Vertex
serves Gemini from dynamic shared quota and an offline replay asks for
decisions far faster than a live race does, so most calls come back 429 and
fall through to the deterministic tier — a 15-lap run landed 3 real model
calls out of 14 cuts. The resulting number is mostly the fallback wearing the
agent's name, and it is not reported as evidence of anything. Doing this
properly needs provisioned throughput.

There is also a real tension worth naming: the director sometimes takes the
better *story* over the closest *gap*, which would trade capture rate for
narrative. Whether that trade is correct has no ground truth — "was that the
better story?" is not scoreable — so it is left as an open question rather
than dressed up as a result.

The policies compared are the deterministic ones, because they decide *where
the camera is*. The Gemini director chooses which of the candidate battles is
the better story; the candidate set comes from the tension scorer and the cut
guard. Measuring those in isolation keeps the number reproducible and free of
model calls.

The same figure is computed live and shown on the metric strip, so it can be
watched rather than taken on trust.

## On a phone

The layout collapses to a single column below 820px, and drops the map, the
metric strip and the passed-over panel rather than crushing them. On a phone
this is a viewer's surface, not an operator's: the shot on air, both
commentary voices, the field, and what the agent decided. A 3D circuit on a
390px screen is decoration and the metric rings are unreadable at that size,
so the canvas and Three.js are not booted at all there.

## Cost

Measured from billing rather than a price list: **$22.71** of Vertex spend
across a day running one decision every ten seconds — 8,640 decisions — is
**$0.0026 a decision**. A Grand Prix at real time is roughly 433 decisions, so
the agent directs **a full race for about $1.14** in model calls. Speech adds a
few cents, and only when a viewer turns audio on.

What that displaces is deliberately not quantified. A broadcast gallery is a
room of people, and this project has no data on what one costs — so no saving
is claimed, only what this side of the comparison actually costs to run. The
figure is live in the Monitors tab and recomputed from real decision counts.

The director only runs while somebody is connected. An empty gallery does not
need one: the replay keeps going on the deterministic tier, which costs
nothing, and Gemini is engaged the moment a viewer opens the page.

This was not a micro-optimisation. Ungated, the director called the model
every ten seconds around the clock — 8,640 calls a day into an empty room,
which was almost the entire bill. Measured at roughly $24/day of credit, most
of it Vertex.

Cloud Run still holds one warm instance so the race loop survives between
requests, which is a few dollars a month. Delete the service when you are done
with it.

## Scaling

One instance, concurrency 250. That is a deliberate ceiling, not an oversight.

All viewers share one race, so WebSocket fanout is cheap and concurrency is
raised accordingly. The instance count is what cannot go up: the race engine is
a per-process global — one orchestrator, one replay, one director. A second Cloud Run instance would start
its own independent race, so two viewers could land on different instances and
see different laps, different cuts, and a circuit switch that only took effect
for one of them. `--max-instances 1` removes that failure mode. Every viewer
sees the same race because there is only one race.

Beyond ~250 concurrent viewers this needs the engine split from the web tier —
a single producer publishing frames to Redis or Pub/Sub, with stateless web
instances subscribing. That is the correct architecture and it is not built
here; the honest position is a hard ceiling rather than silent divergence.

Model cost also scales with instances rather than viewers, since the director
runs per process. One instance serving 250 people costs the same as one
instance serving one.

## Known gaps

**Metrics in Grafana Cloud need three variables set.** Grafana Cloud cannot
scrape a process it has no route to, so the app pushes its own metrics over
Prometheus `remote_write` instead. That path is implemented and the deploy
script already wires it through, but it stays dark until `GRAFANA_PROM_URL`,
`GRAFANA_PROM_USER` and `GRAFANA_PROM_TOKEN` are filled in — URL and user from
your stack's Prometheus "Send Metrics" page, token from a Cloud Access Policy
scoped `metrics:write`. `GET /api/health` reports `metrics.remote_write`, the
push count and the last rejection, so empty panels can be told apart from
absent credentials. Annotations go over the Grafana API and work regardless.

**Gap accuracy.** Gaps are a difference in race distance converted with the
pace the field is currently running. Good to roughly a tenth under racing
conditions; not a substitute for official timing.

## Layout

```
backend/
  agents/     tension.py  director.py  cutguard.py  orchestrator.py
  data/       geometry.py  source.py
  grafana/    client.py  metrics.py  dashboard.json
  main.py     FastAPI + WebSocket
frontend/     broadcast UI
scripts/      deploy, prefetch, Grafana Cloud switchover
tests/        caution handling, metric push shape
```

## License

MIT — see [LICENSE](LICENSE).
