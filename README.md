# CrowdFlow — drones that spot a crowd crush forming, and act before it does

Two camera drones watch a 100-person concert floor in simulation. They notice when the area in front of
the stage is becoming dangerously packed, and try to **prevent** a crush with escalating, non-physical
interventions: a red spotlight, a spoken announcement, and finally an alert to the security team.

![Same crowd without (left) and with (right) CrowdFlow](media/demo.gif)

*Same simulated crowd, same random seed. Left: drones only watch. Right: drones intervene. The chart
underneath is the true peak density in both runs.*

> **Simulation prototype for decision support.** Not a validated crowd-safety system. The drones never
> touch, push or herd anyone, and humans stay in charge.

## Results

Across **10 randomised crowds** (90 s each), the same scenario run with the drones only watching versus
intervening:

| | Drones only watch | Drones intervene |
|---|---|---|
| Time at **≥ 5 people/m²** (dangerous), mean | 62.9 s | **0.7 s** |
| Time at **≥ 4 people/m²** (movement restricted), mean | 66.0 s | **7.4 s** (−89%) |
| Crowds that ever reached 5 people/m² | 10 / 10 | **3 / 10** |
| Mean peak density | 6.11 people/m² | **4.77 people/m²** |

<p>
  <img src="results/density_timeline.png" width="49%" alt="Peak density over time for one crowd, with and without drones">
  <img src="results/per_seed.png" width="49%" alt="Seconds at high density for each of the 10 crowds">
</p>

**How it got there.** The first version only reacted once an area was already crowded. Adding a short-range
prediction (see below) moved the first alert earlier (21.3 s → 18.6 s on average) and, in that
version, cut the number of crowds that reached 5 people/m² from 4/10 to 1/10. Later changes to the
crowd model shifted the figures slightly (now 3/10); 10 crowds is too few to separate those differences
from chance. The evaluations are in `results/`.

### A learned detector that doesn't rely on colour

The simplest way to count people from above is to look for heads by colour, which works only while every
head looks the same. To remove that shortcut, the simulator can also produce a **varied crowd** (different
skin tones, hair, caps and clothing colours), and a **YOLO11n** detector was fine-tuned on about 1,500
drone-camera frames rendered from the simulator (40 epochs, 38 minutes on a MacBook Air GPU). Every label
comes automatically from ground truth, so nothing was labelled by hand. Tested on crowds the model never
saw in training (11,488 heads per crowd type):

| Detector | Uniform crowd | Varied crowd |
|---|---|---|
| Colour segmentation | 100% recall · 100% precision | **41% recall · 79% precision** |
| YOLO11n (learned) | 100% recall · 99.9% precision | **100% recall · 99.8% precision** |

The colour detector misses most people once appearances vary; the learned detector does not. It costs
more compute (21 ms per frame on a MacBook Air, versus 2 ms), still well within the 10 Hz loop.
Reproduce with `python detection/evaluate_detector.py` (results in `results/detector_comparison.json`).

**Why it matters for safety.** The same 10 crowds with varied appearances, run end to end:

| Varied crowd, 10 crowds | Drones only watch | Drones + colour detector | Drones + YOLO |
|---|---|---|---|
| Time at ≥ 5 people/m², mean | 62.9 s | 15.2 s | **0.6 s** |
| Time at ≥ 4 people/m², mean | 66.0 s | 30.8 s | **7.9 s** |
| Crowds that reached 5 people/m² | 10 / 10 | 10 / 10 | **3 / 10** |
| First alert, mean | – | 29.3 s | **18.6 s** |

With the colour detector the drones see too few people, raise the alarm 11 s later and fail to stop
any crowd from reaching dangerous density. With YOLO they perform as well as on the uniform crowd above (0.6 s vs 0.7 s).
Results in `results/varied_colour.json` and `results/varied_yolo.json`.

**How to read these numbers.** Densities are measured *after* each run from simulator ground truth (most
people inside any 1 m-radius circle); the drones never see ground truth. Crowd-safety guidance commonly
treats sustained densities above ~5 people/m² as dangerous. The crowd model is stylised, so this shows the
system works **in this simulation**, not in a real venue. Exact numbers vary slightly between machines.

## The scenario

* **Venue:** a 20 m × 15 m indoor floor, stage on the west wall behind a barrier, bar on the east wall.
  100 people average ~0.37 people/m², but the strip in front of the stage centre is only a few m².
* **People:** 60 fans, 28 casual attendees and 12 people moving to and from the bar. Each fan decides
  *at a random time* to head for the centre of the stage barrier, and keeps pressing forward once there.
* **Nothing is scripted.** Congestion emerges because many people want the same few square metres.

## How it works

Every 0.1 s, the loop runs perception → decision → action, and the cameras then see the result:

```mermaid
flowchart LR
    A[Drone cameras<br/>2 × 640×480] --> B[Head detection<br/>+ ground projection]
    B --> C[Tracking<br/>IDs, speed, heading]
    C --> D[Fleet crowd map<br/>density, score C, prediction]
    D --> E[Coordinator<br/>patrol · respond · escalate]
    E --> F[Act<br/>move · spotlight · announce · alert]
    F --> G[People react<br/>or don't]
    G --> A
```

1. **Perception:** each drone's downward camera looks for heads. In a dense crowd bodies merge into one
   blob but heads stay separate, so counting stays reliable. Two interchangeable detectors:
   *colour segmentation* (`perception/density_detector.py`, fast, but needs uniform heads) and a learned
   *YOLO11n* model (`perception/yolo_detector.py`, works on varied crowds). Detected heads are projected
   onto the floor using the drone's pose.
2. **Tracking** (`perception/tracker.py`): detections are linked across frames (Hungarian matching,
   0.4 m gate), giving each person a temporary ID, a speed and a heading.
3. **Fleet crowd map** (`perception/zone_analytics.py`): both drones' tracks are fused into overlapping
   2 m × 2 m zones. Each zone gets density, speed, growth, share stopped and flow direction, combined into
   a **congestion score C** (0–100: NORMAL / DEVELOPING / HIGH / CRITICAL). Areas no drone can currently
   see are *unknown*, never "empty".
4. **Early warning:** each zone also predicts its density 10/20/30 s ahead from two signals, and takes the
   higher: the trend over the last 30 s, and the people currently *walking towards* it. The second catches
   a wave of fans heading for the stage before the density itself rises.
5. **Coordinator** (`brain/fleet_planner.py`): Alpha patrols the stage half and Bravo the bar half at 7 m.
   When a zone is developing, or predicted to become dense, the nearer drone descends to 5.5 m over it and
   the other repositions to keep the stage front in view. Then it escalates step by step:

| Step | When | What happens |
|---|---|---|
| 0 Monitor | an area is developing, or heading for ≥ 3.5 people/m² in 20 s | a drone moves in to look closer |
| 1 Spotlight | the area is HIGH, filling up, or heading for ≥ 3.5 | a red spotlight marks the area |
| 2 Announce | still HIGH, CRITICAL, or heading for ≥ 4 | "please make space at the front", naming side areas the drones **can see** are free; repeated if nothing improves |
| 3 Security | only if density is **not falling** well after the announcement | notify the security team (a notification only) |
| Resolved | below HIGH for 8 s | spotlight off, back to patrol |

6. **People's response** (`sim/drone_env.py`): the spotlight and announcement only change what people
   know. Each person has their own responsiveness (35–90%) and reaction delay (1–8 s). People in the
   spotlight may stop pushing forward; people who follow an announcement walk to the nearest named side
   area and stop early if it is already busy.

**A lesson built into the design:** an early version sent everyone to one side area, which simply became the
next hotspot. The coordinator now spreads people over several observed, uncrowded areas.

<details>
<summary>Score details and adaptations for a standing audience</summary>

`C = 100 × (0.25·density + 0.20·speed drop + 0.20·growth + 0.15·share stopped + 0.10·direction mix + 0.10·duration)`,
with unavailable terms dropped and the weights renormalised.

The score was designed for walking flows. At a concert, standing still at 1–2 people/m² is normal, so:
speed, stopping and direction terms are scaled by `clamp((ρ − 2.0) / 1.5, 0, 1)`; growth by
`clamp(ρ − 1.0, 0, 1)`; and the critical override uses ρ ≥ 3.0. All thresholds and weights are prototype
choices, not validated safety limits.
</details>

## Run it

Requires Python 3.9+. On macOS this works out of the box; on a headless Linux machine set `MUJOCO_GL=egl`.

```bash
pip install -r requirements.txt

python run_simulation.py                       # one 90 s run -> outputs/submission_video.mp4 (1920×1080)
python run_simulation.py --no-interventions    # baseline: drones only watch
python evaluate.py --seeds 10 --workers 4      # many crowds, on vs off -> results/evaluation.json
python plot_results.py                         # plots from the evaluation -> results/*.png
python side_by_side.py --clip 15 50 --gif      # comparison video + short clip and GIF -> outputs/
```

Videos, run summaries and training runs go to `outputs/`, which git ignores; delete it any time, the scripts recreate it. Evaluation tables and plots that the README uses go to `results/`. To update the README's animation: `cp outputs/side_by_side_clip.gif media/demo.gif`.

**Learned detector (optional).** Needs PyTorch via Ultralytics. A trained model is included in `models/`;
retrain it after changing the simulator:

```bash
pip install -r requirements-yolo.txt
python detection/make_dataset.py               # ~1,500 labelled frames -> datasets/heads/
python detection/train.py                      # fine-tunes YOLO11n (uses the Apple GPU / CUDA if present)
python detection/evaluate_detector.py          # colour vs YOLO, uniform vs varied crowd
python run_simulation.py --detector yolo --appearance varied
python evaluate.py --detector yolo --appearance varied --label varied_yolo
```

Useful options for `run_simulation.py`: `--seed`, `--duration`, `--fps` (multiple of 10), `--no-video`
(numbers only, faster, identical results), `--no-cards` (skip the title and results cards), `--quiet`,
`--camera-fault` (corrupts Alpha's camera at 70–74 s to show the failsafe: spotlight off, climb to 8 m
and hold), `--detector colour|yolo`, `--appearance standard|varied`. `evaluate.py` and `side_by_side.py`
take the same `--detector` and `--appearance` options.

The single-run video (1920×1080) opens with a title card and ends with that run's results. In between it
shows the venue overview with the drones labelled by role, the active drone's camera (tracked people,
with those stopped for 3 s+ highlighted), the fleet crowd map, the escalation steps, the hotspot's
measurements and the operator log. All drawing lives in `viz/hud.py`, so both videos share one style.

## Repository

```text
Scripts you run
  run_simulation.py          one run: video with operator panel + per-run summary
  evaluate.py                many crowds: drones intervening vs only watching
  plot_results.py            timeline and per-crowd plots from evaluate.py results
  side_by_side.py            comparison video (same crowd, without vs with drones) + clip/GIF
  detection/                 YOLO: dataset generation, training, detector comparison

The system
  config/generate_arena.py   venue, 100 pedestrians, 2 drones -> config/arena.xml (generated)
  sim/drone_env.py           physics, crowd behaviour, response to the drones, ground truth
  perception/                head detectors (colour, YOLO), tracker, crowd map, score C, prediction
  brain/fleet_planner.py     patrol, responder/overwatch roles, escalation ladder, failsafe
  viz/hud.py                 shared video style: fonts, colours, layout, title and results cards
  models/head_detector.pt    trained YOLO11n head detector (5.5 MB)

Results and media (in git)
  results/                   evaluation tables and plots used in this README
  media/demo.gif             the animation at the top of this README

Generated, not in git
  outputs/                   videos, run summaries, side-by-side source runs, YOLO training runs
  datasets/                  YOLO training images and labels

Original hackathon version
  legacy/                    run_all.py (earlier 18-person demo) and its pitch notes
```

## Limitations

* **Simulation only:** simple cylinder pedestrians, stylised pushing and compliance, one indoor venue,
  one scenario.
* **Synthetic-only detector:** the YOLO model is trained and tested on simulated frames (simple shapes,
  even lighting). It shows the pipeline needed for real footage; it would need real drone images to work
  on a real crowd.
* **Over-cautious escalation:** security is still alerted in 4 of 10 crowds. In a real venue an alert that
  almost always fires would soon be ignored.
* **Not modelled:** battery life, whether the drone speaker is audible over concert audio, wind, lighting,
  and the safety of flying over people.
* Latency figures printed by the scripts are from the machine they run on, not from drone hardware.

## Credits

CrowdFlow started as a team project at a hackathon in November 2026. This version of the simulation,
perception, decision logic and evaluation was rebuilt afterwards, keeping the team's core idea of
venue-wide congestion monitoring by drones.
