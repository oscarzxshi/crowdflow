#!/usr/bin/env python3
"""
CrowdFlow - stage-front scenario, 100 people, 2 drones (MuJoCo).

Closed loop, every 0.1 s:
    drone cameras -> head detection -> tracking -> (1 Hz) fleet crowd map + score C
    -> fleet coordinator (patrol / respond / escalation ladder) -> drone moves, spotlight,
    announcements -> people react (or not) -> cameras see the result.

Nothing is timed: congestion emerges as fans decide to move to the stage.
The drones only ever use camera-derived data. Ground truth is used once per second
for the evaluation summary at the end, never by the drones.

Usage:
    python run_simulation.py                    # 90 s run, 30 FPS video
    python run_simulation.py --fps 10           # faster rendering
    python run_simulation.py --no-interventions # baseline: drones watch but never act
    python run_simulation.py --camera-fault     # inject a camera fault at 70-74 s
    python run_simulation.py --no-video         # fast: numbers only, no rendering of the overview
    python run_simulation.py --no-cards         # video without the title and results cards
"""

import argparse
import json
import os
import time
import numpy as np
import cv2

from sim.drone_env import CrowdFleetEnv, SPOTLIGHT_RADIUS, VENUE_X, VENUE_Y, BARRIER_X
from perception.density_detector import DensityPerceptionEngine
from perception.tracker import MultiTracker
from perception.zone_analytics import FleetCrowdMap
from brain.fleet_planner import FleetCoordinator
from viz import hud

FRAME_W, FRAME_H = 640, 480          # drone cameras
CONTROL_HZ = 10

# Video layout, 1920x1080:  header | overview (1280x720) + drone camera (640x480)
#                                    bottom panel (1280x300) + fleet crowd map (640x540)
VID_W, VID_H = 1920, 1080
HEADER_H = 60
OV_W, OV_H = 1280, 720
BOTTOM_H = VID_H - HEADER_H - OV_H
MAP_H = VID_H - HEADER_H - FRAME_H
STEP_TEXT = ["Monitor", "Spotlight", "Announcement", "Security"]
STEP_HINT = ["a drone moves in to inspect", "the area is marked with a red light",
             "people are asked to move to free areas", "the security team is notified"]


# ---------------------------------------------------------------------- drawing
def draw_header(t, duration, enabled, coordinator, detail=""):
    cv = hud.Canvas(VID_W, HEADER_H, hud.BG)
    cv.rect(0, HEADER_H - 1, VID_W, HEADER_H, hud.BORDER)
    cv.text((28, HEADER_H / 2), "CrowdFlow", 22, hud.TEXT, "bold", anchor="lm")
    cv.text((28 + hud.text_width("CrowdFlow", 22, "bold") + 20, HEADER_H / 2),
            "Stage-front scenario  ·  100 people  ·  2 drones  ·  simulation" + detail, 15, hud.TEXT_2, anchor="lm")
    x = VID_W - 28
    x -= hud.text_width(f"{t:5.1f} s / {duration:.0f} s", 16, mono=True)
    cv.text((x, HEADER_H / 2), f"{t:5.1f} s / {duration:.0f} s", 16, hud.TEXT, mono=True, anchor="lm")
    label = "Drones intervene" if enabled else "Drones only watch"
    w = hud.text_width(label, 13, "bold") + 16
    cv.chip(x - 24 - w, HEADER_H / 2 - 11.5, label, hud.ACCENT if enabled else hud.TEXT_3)
    if coordinator.security_alert:
        w2 = hud.text_width("Security team notified", 13, "bold") + 16
        cv.chip(x - 24 - w - 12 - w2, HEADER_H / 2 - 11.5, "Security team notified", hud.STATUS[3],
                text_colour=hud.TEXT)
    return cv.finish()


def draw_overview(rgb, env, coordinator, t, last_announcement):
    cv = hud.Canvas(OV_W, OV_H, img=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    for d in ("alpha", "bravo"):                        # name tags above the drones
        p = hud.project_to_image(env.model, env.data, "world_cam", coordinator.pos[d] + [0, 0, 0.55],
                                 OV_W, OV_H)
        if p is not None and 0 < p[0] < OV_W and 30 < p[1] < OV_H:
            label = f"{d.capitalize()}  ·  {coordinator.state[d].lower()}"
            w = hud.text_width(label, 13, "bold") + 16
            cv.chip(p[0] - w / 2, p[1] - 30, label, hud.PANEL, text_colour=hud.TEXT)
    if last_announcement is not None and t - last_announcement[0] < 6.0:
        msg = f"“{last_announcement[1]}”"
        x1 = 24 + max(hud.text_width(msg, 18), 260) + 40
        cv.shade(24, 24, x1, 92, hud.BG, 0.88)
        cv.rect(24, 24, 28, 92, hud.ACCENT)
        cv.text((46, 44), f"ANNOUNCEMENT FROM {last_announcement[2].upper()}", 12, hud.ACCENT, "bold",
                anchor="lm")
        cv.text((46, 70), msg, 18, hud.TEXT, anchor="lm")
    return cv.finish()


def draw_drone_cam(perc, tracks, drone, pose, state, detector, crowd_map, t, beacon):
    cv = hud.Canvas(FRAME_W, FRAME_H, img=perc["bgr"].copy())
    stopped_col = hud.STATUS[1]
    for tr in tracks:
        u, v = detector.world_to_pixel(tr.pos, pose)
        cv.circle((u, v), 8, stopped_col if tr.slow_updates >= 3 else hud.TEXT, 1)
    worst = crowd_map.worst_zone(t, fresh_only=False)
    if worst is not None and worst.status >= 1:
        x0, x1, y0, y1 = worst.rect
        p0 = detector.world_to_pixel((x0, y1), pose)
        p1 = detector.world_to_pixel((x1, y0), pose)
        cv.rect(p0[0], p0[1], p1[0], p1[1], hud.STATUS[worst.status], filled=False, thickness=2)
        cv.chip(p0[0], max(52, p0[1] - 26), f"{hud.STATUS_LABEL[worst.status]}  ·  C {worst.score:.0f}",
                hud.STATUS[worst.status])
    cv.shade(0, 0, FRAME_W, 44, hud.BG, 0.8)
    cv.text((16, 22), f"{drone.upper()} CAMERA", 13, hud.TEXT, "bold", anchor="lm")
    cv.text((16 + hud.text_width(f"{drone.upper()} CAMERA", 13, "bold") + 14, 22), f"{pose[2]:.1f} m  ·  {state.lower()}" + ("  ·  spotlight on" if beacon else ""),
            13, hud.TEXT_2, anchor="lm")
    cv.text((FRAME_W - 16, 22), f"{len(tracks)} people tracked", 13, hud.TEXT_2, anchor="rm")
    cv.shade(0, FRAME_H - 34, FRAME_W, FRAME_H, hud.BG, 0.8)
    cv.circle((22, FRAME_H - 17), 6, hud.TEXT, 1)
    cv.text((34, FRAME_H - 17), "moving", 12, hud.TEXT_2, anchor="lm")
    cv.circle((104, FRAME_H - 17), 6, stopped_col, 1)
    cv.text((116, FRAME_H - 17), "stopped 3 s+", 12, hud.TEXT_2, anchor="lm")
    cv.rect(0, 0, 1, FRAME_H, hud.BORDER)
    return cv.finish()


def draw_map(crowd_map, coordinator, t):
    cv = hud.Canvas(FRAME_W, MAP_H, hud.PANEL)
    cv.rect(0, 0, 1, MAP_H, hud.BORDER)
    cv.rect(0, 0, FRAME_W, 1, hud.BORDER)
    hud.section_title(cv, 24, 20, "Fleet crowd map",
                      (FRAME_W - 24, f"camera-derived  ·  {crowd_map.coverage(t) * 100:.0f}% of floor seen"))
    s = (FRAME_W - 48) / (2 * VENUE_X)
    ox, oy = 24, 52
    to_px = lambda x, y: (ox + (x + VENUE_X) * s, oy + (VENUE_Y - y) * s)
    cv.rect(*to_px(-VENUE_X, VENUE_Y), *to_px(VENUE_X, -VENUE_Y), hud.BG)
    for z in crowd_map.zones:
        fresh = crowd_map.freshness(z, t)
        col = hud.STATUS[z.status] if fresh != "unknown" else hud.STATUS[None]
        weight = 0.22 if (fresh != "unknown" and z.status == 0) else 1.0
        if fresh == "stale":
            weight *= 0.5
        col = tuple(int(c * weight + b * (1 - weight)) for c, b in zip(col, hud.BG))
        p0, p1 = to_px(z.c[0] - 0.5, z.c[1] + 0.5), to_px(z.c[0] + 0.5, z.c[1] - 0.5)
        cv.rect(p0[0] + 1, p0[1] + 1, p1[0] - 1, p1[1] - 1, col)
    sx0, sy0 = to_px(-VENUE_X, 5.5)
    sx1, sy1 = to_px(BARRIER_X, -5.5)
    cv.rect(sx0, sy0, sx1, sy1, hud.BORDER)
    cv.text(((sx0 + sx1) / 2, (sy0 + sy1) / 2), "STAGE", 11, hud.TEXT_3, "bold", anchor="mm")
    for d in ("alpha", "bravo"):
        p = to_px(*coordinator.pos[d][:2])
        if coordinator.beacon[d]:
            cv.circle(p, SPOTLIGHT_RADIUS * s, hud.ACCENT, 2)
        cv.circle(p, 9, hud.BG)
        cv.circle(p, 7, hud.TEXT)
        cv.text(p, d[0].upper(), 11, hud.BG, "bold", anchor="mm")
    ly = oy + 2 * VENUE_Y * s + 22
    x = 24
    for k in (0, 1, 2, 3, None):
        cv.rect(x, ly - 6, x + 12, ly + 6, hud.STATUS[k])
        cv.text((x + 18, ly), hud.STATUS_LABEL[k], 12, hud.TEXT_2, anchor="lm")
        x += 30 + hud.text_width(hud.STATUS_LABEL[k], 12)
    cv.circle((x + 8, ly), 6, hud.ACCENT, 2)
    cv.text((x + 20, ly), "spotlight", 12, hud.TEXT_2, anchor="lm")
    return cv.finish()


def draw_bottom(crowd_map, coordinator, t):
    cv = hud.Canvas(OV_W, BOTTOM_H, hud.PANEL)
    cv.rect(0, 0, OV_W, 1, hud.BORDER)
    inc = coordinator.incident
    level = inc["level"] if inc else -1

    # --- escalation stepper ---
    x0 = 28
    hud.section_title(cv, x0, 22, "Escalation")
    if inc is None:
        cv.chip(x0 + 100, 17, "Patrolling", hud.BORDER, text_colour=hud.TEXT_2, size=12)
    for k in range(4):
        y = 70 + k * 56
        done = k <= level
        col = (hud.STATUS[3] if k == 3 else hud.ACCENT) if done else hud.BORDER
        if k < 3:
            cv.line((x0 + 10, y + 12), (x0 + 10, y + 44), hud.ACCENT if k < level else hud.BORDER, 2)
        cv.circle((x0 + 10, y), 9, col, -1 if done else 2)
        cv.text((x0 + 32, y - 9), STEP_TEXT[k], 16, hud.TEXT if done else hud.TEXT_3,
                "bold" if k == level else "normal")
        cv.text((x0 + 32, y + 13), STEP_HINT[k], 12, hud.TEXT_2 if done else hud.TEXT_3)

    # --- hotspot metrics ---
    x1 = 440
    cv.rect(x1 - 24, 20, x1 - 23, BOTTOM_H - 20, hud.BORDER)
    z = inc.get("zone") if inc else crowd_map.worst_zone(t)
    hud.section_title(cv, x1, 22, "Hotspot" if inc else "Busiest area")
    if z is not None and z.status is not None:
        fmt = lambda v, f: "–" if v is None else f.format(v)
        cv.chip(x1, 46, hud.STATUS_LABEL[z.status], hud.STATUS[z.status])
        cv.text((x1 + hud.text_width(hud.STATUS_LABEL[z.status], 13, "bold") + 30, 57),
                f"score C {z.score:.0f} / 100", 14, hud.TEXT_2, anchor="lm")
        rows = [("Density", f"{z.rho:.2f} people/m²"),
                ("Growth", fmt(z.growth, "{:+.2f} people/m² per min")),
                ("Median speed", fmt(z.speed, "{:.2f} m/s")),
                ("Stopped", fmt(z.p_stop, "{:.0%}")),
                ("Expected in 20 s", "–" if z.pred is None else f"{z.pred[20]:.1f} people/m²"),
                ("Data", crowd_map.freshness(z, t))]
        for k, (label, value) in enumerate(rows):
            y = 96 + k * 30
            cv.text((x1, y), label, 14, hud.TEXT_2, anchor="lm")
            cv.text((x1 + 150, y), value, 14, hud.TEXT, anchor="lm")
    else:
        cv.text((x1, 60), "No observations yet", 14, hud.TEXT_3, anchor="lm")

    # --- operator log ---
    x2 = 836
    cv.rect(x2 - 24, 20, x2 - 23, BOTTOM_H - 20, hud.BORDER)
    hud.section_title(cv, x2, 22, "Operator log")
    y = 52
    for et, text in coordinator.events[-6:]:
        text = text.replace(" p/m2", " people/m²").replace("->", "→").replace(" - ", " · ")
        lines = hud.wrap(text, 13, OV_W - x2 - 90, 2)
        cv.text((x2, y), f"{et:5.1f}s", 13, hud.TEXT_3, mono=True)
        for j, line in enumerate(lines):
            cv.text((x2 + 62, y + 18 * j), line, 13, hud.TEXT)
        y += 18 * len(lines) + 8
        if y > BOTTOM_H - 30:
            break
    return cv.finish()


# ---------------------------------------------------------------------- main loop
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--fps", type=int, default=30, help="video FPS (multiple of 10)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-interventions", action="store_true", help="baseline: observe only")
    ap.add_argument("--camera-fault", action="store_true", help="inject Alpha camera fault 70-74 s")
    ap.add_argument("--out", default=os.path.join("outputs", "submission_video.mp4"))
    ap.add_argument("--summary", default=os.path.join("outputs", "run_summary.json"))
    ap.add_argument("--no-video", action="store_true",
                    help="skip the overview camera and video file (much faster; drones still use their cameras)")
    ap.add_argument("--quiet", action="store_true", help="only print the final evaluation")
    ap.add_argument("--no-cards", action="store_true", help="no title and results cards in the video")
    ap.add_argument("--detector", choices=("colour", "yolo"), default="colour",
                    help="colour: head-colour segmentation; yolo: learned detector (models/head_detector.pt)")
    ap.add_argument("--appearance", choices=("standard", "varied"), default="standard",
                    help="varied: different skin tones, hair, caps and clothing (breaks the colour detector)")
    ap.add_argument("--model", default=None, help="YOLO weights (default models/head_detector.pt)")
    return ap


def run(args):
    """Runs one simulation and returns the evaluation summary (also written to args.summary)."""
    assert args.fps % CONTROL_HZ == 0, "--fps must be a multiple of 10"
    say = (lambda *a, **k: None) if args.quiet else print

    base = os.path.dirname(os.path.abspath(__file__))
    env = CrowdFleetEnv(frame_width=FRAME_W, frame_height=FRAME_H, seed=args.seed,
                        appearance=getattr(args, "appearance", "standard"))
    if getattr(args, "detector", "colour") == "yolo":
        from perception.yolo_detector import YoloPerceptionEngine, DEFAULT_MODEL
        detector = YoloPerceptionEngine(FRAME_W, FRAME_H, model_path=args.model or DEFAULT_MODEL)
    else:
        detector = DensityPerceptionEngine(FRAME_W, FRAME_H)
    trackers = {"alpha": MultiTracker(), "bravo": MultiTracker()}
    crowd_map = FleetCrowdMap()
    coordinator = FleetCoordinator(interventions_enabled=not args.no_interventions)

    video_path = os.path.join(base, args.out)
    for path in (video_path, os.path.join(base, args.summary)):        # e.g. outputs/
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = None if args.no_video else cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (VID_W, VID_H))
    cards = writer is not None and not getattr(args, "no_cards", False)
    if cards:
        card = hud.title_card(VID_W, VID_H, "CrowdFlow",
                              "Drones that spot a crowd crush forming, and act before it does",
                              ["Simulation  ·  100 people  ·  2 camera drones  ·  stage-front scenario",
                               "Drones intervene" if not args.no_interventions else "Baseline: drones only watch"])
        for _ in range(int(2.5 * args.fps)):
            writer.write(card)
    frames_per_tick = args.fps // CONTROL_HZ
    steps_per_tick = int(round(1.0 / CONTROL_HZ / env.sim_dt))
    dt = 1.0 / CONTROL_HZ
    n_ticks = int(args.duration * CONTROL_HZ)

    lat = {"perception": [], "tracking": [], "analytics": [], "brain": []}
    timeline = []
    last_announcement = None
    wall0 = time.perf_counter()
    say(f"Running {args.duration:.0f}s, {env.num_pedestrians} people, "
          f"interventions {'OFF' if args.no_interventions else 'ON'} ...")

    for tick in range(n_ticks):
        t = env.data.time

        # 1. Perception (each drone's own camera)
        percepts, tracks, cam_ok = {}, {}, {}
        tp = 0.0
        for d in ("alpha", "bravo"):
            rgb = env.render_camera(f"cam_{d}")
            fault = args.camera_fault and d == "alpha" and 70.0 <= t <= 74.0
            percepts[d] = detector.analyze_frame(rgb, coordinator.pos[d], simulate_dropout=fault)
            percepts[d]["pose"] = coordinator.pos[d].copy()        # where this frame was taken from
            tp += percepts[d]["latency_ms"]
            cam_ok[d] = percepts[d]["vision_confidence"] >= 0.5
            t0 = time.perf_counter()
            tracks[d] = trackers[d].update(percepts[d]["detections"], t, percepts[d]["footprint"]) \
                if cam_ok[d] else []
            lat["tracking"].append((time.perf_counter() - t0) * 1000.0)
        lat["perception"].append(tp)

        # 2. Fleet crowd map at 1 Hz
        if tick % CONTROL_HZ == 0:
            t0 = time.perf_counter()
            crowd_map.update(t, [{"drone": d, "xy": coordinator.pos[d][:2],
                                  "footprint": percepts[d]["footprint"], "tracks": tracks[d]}
                                 for d in ("alpha", "bravo") if cam_ok[d]])
            lat["analytics"].append((time.perf_counter() - t0) * 1000.0)

        # 3. Decide
        n_events = len(coordinator.events)
        out = coordinator.update(crowd_map, cam_ok, t, dt)
        lat["brain"].append(out["brain_ms"])
        for et, text in coordinator.events[n_events:]:
            say(f"  [{et:5.1f}s] {text}")

        # 4. Act
        for d in ("alpha", "bravo"):
            env.set_beacon(d, coordinator.beacon[d])
        ann = coordinator.pending_announcement
        if ann is not None:
            n = env.announce(ann["drone"], ann["hotspot"], ann["relief"])
            last_announcement = (t, "Please make space at the front and move to the sides."
                                 if len(ann["relief"]) else "Please stop pushing forward.", ann["drone"])
            coordinator.log(t, f"{n} people chose to follow the announcement")
            say(f"  [{t:5.1f}s] {coordinator.events[-1][1]}")

        # 5. Physics + video frames
        if writer is None:
            env.step(coordinator.pos, substeps=steps_per_tick)
        else:
            write_frames(env, writer, args, coordinator, crowd_map, detector, percepts, tracks,
                         t, last_announcement, frames_per_tick, steps_per_tick)

        # Offline evaluation sample (ground truth, NOT used by the drones)
        if tick % CONTROL_HZ == 0:
            peak, where = env.gt_peak_local_density(1.0)
            timeline.append({"t": round(t, 1), "gt_peak_density": round(float(peak), 2),
                             "incident_level": coordinator.incident["level"] if coordinator.incident else -1})
            if tick % (10 * CONTROL_HZ) == 0:
                say(f"t={t:5.1f}s  true peak density {peak:4.2f} p/m2  "
                    f"| alpha {coordinator.state['alpha']}, bravo {coordinator.state['bravo']}")

    env.close()
    summary = finish(args, base, coordinator, timeline, lat, wall0, video_path, say)
    if writer is not None:
        if cards:
            card = results_card(summary)
            for _ in range(int(4 * args.fps)):
                writer.write(card)
        writer.release()
    return summary


def results_card(summary):
    alert = next((e["t"] for e in summary["events"] if e["text"].startswith("MONITOR")), None)
    stats = [(f"{summary['gt_max_peak_density']:.1f}", "peak density, people/m²"),
             (f"{summary['gt_seconds_at_or_above_4']} s", "at ≥ 4 people/m²"),
             (f"{summary['gt_seconds_at_or_above_5']} s", "at ≥ 5 people/m² (dangerous)")]
    if summary["interventions"]:
        stats.append((f"{alert:.0f} s" if alert is not None else "–", "first alert"))
    heading = "This run, with drones intervening" if summary["interventions"] else \
        "This run, with drones only watching"
    return hud.end_card(VID_W, VID_H, heading, stats,
                        [f"Seed {summary['seed']}  ·  {summary['duration_s']:.0f} s  ·  densities from simulator "
                         "ground truth, measured after the run",
                         "Simulation prototype for decision support, not a validated crowd-safety system"])


def write_frames(env, writer, args, coordinator, crowd_map, detector, percepts, tracks,
                 t, last_announcement, frames_per_tick, steps_per_tick):
    """Advances physics for one control tick while writing the video frames."""
    sub = steps_per_tick // frames_per_tick
    show = coordinator.incident["responder"] if coordinator.incident else "alpha"
    frame = np.zeros((VID_H, VID_W, 3), np.uint8)
    frame[HEADER_H:HEADER_H + FRAME_H, OV_W:] = draw_drone_cam(
        percepts[show], tracks[show], show, percepts[show]["pose"], coordinator.state[show],
        detector, crowd_map, t, coordinator.beacon[show])
    frame[HEADER_H + FRAME_H:, OV_W:] = draw_map(crowd_map, coordinator, t)
    frame[HEADER_H + OV_H:, :OV_W] = draw_bottom(crowd_map, coordinator, t)
    for f in range(frames_per_tick):
        env.step(coordinator.pos, substeps=sub)
        now = env.data.time
        detail = ("  ·  YOLO detector" if getattr(args, "detector", "colour") == "yolo" else "") + \
                 ("  ·  varied crowd" if getattr(args, "appearance", "standard") == "varied" else "")
        frame[:HEADER_H] = draw_header(now, args.duration, not args.no_interventions, coordinator, detail)
        frame[HEADER_H:HEADER_H + OV_H, :OV_W] = draw_overview(env.render_camera("world_cam"), env,
                                                                coordinator, now, last_announcement)
        writer.write(frame)
    leftover = steps_per_tick - sub * frames_per_tick
    if leftover:
        env.step(coordinator.pos, substeps=leftover)


def finish(args, base, coordinator, timeline, lat, wall0, video_path, say):
    peaks = np.array([s["gt_peak_density"] for s in timeline])
    summary = {
        "interventions": not args.no_interventions,
        "seed": args.seed,
        "duration_s": args.duration,
        "detector": getattr(args, "detector", "colour"),
        "appearance": getattr(args, "appearance", "standard"),
        "gt_max_peak_density": float(peaks.max()),
        "gt_seconds_at_or_above_4": int((peaks >= 4.0).sum()),
        "gt_seconds_at_or_above_5": int((peaks >= 5.0).sum()),
        "security_alerted": any("SECURITY" in e[1] for e in coordinator.events),
        "events": [{"t": round(et, 1), "text": tx} for et, tx in coordinator.events],
        "latency_ms_mean": {k: round(float(np.mean(v)), 3) for k, v in lat.items() if v},
        "timeline": timeline,
    }
    with open(os.path.join(base, args.summary), "w") as f:
        json.dump(summary, f, indent=1)

    say("\n" + "=" * 70)
    say(f"Done in {time.perf_counter() - wall0:.0f}s wall time."
        + ("" if args.no_video else f" Video: {video_path}"))
    say("EVALUATION (simulator ground truth, peak density in any 1 m-radius circle):")
    say(f"  max peak density      {summary['gt_max_peak_density']:.2f} people/m2")
    say(f"  seconds >= 4 p/m2     {summary['gt_seconds_at_or_above_4']}")
    say(f"  seconds >= 5 p/m2     {summary['gt_seconds_at_or_above_5']}")
    say("Mean stage latency on THIS machine (ms): " +
        ", ".join(f"{k} {v}" for k, v in summary["latency_ms_mean"].items()))
    say(f"Summary written to {args.summary}")
    return summary


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
