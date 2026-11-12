#!/usr/bin/env python3
"""
Side-by-side comparison video (1920x1080): the same crowd (same seed) without and with CrowdFlow.

    left   drones only watch     (--no-interventions)
    right  drones intervene
    bottom live chart of the true peak density in both runs, with the escalation steps marked

Usage:
    python side_by_side.py                         # seed 5, 90 s -> side_by_side.mp4
    python side_by_side.py --seed 5 --fps 20
    python side_by_side.py --clip 15 50 --gif      # also export a short clip (+ GIF) for posts

Densities in the chart are simulator ground truth, measured for evaluation only.
"""

import argparse
import json
import os
import numpy as np
import cv2

import run_simulation as rs
from viz import hud

W, H = 1920, 1080
HEADER_H = 60
PANE_W, PANE_H = 960, 540
CHART_Y = HEADER_H + PANE_H
CHART_H = H - CHART_Y


def simulate(seed, interventions, duration, fps, base):
    args = rs.build_parser().parse_args([])
    tag = "on" if interventions else "off"
    args.seed, args.duration, args.fps = seed, duration, fps
    args.no_interventions = not interventions
    args.no_cards = True
    args.out = os.path.join("results", f"sbs_seed{seed}_{tag}.mp4")
    args.summary = os.path.join("results", f"sbs_seed{seed}_{tag}.json")
    args.quiet = True
    print(f"  simulating seed {seed}, drones {'intervene' if interventions else 'only watch'} ...")
    rs.run(args)
    with open(os.path.join(base, args.summary)) as f:
        return os.path.join(base, args.out), json.load(f)


def header(now, duration):
    cv = hud.Canvas(W, HEADER_H)
    cv.rect(0, HEADER_H - 1, W, HEADER_H, hud.BORDER)
    cv.text((28, HEADER_H / 2), "CrowdFlow", 22, hud.TEXT, "bold", anchor="lm")
    cv.text((28 + hud.text_width("CrowdFlow", 22, "bold") + 20, HEADER_H / 2),
            "Same simulated crowd, same random seed: without vs with the drones intervening",
            15, hud.TEXT_2, anchor="lm")
    cv.text((W - 28, HEADER_H / 2), f"{now:5.1f} s / {duration:.0f} s", 16, hud.TEXT, mono=True, anchor="rm")
    return cv.finish()


def pane(frame, label, colour, peak):
    ov = frame[rs.HEADER_H:rs.HEADER_H + rs.OV_H, :rs.OV_W]
    img = cv2.resize(ov, (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)
    cv = hud.Canvas(PANE_W, PANE_H, img=img)
    cv.shade(0, 0, PANE_W, 52, hud.BG, 0.85)
    cv.rect(20, 18, 34, 32, colour)
    cv.text((44, 26), label, 17, hud.TEXT, "bold", anchor="lm")
    if peak is not None:
        cv.text((PANE_W - 20, 26), f"peak density now  {peak:.1f} people/m²", 14, hud.TEXT_2, anchor="rm")
    return cv.finish()


class DensityChart:
    """Live line chart of both runs' true peak density, revealed as the video plays."""

    def __init__(self, off, on, duration):
        self.t = np.array([s["t"] for s in off["timeline"]])
        self.off = np.array([s["gt_peak_density"] for s in off["timeline"]])
        self.on = np.array([s["gt_peak_density"] for s in on["timeline"]])
        self.duration = duration
        self.events, seen = [], set()
        for e in on["events"]:
            for key, short in (("SPOTLIGHT:", "Spotlight"), ("ANNOUNCE:", "Announcement"),
                               ("SECURITY:", "Security alerted")):
                if e["text"].startswith(key) and key not in seen:
                    seen.add(key)
                    self.events.append((e["t"], short))
        self.x0, self.x1 = 96, W - 240
        self.y0, self.y1 = 64, CHART_H - 52
        self.ymax = 7.0

    def px(self, t, v):
        return (self.x0 + (self.x1 - self.x0) * t / self.duration,
                self.y1 - (self.y1 - self.y0) * min(v, self.ymax) / self.ymax)

    def value_at(self, series, now):
        k = int(np.searchsorted(self.t, now, side="right"))
        return series[k - 1] if k else None

    def draw(self, now):
        cv = hud.Canvas(W, CHART_H, hud.PANEL)
        cv.rect(0, 0, W, 1, hud.BORDER)
        hud.section_title(cv, 28, 22, "Peak crowd density, people per m²",
                          (W - 28, "simulator ground truth, measured for evaluation"))
        for v in range(0, 8):
            _, y = self.px(0, v)
            strong = v in (4, 5)
            cv.line((self.x0, y), (self.x1, y), hud.TEXT_3 if strong else hud.BORDER, 1)
            cv.text((self.x0 - 14, y), f"{v}", 13, hud.TEXT_2, anchor="rm")
        cv.text((self.x1 + 14, self.px(0, 5)[1]), "5  dangerous", 13, hud.TEXT_2, anchor="lm")
        cv.text((self.x1 + 14, self.px(0, 4)[1]), "4  movement restricted", 13, hud.TEXT_2, anchor="lm")
        for s in range(0, int(self.duration) + 1, 10):
            x, _ = self.px(s, 0)
            cv.text((x, self.y1 + 22), f"{s} s", 13, hud.TEXT_2, anchor="mm")
        for k, (et, name) in enumerate(self.events):
            if et <= now:
                x, _ = self.px(et, 0)
                cv.line((x, self.y0), (x, self.y1), hud.ON_COLOUR, 1)
                cv.text((x + 6, self.y0 + 2 + 18 * k), name, 12, hud.TEXT_2)
        k = int(np.searchsorted(self.t, now, side="right"))
        for series, col in ((self.off, hud.OFF_COLOUR), (self.on, hud.ON_COLOUR)):
            if k >= 2:
                pts = np.array([self.px(tt, vv) for tt, vv in zip(self.t[:k], series[:k])], np.int32)
                cv2.polylines(cv.img, [pts], False, col, 3, cv2.LINE_AA)
            if k >= 1:
                end = self.px(self.t[k - 1], series[k - 1])
                cv.circle(end, 9, hud.PANEL)
                cv.circle(end, 6, col)
        return cv.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="side_by_side.mp4")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("START", "END"),
                    help="also write a short clip between these times (seconds)")
    ap.add_argument("--gif", action="store_true", help="also write the clip as a GIF (needs Pillow)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)
    os.makedirs("results", exist_ok=True)
    print(f"Side-by-side for seed {args.seed}:")
    off_path, off = simulate(args.seed, False, args.duration, args.fps, base)
    on_path, on = simulate(args.seed, True, args.duration, args.fps, base)

    cap_off, cap_on = cv2.VideoCapture(off_path), cv2.VideoCapture(on_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))
    clip_writer, gif_frames = None, []
    if args.clip:
        clip_name = os.path.splitext(args.out)[0] + "_clip.mp4"
        clip_writer = cv2.VideoWriter(clip_name, fourcc, args.fps, (W, H))

    title = hud.title_card(W, H, "CrowdFlow", "Drones that spot a crowd crush forming, and act before it does",
                           ["The same simulated crowd, twice:",
                            "left, the drones only watch  ·  right, they spotlight, announce and alert security"])
    for _ in range(int(3 * args.fps)):
        writer.write(title)

    chart = DensityChart(off, on, args.duration)
    frame_i = 0
    while True:
        ok1, f_off = cap_off.read()
        ok2, f_on = cap_on.read()
        if not (ok1 and ok2):
            break
        now = frame_i / args.fps
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:HEADER_H] = header(now, args.duration)
        frame[HEADER_H:CHART_Y, :PANE_W] = pane(f_off, "Drones only watch", hud.OFF_COLOUR,
                                                chart.value_at(chart.off, now))
        frame[HEADER_H:CHART_Y, PANE_W:] = pane(f_on, "Drones intervene", hud.ON_COLOUR,
                                                chart.value_at(chart.on, now))
        frame[HEADER_H:CHART_Y, PANE_W:PANE_W + 1] = hud.BORDER
        frame[CHART_Y:] = chart.draw(now)
        writer.write(frame)
        if args.clip and args.clip[0] <= now <= args.clip[1]:
            clip_writer.write(frame)
            if args.gif and frame_i % max(1, args.fps // 6) == 0:      # GIF at ~6 fps, 720 px wide
                gif_frames.append(cv2.cvtColor(cv2.resize(frame, (720, 405), interpolation=cv2.INTER_AREA),
                                               cv2.COLOR_BGR2RGB))
        frame_i += 1

    end = hud.end_card(W, H, "Same crowd, without vs with CrowdFlow", [
        (f"{off['gt_seconds_at_or_above_5']} s → {on['gt_seconds_at_or_above_5']} s", "at ≥ 5 people/m² (dangerous)"),
        (f"{off['gt_seconds_at_or_above_4']} s → {on['gt_seconds_at_or_above_4']} s", "at ≥ 4 people/m²"),
        (f"{off['gt_max_peak_density']:.1f} → {on['gt_max_peak_density']:.1f}", "peak density, people/m²")],
        [f"Seed {args.seed}  ·  {args.duration:.0f} s  ·  densities from simulator ground truth",
         "Simulation prototype for decision support, not a validated crowd-safety system"])
    for _ in range(int(5 * args.fps)):
        writer.write(end)
    writer.release()
    print(f"Wrote {args.out}")
    if clip_writer is not None:
        clip_writer.release()
        print(f"Wrote {clip_name}")
    if gif_frames:
        try:
            from PIL import Image
            imgs = [Image.fromarray(f).quantize(colors=128) for f in gif_frames]
            gif_name = os.path.splitext(args.out)[0] + "_clip.gif"
            imgs[0].save(gif_name, save_all=True, append_images=imgs[1:],
                         duration=int(1000 / 6), loop=0, optimize=True)
            print(f"Wrote {gif_name}")
        except ImportError:
            print("Pillow not installed - skipped the GIF (pip install pillow)")

    print(f"\nSeed {args.seed}: peak density {off['gt_max_peak_density']:.2f} without vs "
          f"{on['gt_max_peak_density']:.2f} with CrowdFlow; seconds >= 5 people/m2: "
          f"{off['gt_seconds_at_or_above_5']} vs {on['gt_seconds_at_or_above_5']}")


if __name__ == "__main__":
    main()
