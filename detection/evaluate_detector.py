#!/usr/bin/env python3
"""
Compares the colour detector and the YOLO detector against simulator ground truth.

Frames come from evaluation crowds (seeds 1-3) that the detector never saw in training
(training uses seeds 1000+). Each frame is rendered once and given to both detectors.
A detection counts as correct if it lies within 0.3 m of a real head (one-to-one matching).

    recall     share of real heads that were found
    precision  share of detections that were real heads
    error      median distance between a matched detection and the real head

Usage:
    python detection/evaluate_detector.py
    python detection/evaluate_detector.py --frames 60
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.drone_env import CrowdFleetEnv                         # noqa: E402
from perception.density_detector import DensityPerceptionEngine  # noqa: E402
from perception.yolo_detector import YoloPerceptionEngine, DEFAULT_MODEL  # noqa: E402

MATCH_M = 0.30
POSES = [([-7.5, 0.3, 5.5], True), ([-4.5, 2.0, 7.0], False), ([4.5, -2.0, 7.0], False),
         ([-6.0, -1.0, 6.0], True), ([0.0, 0.0, 8.5], False)]


def score(det, gt, footprint, stats):
    x0, x1, y0, y1 = footprint
    inner = lambda p: (p[:, 0] > x0 + .3) & (p[:, 0] < x1 - .3) & (p[:, 1] > y0 + .3) & (p[:, 1] < y1 - .3)
    gt = gt[inner(gt)]
    det_in = det[inner(det)] if len(det) else det
    matched = 0
    if len(det) and len(gt):
        c = np.linalg.norm(gt[:, None] - det[None], axis=2)
        r, k = linear_sum_assignment(c)
        ok = c[r, k] < MATCH_M
        matched = int(ok.sum())
        stats["err"] += list(c[r, k][ok])
    stats["tp"] += matched
    stats["fn"] += len(gt) - matched
    stats["fp"] += max(0, len(det_in) - matched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=15, help="sampled moments per crowd (x5 camera poses)")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    detectors = {"colour": DensityPerceptionEngine(), "yolo": YoloPerceptionEngine(model_path=args.model)}
    results = {}
    for appearance in ("standard", "varied"):
        stats = {d: {"tp": 0, "fp": 0, "fn": 0, "err": [], "ms": []} for d in detectors}
        for seed in map(int, args.seeds.split(",")):
            env = CrowdFleetEnv(seed=seed, appearance=appearance)
            every = max(1, int(90 / 0.1 / args.frames))
            for k in range(int(90 / 0.1)):
                env.step({"alpha": [-4.5, 0, 7], "bravo": [4.5, 0, 7]}, substeps=20)
                if k % every != every - 1:
                    continue
                for pose, beacon in POSES:
                    env.set_beacon("alpha", beacon)
                    env.step({"alpha": pose, "bravo": [4.5, 0, 7]}, substeps=1)
                    img = env.render_camera("cam_alpha")
                    gt = env.get_pedestrian_positions()
                    for name, det in detectors.items():
                        r = det.analyze_frame(img, pose)
                        stats[name]["ms"].append(r["latency_ms"])
                        score(r["detections"], gt, r["footprint"], stats[name])
            env.close()
        for name, s in stats.items():
            results[f"{name}/{appearance}"] = {
                "recall": s["tp"] / max(1, s["tp"] + s["fn"]),
                "precision": s["tp"] / max(1, s["tp"] + s["fp"]),
                "median_error_m": float(np.median(s["err"])) if s["err"] else None,
                "latency_ms": float(np.median(s["ms"])),
                "heads": s["tp"] + s["fn"],
            }

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "detector_comparison.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("| Detector | Crowd | Recall | Precision | Median error | Time per frame |")
    print("|---|---|---|---|---|---|")
    for key, r in results.items():
        name, appearance = key.split("/")
        err = "–" if r["median_error_m"] is None else f"{r['median_error_m'] * 100:.0f} cm"
        print(f"| {'Colour' if name == 'colour' else 'YOLO'} | {appearance} | {r['recall']:.1%} | "
              f"{r['precision']:.1%} | {err} | {r['latency_ms']:.0f} ms |")
    print(f"\n{results['yolo/varied']['heads']} heads per crowd type; saved results/detector_comparison.json")


if __name__ == "__main__":
    main()
