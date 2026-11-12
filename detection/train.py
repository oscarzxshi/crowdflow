#!/usr/bin/env python3
"""
Trains the head detector (YOLO11n, the smallest Ultralytics model) on the simulator dataset.

Starts from weights pre-trained on COCO and fine-tunes them on top-down drone frames.
Picks the fastest device available: Apple Silicon GPU ("mps"), NVIDIA GPU ("0"), else CPU.
The best checkpoint is copied to models/head_detector.pt, which the simulation loads.

Usage:
    python detection/make_dataset.py            # first, once
    python detection/train.py                   # 40 epochs
    python detection/train.py --epochs 10       # quick test
"""

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "heads", "heads.yaml"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default=os.path.join(ROOT, "outputs", "yolo11n.pt"),
                    help="starting weights (downloaded there on first use)")
    ap.add_argument("--device", default=None, help="mps / 0 / cpu (default: auto)")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"No dataset at {args.data}. Run: python detection/make_dataset.py")

    from ultralytics import YOLO
    device = args.device or pick_device()
    print(f"Training on device: {device}")
    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        project=os.path.join(ROOT, "outputs", "training"), name="head_detector", exist_ok=True,
        # Top-down view: flips in both directions are realistic; keep colour jitter moderate.
        flipud=0.5, fliplr=0.5, hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, mosaic=1.0,
        patience=15, plots=True, verbose=False,
    )
    best = os.path.join(ROOT, "outputs", "training", "head_detector", "weights", "best.pt")
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    shutil.copy(best, os.path.join(ROOT, "models", "head_detector.pt"))
    print(f"Saved models/head_detector.pt (from {best})")


if __name__ == "__main__":
    main()
