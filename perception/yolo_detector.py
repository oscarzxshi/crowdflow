"""
Learned head detector: a YOLO11n model fine-tuned on simulator frames (detection/train.py).

Drop-in replacement for DensityPerceptionEngine: same analyze_frame() inputs and outputs, so
the tracker, crowd map and decisions are unchanged. Unlike the colour detector it does not
assume anything about skin tone, hair or clothing.
"""

import os
import time

import cv2
import numpy as np

from perception.density_detector import DensityPerceptionEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "models", "head_detector.pt")


class YoloPerceptionEngine(DensityPerceptionEngine):
    def __init__(self, frame_width=640, frame_height=480, fovy_deg=78.0,
                 model_path=DEFAULT_MODEL, conf=0.35, device=None):
        super().__init__(frame_width, frame_height, fovy_deg)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No detector weights at {model_path}. Train one with "
                                    "detection/make_dataset.py and detection/train.py.")
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device or "cpu"     # small model, 640x480: CPU is fast enough and portable

    def analyze_frame(self, rgb_frame, drone_pos, simulate_dropout=False):
        t0 = time.perf_counter()
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        if simulate_dropout:
            noise = np.random.randint(0, 256, bgr.shape, dtype=np.uint8)
            bgr = cv2.addWeighted(bgr, 0.10, noise, 0.90, 0)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        confidence = 0.12 if cv2.Laplacian(gray, cv2.CV_64F).var() > 3200.0 else 0.98

        detections, pixels, scores = [], [], []
        if confidence >= 0.5:
            res = self.model.predict(bgr, imgsz=640, conf=self.conf, device=self.device, verbose=False)[0]
            for (x0, y0, x1, y1), s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                pixels.append((u, v))
                scores.append(float(s))
                detections.append(self.pixel_to_world(u, v, drone_pos))

        return {
            "detections": np.array(detections).reshape(-1, 2),
            "pixels": pixels,
            "scores": scores,
            "footprint": self.footprint(drone_pos),
            "vision_confidence": confidence,
            "bgr": bgr,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
