#!/usr/bin/env python3
"""
Builds a YOLO training set from the simulator, labelled automatically from ground truth.

For each episode (a crowd seed), the crowd is simulated for 90 s. Every few seconds a drone
camera is placed at several random poses over the venue (4-9 m up) and the frame is saved.
Every person's head position is known exactly, so each frame gets a perfect label file -
no hand labelling.

Mix: 75% of episodes use the "varied" appearance (skin tones, hair, caps, warm clothing),
25% the "standard" one, so the model learns people, not one colour. Training and validation
use different crowds (episodes), so validation scores are not inflated by near-duplicates.

Output (YOLO format):
    datasets/heads/images/{train,val}/*.jpg
    datasets/heads/labels/{train,val}/*.txt     one line per head: 0 cx cy w h (normalised)
    datasets/heads/heads.yaml

Usage:
    python detection/make_dataset.py                       # 12 episodes, ~1,400 images
    python detection/make_dataset.py --episodes 24 --poses 5
"""

import argparse
import os
import sys

import cv2
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.drone_env import CrowdFleetEnv          # noqa: E402

CAM = "cam_alpha"
BOX_RADIUS = 0.19          # m around the head centre (head 0.15 m + cap/hair margin)


def head_labels(env, w, h):
    """YOLO boxes for every head visible in CAM, from the rendered head positions."""
    m, d = env.model, env.data
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
    cam_pos, cam_mat = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
    f = (h / 2) / np.tan(np.radians(m.cam_fovy[cid]) / 2)
    lines = []
    for i in range(env.num_pedestrians):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"ped_{i}_head")
        rel = cam_mat.T @ (d.geom_xpos[g] - cam_pos)       # camera frame, looking along -z
        depth = -rel[2]
        if depth <= 0.2:
            continue
        u = w / 2 + f * rel[0] / depth
        v = h / 2 - f * rel[1] / depth
        r = f * BOX_RADIUS / depth
        x0, x1, y0, y1 = u - r, u + r, v - r, v + r
        cx0, cx1, cy0, cy1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < 0.4 * (x1 - x0) * (y1 - y0):   # mostly outside the frame
            continue
        lines.append(f"0 {(cx0 + cx1) / 2 / w:.6f} {(cy0 + cy1) / 2 / h:.6f} "
                     f"{(cx1 - cx0) / w:.6f} {(cy1 - cy0) / h:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--val-episodes", type=int, default=2)
    ap.add_argument("--poses", type=int, default=4, help="random camera poses per sampled moment")
    ap.add_argument("--every", type=float, default=3.0, help="seconds between sampled moments")
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "heads"))
    ap.add_argument("--first-seed", type=int, default=1000, help="seeds kept apart from evaluation seeds")
    args = ap.parse_args()

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    rng = np.random.default_rng(42)
    n_img = n_box = 0
    for ep in range(args.episodes):
        seed = args.first_seed + ep
        split = "val" if ep >= args.episodes - args.val_episodes else "train"
        appearance = "varied" if (ep % 4 != 3) else "standard"
        env = CrowdFleetEnv(seed=seed, appearance=appearance)
        park = {"alpha": [-4.5, 0.0, 7.0], "bravo": [4.5, 0.0, 7.0]}
        t_next = 0.0
        while env.data.time < args.duration:
            env.step(park, substeps=20)
            if env.data.time < t_next:
                continue
            t_next += args.every
            for k in range(args.poses):
                pose = [rng.uniform(-9.0, 8.0), rng.uniform(-6.5, 6.5), rng.uniform(4.0, 9.0)]
                env.set_beacon("alpha", rng.random() < 0.25)
                env.data.mocap_pos[env.drone_mocap["alpha"]] = pose
                env._place_disc("alpha")
                mujoco.mj_forward(env.model, env.data)
                img = env.render_camera(CAM)
                h, w = img.shape[:2]
                labels = head_labels(env, w, h)
                name = f"s{seed}_t{env.data.time:05.1f}_p{k}"
                cv2.imwrite(os.path.join(args.out, "images", split, name + ".jpg"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(os.path.join(args.out, "labels", split, name + ".txt"), "w") as f:
                    f.write("\n".join(labels) + ("\n" if labels else ""))
                n_img += 1
                n_box += len(labels)
            env.data.mocap_pos[env.drone_mocap["alpha"]] = park["alpha"]
            env.set_beacon("alpha", False)
        env.close()
        print(f"  episode {ep + 1}/{args.episodes} (seed {seed}, {appearance}, {split}): "
              f"{n_img} images, {n_box} heads so far")

    with open(os.path.join(args.out, "heads.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\n"
                "names:\n  0: head\n")
    print(f"Done: {n_img} images, {n_box} labelled heads -> {args.out}")


if __name__ == "__main__":
    main()
