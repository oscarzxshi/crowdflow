"""
Shared visual style for every video CrowdFlow renders (single run and side-by-side).

One typeface (DejaVu Sans, shipped with matplotlib), one neutral dark palette, one accent
colour for drone activity, and status colours used ONLY for crowd status.

Drawing model: shapes are drawn straight onto a BGR numpy image with OpenCV; text is queued
and drawn in a single Pillow pass at the end (OpenCV's built-in fonts look like debug output).
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

# ---------------------------------------------------------------- palette (BGR)
def bgr(hex_colour):
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


BG = bgr("#0f1012")
PANEL = bgr("#16171a")
BORDER = bgr("#2a2c31")
TEXT = bgr("#e6e7ea")
TEXT_2 = bgr("#9a9ca4")
TEXT_3 = bgr("#6b6e76")
ACCENT = bgr("#4c8df6")          # drone activity (spotlight, announcements, active steps)
STATUS = {                        # crowd status only
    None: bgr("#33353b"),         # unknown (not observed)
    0: bgr("#3fb27f"),            # normal
    1: bgr("#d9b13b"),            # developing
    2: bgr("#ef8a3a"),            # high
    3: bgr("#e5484d"),            # critical
}
STATUS_LABEL = {None: "Unknown", 0: "Normal", 1: "Developing", 2: "High", 3: "Critical"}
OFF_COLOUR = bgr("#d95926")      # comparison: drones only watch
ON_COLOUR = bgr("#3987e5")       # comparison: drones intervene

# ---------------------------------------------------------------- fonts
_FONT_CACHE = {}


def font(size, weight="normal", mono=False):
    key = (size, weight, mono)
    if key not in _FONT_CACHE:
        family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
        path = font_manager.findfont(font_manager.FontProperties(family=family, weight=weight))
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def text_width(s, size, weight="normal", mono=False):
    return font(size, weight, mono).getlength(s)


def fit(s, size, max_w, weight="normal", mono=False):
    """Truncates `s` with an ellipsis so it fits in max_w pixels."""
    if text_width(s, size, weight, mono) <= max_w:
        return s
    while s and text_width(s + "…", size, weight, mono) > max_w:
        s = s[:-1]
    return s.rstrip() + "…"


def wrap(s, size, max_w, max_lines=2, weight="normal"):
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_width(trial, size, weight) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = fit(lines[-1] + " …", size, max_w, weight)
    return lines


# ---------------------------------------------------------------- canvas
class Canvas:
    """BGR image + queued text, rendered in one Pillow pass by `finish()`."""

    def __init__(self, w, h, fill=BG, img=None):
        self.img = img if img is not None else np.full((h, w, 3), fill, np.uint8)
        self._text = []

    # shapes
    def rect(self, x0, y0, x1, y1, colour, filled=True, thickness=1):
        cv2.rectangle(self.img, (int(x0), int(y0)), (int(x1), int(y1)), colour, -1 if filled else thickness)

    def rounded(self, x0, y0, x1, y1, colour, r=6):
        x0, y0, x1, y1, r = int(x0), int(y0), int(x1), int(y1), int(r)
        cv2.rectangle(self.img, (x0 + r, y0), (x1 - r, y1), colour, -1)
        cv2.rectangle(self.img, (x0, y0 + r), (x1, y1 - r), colour, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(self.img, (cx, cy), r, colour, -1, cv2.LINE_AA)

    def line(self, p0, p1, colour, thickness=1):
        cv2.line(self.img, tuple(map(int, p0)), tuple(map(int, p1)), colour, thickness, cv2.LINE_AA)

    def circle(self, c, r, colour, thickness=-1):
        cv2.circle(self.img, tuple(map(int, c)), int(r), colour, thickness, cv2.LINE_AA)

    def shade(self, x0, y0, x1, y1, colour=(0, 0, 0), alpha=0.6):
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), int(x1), int(y1)
        roi = self.img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - alpha) + np.array(colour) * alpha).astype(np.uint8)

    # text
    def text(self, xy, s, size=15, colour=TEXT, weight="normal", mono=False, anchor="la"):
        self._text.append((xy, s, size, colour, weight, mono, anchor))

    def chip(self, x, y, s, colour, size=13, text_colour=None, weight="bold"):
        """Small rounded label; returns its right edge."""
        w = text_width(s, size, weight) + 16
        h = size + 10
        self.rounded(x, y, x + w, y + h, colour, r=h // 2)
        self.text((x + 8, y + h / 2), s, size, text_colour or bgr("#0f1012"), weight, anchor="lm")
        return x + w

    def finish(self):
        if self._text:
            pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            for xy, s, size, colour, weight, mono, anchor in self._text:
                draw.text(xy, s, fill=(colour[2], colour[1], colour[0]),
                          font=font(size, weight, mono), anchor=anchor)
            self.img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            self._text = []
        return self.img


def section_title(cv, x, y, s, right=None):
    cv.text((x, y), s.upper(), 12, TEXT_3, "bold")
    if right:
        cv.text((right[0], y), right[1], 12, TEXT_3, anchor="ra")


# ---------------------------------------------------------------- cards
def title_card(w, h, title, subtitle, lines):
    cv = Canvas(w, h)
    cy = h // 2 - 70
    cv.rect(w // 2 - 28, cy - 50, w // 2 + 28, cy - 46, ACCENT)
    cv.text((w / 2, cy), title, 64, TEXT, "bold", anchor="mm")
    cv.text((w / 2, cy + 64), subtitle, 26, TEXT_2, anchor="mm")
    for k, s in enumerate(lines):
        cv.text((w / 2, cy + 130 + 32 * k), s, 18, TEXT_3, anchor="mm")
    return cv.finish()


def end_card(w, h, heading, stats, footer):
    """stats: list of (big value, label)."""
    cv = Canvas(w, h)
    cv.text((w / 2, h / 2 - 170), heading, 34, TEXT, "bold", anchor="mm")
    n = len(stats)
    col_w = min(380, (w - 200) / max(n, 1))
    x0 = w / 2 - col_w * n / 2
    for k, (value, label) in enumerate(stats):
        cx = x0 + col_w * (k + 0.5)
        cv.text((cx, h / 2 - 40), value, 58, TEXT, "bold", anchor="mm")
        for j, line in enumerate(wrap(label, 18, col_w - 40, 2)):
            cv.text((cx, h / 2 + 30 + 26 * j), line, 18, TEXT_2, anchor="mm")
    for j, line in enumerate(footer):
        cv.text((w / 2, h - 120 + 28 * j), line, 16, TEXT_3, anchor="mm")
    return cv.finish()


def project_to_image(model, data, cam_name, xyz, width, height):
    """Pixel position of a world point in a MuJoCo camera image (None if behind the camera)."""
    import mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    pos, mat = data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3)
    rel = mat.T @ (np.asarray(xyz) - pos)            # camera frame: x right, y up, looks along -z
    if rel[2] >= -1e-6:
        return None
    f = (height / 2) / np.tan(np.radians(model.cam_fovy[cid]) / 2)
    return (width / 2 + f * rel[0] / -rel[2], height / 2 - f * rel[1] / -rel[2])
