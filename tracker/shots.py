"""Shot (camera-cut) detection and static-overlay detection for real match recordings.

Broadcast footage contains hard cuts (to a close-up, a replay, another camera) and
static graphics (score bug, channel logo, watermark).

* Cuts: every frame is reduced to a 32x8 Hue-Saturation histogram. A cut is declared when
  the Bhattacharyya distance between consecutive histograms is large in absolute terms
  (> 0.35) AND is a clear spike compared with the recent motion (> 4x the running median).
  Tracking across a cut is meaningless (different camera), so routes are split there.
* Overlays: pixels whose intensity barely changes over the whole clip while the camera
  moves are graphics burnt into the video. They are excluded from camera-motion feature
  points, otherwise they would vote for "no camera movement".
"""
import json
import os

import cv2
import numpy as np


def frame_signature(bgr):
    small = cv2.resize(bgr, (160, max(1, int(160 * bgr.shape[0] / bgr.shape[1]))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 8], [0, 180, 0, 256])
    cv2.normalize(h, h, 1.0, 0, cv2.NORM_L1)
    return h


class CutDetector:
    def __init__(self):
        self.prev = None
        self.hist = []

    def update(self, bgr):
        sig = frame_signature(bgr)
        cut = False
        if self.prev is not None:
            d = cv2.compareHist(self.prev, sig, cv2.HISTCMP_BHATTACHARYYA)
            med = float(np.median(self.hist[-25:])) if self.hist else 0.0
            cut = d > 0.35 and d > 4 * max(med, 0.02)
            self.hist.append(d)
        self.prev = sig
        return cut


def analyse(video, out_json, overlay_png, progress=None):
    """One pass over the clip: cut positions + static overlay mask. Writes JSON + PNG."""
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    det = CutDetector()
    cuts, stack = [], []
    every = max(1, n // 60)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if det.update(f) and i > 0:
            cuts.append(i)
        if i % every == 0:
            g = cv2.cvtColor(cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY)
            stack.append(g)
        i += 1
        if progress and i % 50 == 0:
            progress(i / max(n, 1))
    cap.release()
    overlay = None
    if len(stack) >= 8:
        S = np.stack(stack).astype(np.float32)
        std = S.std(axis=0)
        static = (std < 3.0).astype(np.uint8) * 255
        # graphics have structure (text, borders); keep static areas with some edges nearby
        edges = cv2.Canny(S.mean(axis=0).astype(np.uint8), 60, 160)
        near_edges = cv2.dilate(edges, np.ones((9, 9), np.uint8))
        overlay = cv2.bitwise_and(static, near_edges)
        overlay = cv2.dilate(overlay, np.ones((7, 7), np.uint8))
        frac = overlay.mean() / 255
        if frac > 0.5:  # camera never moved (tripod): everything is static, not an overlay
            overlay = None
    if overlay is not None:
        cv2.imwrite(overlay_png, overlay)
    elif os.path.exists(overlay_png):
        os.remove(overlay_png)
    res = {"frames_scanned": i, "cuts": cuts, "overlay": overlay is not None,
           "overlay_fraction": round(float(overlay.mean() / 255), 4) if overlay is not None else 0.0}
    with open(out_json, "w") as fh:
        json.dump(res, fh)
    return res


def load_overlay(path, W, H):
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if m is None else cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
