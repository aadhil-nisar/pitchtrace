"""Semi-real stress clip with ground truth, built from a REAL broadcast frame.

The grass background, the player sprites and the ball sprite are cut out of
docs/sample_input_frames/frame_000000.jpg (players are inpainted out of the background),
then animated along scripted paths with a panning camera, so the result looks like real
broadcast footage but every position is known. Scenarios are the ones that break
trackers on real matches:

  * two team-mates in the same kit crossing each other (X-crossing)
  * a target running side by side with a team-mate, then the team-mate peeling off
  * an opponent overtaking a target closely
  * the ball dribbled at a player's feet (merges with the player's blob)
  * fast ground passes and a long lofted pass (motion blur, ball grows in the air)
  * the ball passing behind a player (short occlusion)

    python tools/make_semireal_clip.py out.mp4 gt.json [seed]
    python tools/evaluate_semireal.py out.mp4 gt.json [frame_step]

A seed gives a variant of the same scenarios (key points moved by up to 22 px).
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker.segmentation import PitchSegmenter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "sample_input_frames", "frame_000000.jpg")
FPS, SECONDS = 25, 14
N = FPS * SECONDS
OW, OH = 960, 540
SCALE = 1.15  # world = source frame upscaled; the camera window pans over it
WHITE = {12: [111, 256, 17, 40], 13: [293, 295, 22, 41], 18: [178, 361, 19, 52], 20: [685, 409, 40, 46]}
LIME = {14: [786, 307, 21, 40], 15: [424, 322, 23, 37], 16: [267, 344, 22, 49], 11: [166, 243, 15, 40]}
REF = {19: [925, 403, 23, 43]}
BALL = [592, 423, 11, 11]


def cut(img, grass, box, pad=3):
    x, y, w, h = box
    x0, y0, x1, y1 = x - pad, y - pad, x + w + pad, y + h + pad
    rgb = img[y0:y1, x0:x1].astype(np.float32)
    a = (grass[y0:y1, x0:x1] == 0).astype(np.uint8) * 255
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(a, connectivity=8)
    if n > 1:  # keep the biggest part (the player)
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        keep = [i for i in range(1, n) if i == big or st[i, cv2.CC_STAT_AREA] > 0.15 * st[big, cv2.CC_STAT_AREA]]
        a = np.isin(lab, keep).astype(np.uint8) * 255
    a = cv2.GaussianBlur(a, (3, 3), 0).astype(np.float32) / 255
    return rgb, a


def paste(canvas, rgb, a, cx, foot_y, s=1.0, anchor="foot"):
    if s != 1.0:
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    h, w = a.shape
    x0 = int(round(cx - w / 2))
    y0 = int(round(foot_y - h)) if anchor == "foot" else int(round(foot_y - h / 2))
    H, W = canvas.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return None
    sub = canvas[ya:yb, xa:xb]
    al = a[ya - y0:yb - y0, xa - x0:xb - x0, None]
    sub[:] = sub * (1 - al) + rgb[ya - y0:yb - y0, xa - x0:xb - x0] * al
    ys, xs = np.where(a > 0.35)
    if len(xs) == 0:
        return None
    return [x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]


JIT = None  # random generator for scenario variants (set from the optional seed argument)


def lerp_path(keys):
    """keys: list of (t_seconds, x, y) -> per-frame array"""
    t = np.arange(N) / FPS
    k = np.array(keys, np.float64)
    if JIT is not None:  # variant: move every key point a little (scenario stays the same)
        k[1:, 1:] += JIT.uniform(-22, 22, (len(k) - 1, 2))
    return np.stack([np.interp(t, k[:, 0], k[:, 1]), np.interp(t, k[:, 0], k[:, 2])], 1)


def main(out_path, gt_path, seed=None):
    global JIT
    JIT = np.random.default_rng(seed) if seed is not None else None
    img = cv2.imread(SRC)
    seg = PitchSegmenter(img)
    grass = seg.grass_mask(img)
    m = seg.masks(img)
    # background: inpaint every player / ball blob (not the pitch lines)
    hole = cv2.bitwise_and(m["fg_raw"], cv2.bitwise_not(m["lines"]))
    hole = cv2.bitwise_and(hole, m["field"])
    for b in seg.detect_players(m["fg"], *img.shape[:2]) + [BALL]:
        x, y, w, h = b
        hole[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] |= (grass[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] == 0).astype(np.uint8) * 255
    hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))
    hole[:118] = 0  # keep stands / boards untouched
    bg = cv2.inpaint(img, hole, 5, cv2.INPAINT_TELEA)
    world = cv2.resize(bg, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC).astype(np.float32)

    sw = {k: cut(img, grass, b) for k, b in WHITE.items()}
    sl = {k: cut(img, grass, b) for k, b in LIME.items()}
    sr = {k: cut(img, grass, b) for k, b in REF.items()}
    bx, by, bw, bh = BALL
    ball_rgb = img[by:by + bh, bx:bx + bw].astype(np.float32)
    yy, xx = np.mgrid[0:bh, 0:bw]
    ball_a = np.clip(1.0 - (np.hypot(xx - (bw - 1) / 2, yy - (bh - 1) / 2) - 2.6) / 1.5, 0, 1).astype(np.float32)

    # ----- actors in WORLD coordinates (world = 1104 x 621); foot position
    S = SCALE
    actors = {
        # A: same-team crossing (white vs white), cross at t=3s, W2 passes IN FRONT of W1
        "W1": ("white", [12, 13], lerp_path([(0, 300, 420), (3, 520, 424), (6, 740, 430), (9, 820, 380), (14, 900, 330)])),
        "W2": ("white", [13, 12], lerp_path([(0, 740, 430), (3, 520, 428), (6, 300, 418), (9.5, 230, 400), (11, 520, 420), (14, 620, 420)])),
        # B: lime target shoulder to shoulder with a lime team-mate (blobs merge), then the
        # team-mate peels off
        "L1": ("lime", [14, 15], lerp_path([(0, 420, 300), (2, 520, 305), (5, 640, 312), (8, 700, 360), (14, 780, 470)])),
        "L2": ("lime", [15, 14], lerp_path([(0, 430, 312), (2, 530, 316), (5, 652, 322), (6.5, 640, 380), (14, 480, 520)])),
        # C: same-team overtake at a shallow angle: W3 runs past W1 almost in parallel
        "W3": ("white", [12, 13], lerp_path([(0, 330, 280), (5, 600, 300), (6.5, 690, 420), (8, 800, 410), (9, 870, 375), (14, 1000, 330)])),
        # distractors
        "L3": ("lime", [15, 14], lerp_path([(0, 880, 250), (14, 860, 280)])),
        "W4": ("white", [20, 18], lerp_path([(0, 200, 230), (14, 260, 260)])),
        "RF": ("ref", [19, 19], lerp_path([(0, 560, 200), (14, 640, 230)])),
    }
    phase = {n: i for i, n in enumerate(actors)}
    # ball: dribbled by W1 (ahead of his feet), fast pass to L1 at 3.6 s, L1 dribbles, long lofted
    # pass at 8.4 s towards W2 area, passing behind W3, then ground pass at 11.5 s
    def feet(name, k, dx=7, dy=-2):
        p = actors[name][2][k]
        return np.array([p[0] + dx, p[1] + dy])

    t = np.arange(N) / FPS
    ball = np.zeros((N, 2))
    height = np.zeros(N)
    segs = [(0.0, 3.6, ("drib", "W1")), (3.6, 4.6, ("pass", "W1", "L1")), (4.6, 8.4, ("drib", "L1")),
            (8.4, 10.0, ("loft", "L1", "W2")), (10.0, 11.5, ("drib", "W2")), (11.5, 12.4, ("pass", "W2", "W4")),
            (12.4, 14.1, ("drib", "W4"))]
    for k in range(N):
        for a, b, s in segs:
            if a <= t[k] < b:
                if s[0] == "drib":
                    wob = 4 * np.sin(2 * np.pi * 1.3 * t[k])
                    ball[k] = feet(s[1], k, 7 + wob, -2)
                else:
                    ka, kb = int(round(a * FPS)), int(round(b * FPS))
                    pa, pb = feet(s[1], ka), feet(s[2], min(kb, N - 1), -6)
                    w = (k - ka) / max(1, kb - ka)
                    ball[k] = (1 - w) * pa + w * pb
                    if s[0] == "loft":
                        height[k] = np.sin(np.pi * w)
                        ball[k][1] -= 55 * height[k]  # in the image a lofted ball rises
                break

    rng = np.random.default_rng(1 if seed is None else seed + 100)
    cam = lerp_path([(0, 20, 30), (5, 90, 40), (9, 140, 20), (14, 60, 50)])  # top-left of view in world
    tmp = tempfile.mkdtemp()
    raw = os.path.join(tmp, "raw.avi")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (OW, OH))
    gt = []
    for k in range(N):
        cx, cy = cam[k]
        canvas = world.copy()
        # draw back-to-front (by foot y)
        order = sorted(actors.items(), key=lambda kv: kv[1][2][k][1])
        boxes = {}
        ball_drawn = False
        for name, (team, pool, path) in order:
            # ball is drawn before players whose feet are lower (in front of it)
            if not ball_drawn and path[k][1] > ball[k][1] + 2 + 55 * height[k]:
                _draw_ball(canvas, ball_rgb, ball_a, ball, height, k)
                ball_drawn = True
            spr = (sw if team == "white" else sl if team == "lime" else sr)[pool[((k + 2 * phase[name]) // 4) % 2]]
            flip = path[min(k + 1, N - 1)][0] < path[max(k - 1, 0)][0]
            rgb, a = spr
            if flip:
                rgb, a = rgb[:, ::-1], a[:, ::-1]
            bb = paste(canvas, rgb, a, path[k][0], path[k][1], S)
            boxes[name] = bb
        if not ball_drawn:
            _draw_ball(canvas, ball_rgb, ball_a, ball, height, k)
        x0 = int(np.clip(round(cx), 0, world.shape[1] - OW))
        y0 = int(np.clip(round(cy), 0, world.shape[0] - OH))
        view = canvas[y0:y0 + OH, x0:x0 + OW]
        view = np.clip(view + rng.normal(0, 2.0, view.shape), 0, 255).astype(np.uint8)
        vw.write(view)
        gt.append({"players": {n: ([int(b[0] - x0), int(b[1] - y0), int(b[2]), int(b[3])] if b else None) for n, b in boxes.items()},
                   "ball": [float(ball[k][0] - x0), float(ball[k][1] - y0)]})
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-crf", "26",
                    "-pix_fmt", "yuv420p", out_path], check=True)
    json.dump(gt, open(gt_path, "w"))
    print("wrote", out_path, N, "frames")


def _draw_ball(canvas, rgb, a, ball, height, k):
    v = ball[k] - ball[max(k - 1, 0)]
    s = 1.15 * (1 + 0.35 * height[k])
    n = int(min(8, max(1, np.hypot(*v) / 3)))
    if n == 1:
        paste(canvas, rgb, a, ball[k][0], ball[k][1], s, anchor="center")
        return
    # motion blur: average shifted copies along the motion vector
    acc = np.zeros_like(canvas)
    for i in range(n):
        c = canvas.copy()
        p = ball[k] - v * (i / n)
        paste(c, rgb, a, p[0], p[1], s, anchor="center")
        acc += c
    canvas[:] = acc / n


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
