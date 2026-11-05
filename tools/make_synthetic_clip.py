"""Generate a synthetic broadcast-style football clip with known ground truth.

Used only to test the pipeline end-to-end (camera pan + zoom, two teams, crossings,
a ball, noise and blur). Real experiments must use the YouTube matches.
"""
import json
import sys

import cv2
import numpy as np

rng = np.random.default_rng(3)
OW, OH = 1280, 720
FPS, SECONDS = 25, 8
N = FPS * SECONDS
WW, WH = 3000, 1500  # world size


def world():
    img = np.zeros((WH, WW, 3), np.uint8)
    for i in range(0, WW, 150):
        img[:, i:i + 150] = (40, 140, 50) if (i // 150) % 2 else (50, 160, 60)
    img = cv2.add(img, rng.integers(0, 18, img.shape, dtype=np.uint8))
    stands = rng.integers(30, 230, (60, WW // 10, 3), dtype=np.uint8)
    img[:300] = cv2.resize(stands, (WW, 300), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(img, (0, 300), (WW, 340), (30, 30, 200), -1)  # ad boards
    for x in range(0, WW, 300):
        cv2.putText(img, "SPONSOR", (x + 40, 332), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    w = (245, 245, 245)
    cv2.rectangle(img, (100, 380), (WW - 100, WH - 40), w, 4)
    cv2.line(img, (WW // 2, 380), (WW // 2, WH - 40), w, 4)
    cv2.circle(img, (WW // 2, 920), 180, w, 4)
    cv2.rectangle(img, (100, 650), (420, 1190), w, 4)
    cv2.rectangle(img, (WW - 420, 650), (WW - 100, 1190), w, 4)
    return cv2.GaussianBlur(img, (3, 3), 0)


def path(seed):
    r = np.random.default_rng(seed)
    p0 = np.array([r.uniform(700, 2300), r.uniform(520, 1350)])
    ph = r.uniform(0, 6.28, 4)
    a = r.uniform(80, 260, 2)
    t = np.arange(N) / FPS
    x = p0[0] + a[0] * np.sin(0.35 * t + ph[0]) + 90 * t * np.cos(ph[2])
    y = p0[1] + a[1] * np.sin(0.5 * t + ph[1]) * 0.6
    return np.stack([x, y], 1)


def draw_player(img, x, y, shirt, shorts, scale):
    h = int(70 * scale); wdt = int(22 * scale)
    x, y = int(x), int(y)
    cv2.rectangle(img, (x - wdt // 3, y - h // 3), (x - 2, y), (60, 60, 60), -1)
    cv2.rectangle(img, (x + 2, y - h // 3), (x + wdt // 3, y), (60, 60, 60), -1)
    cv2.rectangle(img, (x - wdt // 2, y - h // 2), (x + wdt // 2, y - h // 3), shorts, -1)
    cv2.rectangle(img, (x - wdt // 2, y - int(h * 0.85)), (x + wdt // 2, y - h // 2), shirt, -1)
    cv2.circle(img, (x, y - int(h * 0.92)), max(3, int(7 * scale)), (140, 170, 220), -1)


def main(out_path, gt_path):
    base = world()
    players = []
    for i in range(10):
        team = i % 2
        players.append({"path": path(10 + i), "shirt": (40, 40, 210) if team == 0 else (210, 120, 30),
                        "shorts": (255, 255, 255) if team == 0 else (20, 20, 20)})
    # make players 0 and 1 cross in the middle of the clip
    t = np.arange(N)
    players[0]["path"] = np.stack([1100 + 2.6 * t * 1.6, 900 + 40 * np.sin(t / 20)], 1)
    players[1]["path"] = np.stack([1900 - 2.0 * t * 1.6, 920 + 30 * np.cos(t / 25)], 1)
    ball = np.stack([1200 + 3.2 * t * 1.5 + 60 * np.sin(t / 9), 960 + 120 * np.sin(t / 30)], 1)
    cam_c = np.stack([1250 + 4.0 * t, 820 + 40 * np.sin(t / 50)], 1)
    zoom = 1.0 + 0.12 * np.sin(t / 60)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OW, OH))
    gt = []
    for k in range(N):
        img = base.copy()
        for p in sorted(players, key=lambda p: p["path"][k][1]):
            x, y = p["path"][k]
            draw_player(img, x, y, p["shirt"], p["shorts"], 0.8 + 0.4 * (y - 400) / 1100)
        bx, by = ball[k]
        cv2.circle(img, (int(bx), int(by)), 7, (250, 250, 250), -1)
        vw_, vh_ = OW * 1.25 / zoom[k], OH * 1.25 / zoom[k]
        cx, cy = cam_c[k]
        x0, y0 = cx - vw_ / 2, cy - vh_ / 2
        sc = OW / vw_
        A = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc]])
        frame = cv2.warpAffine(img, A, (OW, OH))
        frame = cv2.add(frame, rng.integers(0, 10, frame.shape, dtype=np.uint8))
        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=-5)  # slightly dark, low contrast
        vw.write(frame)
        rec = {"frame": k, "players": [], "ball": None}
        for i, p in enumerate(players):
            x, y = p["path"][k]
            scl = 0.8 + 0.4 * (y - 400) / 1100
            h, w_ = 70 * scl, 22 * scl
            fx, fy = (x - x0) * sc, (y - y0) * sc
            rec["players"].append([fx - w_ * sc / 2, fy - h * sc, w_ * sc, h * sc])
        rec["ball"] = [(bx - x0) * sc, (by - y0) * sc]
        gt.append(rec)
    vw.release()
    json.dump(gt, open(gt_path, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
