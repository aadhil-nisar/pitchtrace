"""Score the tracker on the semi-real stress clip (see make_semireal_clip.py).

For each player target, every frame is classed as
    correct  - the box centre is nearest to the right player (and close to it)
    switched - the box is on a DIFFERENT player (identity switch, the "box jumps" bug)
    off      - the box is on nobody
For the ball: success = centre error < 12 px, plus the longest run of failed frames.

    python tools/evaluate_semireal.py clip.mp4 gt.json
"""
import csv
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker import pipeline  # noqa: E402
from tracker.video_io import normalise  # noqa: E402

TARGETS = ["W1", "W2", "L1", "W3"]


def ctr(b):
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])


def main(video, gt_path, step=1):
    gt = json.load(open(gt_path))
    jd = tempfile.mkdtemp()
    meta = normalise(video, os.path.join(jd, "input.mp4"))
    targets = []
    for i, n in enumerate(TARGETS):
        targets.append({"id": n, "name": n, "kind": "player", "color": "#ffd60a", "box": gt[0]["players"][n]})
    bx, by = gt[0]["ball"]
    targets.append({"id": "ball", "name": "Ball", "kind": "ball", "color": "#ff3b30",
                    "box": [int(bx - 6), int(by - 6), 12, 12]})
    pipeline.run(jd, os.path.join(jd, "input.mp4"), meta,
                 {"start": 0, "end": meta["frames"] - 1, "step": step, "enhance": True}, targets,
                 lambda *a, **k: None)
    rows = list(csv.DictReader(open(os.path.join(jd, "output", "tracks.csv"))))
    out = {}
    print(f"{'target':6} {'correct':>8} {'switched':>9} {'off':>6}  first switch")
    for n in TARGETS:
        c = s = o = 0
        first = None
        for r in rows:
            if r["target_id"] != n:
                continue
            k = int(r["frame"])
            b = [float(r[q]) for q in "xywh"]
            p = ctr(b)
            best, bd = None, 1e9
            for m, g in gt[k]["players"].items():
                if g is None:
                    continue
                d = np.linalg.norm(p - ctr(g))
                if d < bd:
                    best, bd = m, d
            h = gt[k]["players"][n][3]
            if best == n and bd < 0.6 * h:
                c += 1
            elif best != n and bd < 0.6 * h and np.linalg.norm(p - ctr(gt[k]["players"][n])) > 0.6 * h:
                s += 1
                first = k if first is None else first
            else:
                o += 1
        tot = c + s + o
        out[n] = {"correct": c / tot, "switched": s / tot}
        print(f"{n:6} {100 * c / tot:7.1f}% {100 * s / tot:8.1f}% {100 * o / tot:5.1f}%  {first}")
    errs, ks, sts = [], [], []
    for r in rows:
        if r["target_id"] != "ball":
            continue
        k = int(r["frame"])
        b = [float(r[q]) for q in "xywh"]
        errs.append(np.linalg.norm(ctr(b) - np.array(gt[k]["ball"])))
        ks.append(k)
        sts.append(r["status"])
    e = np.array(errs)
    ok = e < 12
    run = best_run = 0
    for x in ok:
        run = 0 if x else run + 1
        best_run = max(best_run, run)
    fail_at = next((ks[i] for i in range(len(ok)) if not ok[i]), None)
    print(f"ball   success {100 * ok.mean():.1f}%  median err {np.median(e):.1f}px  longest failure {best_run / 25 * step:.2f}s"
          f"  first fail frame {fail_at}  lost-status {100 * np.mean([s_ == 'lost' for s_ in sts]):.1f}%")
    if os.environ.get("KEEP"):
        print("output kept in", jd)
    else:
        shutil.rmtree(jd, ignore_errors=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
