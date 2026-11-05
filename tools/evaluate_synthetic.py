"""Accuracy check against ground truth on the synthetic clip.

    python tools/make_synthetic_clip.py synth.mp4 gt.json
    python tools/evaluate_synthetic.py synth.mp4 gt.json

Prints centre error and success rate per target (a known-answer sanity test of the
tracker; real-match accuracy is measured with the website's "Check accuracy" tab).
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


def main(video, gt_path):
    gt = json.load(open(gt_path))
    jd = tempfile.mkdtemp()
    meta = normalise(video, os.path.join(jd, "input.mp4"))
    p = gt[0]["players"]
    b = gt[0]["ball"]
    targets = [
        {"id": "t1", "name": "Red 1", "kind": "player", "color": "#ffd60a", "box": [int(v) for v in p[0]], "gt": 0},
        {"id": "t2", "name": "Blue 2", "kind": "player", "color": "#00c7ff", "box": [int(v) for v in p[1]], "gt": 1},
        {"id": "t3", "name": "Red 5", "kind": "player", "color": "#ff2dd1", "box": [int(v) for v in p[4]], "gt": 4},
        {"id": "t4", "name": "Ball", "kind": "ball", "color": "#ff3b30", "box": [int(b[0] - 8), int(b[1] - 8), 16, 16]},
    ]
    pipeline.run(jd, os.path.join(jd, "input.mp4"), meta,
                 {"start": 0, "end": meta["frames"] - 1, "step": 1, "enhance": True}, targets,
                 lambda *a, **k: None)
    rows = list(csv.DictReader(open(os.path.join(jd, "output", "tracks.csv"))))
    print(f"{'target':8} {'median err px':>14} {'90th pct':>9} {'success (<20px)':>16}")
    for t in targets:
        errs = []
        for r in rows:
            if r["target_id"] != t["id"]:
                continue
            k = int(r["frame"])
            x, y, w, h = (float(r[c]) for c in "xywh")
            c = np.array([x + w / 2, y + h / 2])
            if t["kind"] == "ball":
                g = np.array(gt[k]["ball"])
            else:
                gb = gt[k]["players"][t["gt"]]
                g = np.array([gb[0] + gb[2] / 2, gb[1] + gb[3] / 2])
            errs.append(np.linalg.norm(c - g))
        e = np.array(errs)
        print(f"{t['name']:8} {np.median(e):14.1f} {np.percentile(e, 90):9.1f} {100 * (e < 20).mean():15.1f}%")
    shutil.rmtree(jd, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
