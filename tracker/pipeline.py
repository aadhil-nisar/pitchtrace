"""End-to-end job: enhancement -> segmentation -> tracking -> post-processing -> outputs."""
import csv
import json
import os
import time

import cv2
import numpy as np
from scipy.signal import savgol_filter

from . import analysis, ballpath
from .camera_motion import CameraMotion, apply, scale_matrix
from .shots import CutDetector, load_overlay
from .enhancement import Enhancer, quality_stats
from .render import RouteMap, draw_route, draw_target, hex_to_bgr
from .segmentation import PitchSegmenter
from .tracking import LOST, PREDICTED, REDETECTED, TRACKED, MultiTracker
from .video_io import H264Writer, iter_frames, read_frame

PROC_WIDTH = 960
PLAYER_HEIGHT_M = 1.8


class Cancelled(Exception):
    pass


def _odd(n, lo=5):
    n = max(lo, int(n))
    return n if n % 2 else n + 1


def run(job_dir, video, meta, settings, targets, progress, cancelled=lambda: False):
    out = os.path.join(job_dir, "output")
    for d in ("", "stages", "charts", "samples", "spot"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    W, H, fps = meta["width"], meta["height"], meta["fps"]
    start = int(settings.get("start", 0))
    end = int(min(settings.get("end", meta["frames"] - 1), meta["frames"] - 1))
    step = int(max(1, settings.get("step", 1)))
    use_enh = bool(settings.get("enhance", True))
    trail = settings.get("trail", "full")
    fps_eff = fps / step

    pw = min(PROC_WIDTH, W)
    s = W / float(pw)
    ph = int(round(H / s))

    def proc(frame):
        return frame if pw == W else cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)

    for t in targets:
        t["bgr"] = hex_to_bgr(t["color"])

    # ------------------------------------------------------------ 1. models from the start frame
    progress("Analysing the first frame", 2)
    first = read_frame(video, start)
    if first is None:
        raise RuntimeError("Could not read the selected start frame.")
    first_p = proc(first)
    enhancer = Enhancer(first_p, enabled=use_enh)
    first_e, enh_stages = enhancer.enhance(first_p, stages=True)
    seg = PitchSegmenter(first_e)
    masks0 = seg.masks(first_e)
    players0 = seg.detect_players(masks0["fg"], ph, pw)
    targets_px = [(t["name"], [int(round(v / s)) for v in t["box"]]) for t in targets]
    stage_items = analysis.save_stage_images(os.path.join(out, "stages"), first_p, enh_stages, first_e,
                                             seg, masks0, players0, targets_px)

    # ------------------------------------------------------------ 2. enhancement evaluation + sample frames
    progress("Measuring enhancement quality", 5)
    seg_orig = PitchSegmenter(first_p)
    sample_idx = np.unique(np.linspace(start, end, 8).astype(int))
    q_o, q_e, d_o, d_e, samples = [], [], [], [], []
    for i, fi in enumerate(sample_idx):
        f = read_frame(video, fi)
        if f is None:
            continue
        fp = proc(f)
        fe = enhancer.enhance(fp)
        q_o.append(quality_stats(fp))
        q_e.append(quality_stats(fe))
        d_o.append(len(seg_orig.detect_players(seg_orig.masks(fp)["fg"], ph, pw)))
        d_e.append(len(seg.detect_players(seg.masks(fe)["fg"], ph, pw)))
        name = f"frame_{fi:06d}.jpg"
        cv2.imwrite(os.path.join(out, "samples", name), fp, [cv2.IMWRITE_JPEG_QUALITY, 88])
        samples.append({"file": name, "frame": int(fi), "time": round(fi / fps, 2)})

    def mean_stats(lst):
        return {k: round(float(np.mean([q[k] for q in lst])), 3) for k in lst[0]} if lst else {}

    enh_eval = {"original": mean_stats(q_o), "enhanced": mean_stats(q_e),
                "detections_original": round(float(np.mean(d_o)), 2) if d_o else 0,
                "detections_enhanced": round(float(np.mean(d_e)), 2) if d_e else 0,
                "frames": len(q_o)}
    if q_o:
        analysis.chart_enhancement(os.path.join(out, "charts", "enhancement.png"), enh_eval)

    # ------------------------------------------------------------ 3. tracking pass
    n_total = len(range(start, end + 1, step))
    cam = CameraMotion()
    cuts = CutDetector()
    overlay = load_overlay(os.path.join(job_dir, "overlay.png"), pw, ph)
    if overlay is not None:
        cam.static_mask = cv2.bitwise_not(overlay)
    refresh_every = max(1, int(2 * fps_eff))
    mt = None
    Ms, frames_idx, shots = [], [], []
    shot = 0
    cut_frames = []
    res = {t["id"]: [] for t in targets}
    times = {"enhance": [], "segment": [], "camera": [], "track": []}
    has_ball = any(t["kind"] == "ball" for t in targets)
    ball_cands, ball_d = [], []  # per frame, for the whole-clip ball path (ballpath.py)
    t_start = time.time()

    for k, (fi, frame) in enumerate(iter_frames(video, start, end, step)):
        if cancelled():
            raise Cancelled()
        t0 = time.perf_counter()
        fp = proc(frame)
        is_cut = cuts.update(fp) and k > 0
        if is_cut:
            # new camera shot (replay, close-up, other camera): routes are split here,
            # camera-motion chaining restarts and every target must be re-identified
            shot += 1
            cut_frames.append(fi)
            cam.prev = None
            if mt:
                for tr in mt.trackers:
                    tr.reset_after_cut()
        fe = enhancer.enhance(fp)
        if k > 0 and (k % refresh_every == 0 or is_cut):
            seg.refresh(fe)  # follow lighting / shadow changes as the camera pans
        t1 = time.perf_counter()
        m = seg.masks(fe)
        if overlay is not None:
            m["fg"] = cv2.bitwise_and(m["fg"], cv2.bitwise_not(overlay))
        players = seg.detect_players(m["fg"], ph, pw)
        nongrass = cv2.bitwise_not(m["grass"])
        balls = []
        if has_ball:
            hint = mt.ball_size_hint() if mt else max(max(b[2], b[3]) for n, b in targets_px)
            balls = seg.detect_ball_candidates(m["fg_raw"], m["field"], hint)
        t2 = time.perf_counter()
        gray = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY)
        M = cam.update(gray, mt.boxes() if mt else [b for n, b in targets_px]).copy()
        t3 = time.perf_counter()
        if mt is None:
            mt = MultiTracker([dict(t, box=b) for t, (n, b) in zip(targets, targets_px)], fe, m["hsv"], nongrass, M, fps_eff)
            for t, (n, b) in zip(targets, targets_px):
                res[t["id"]].append({"box": b, "status": TRACKED, "conf": 1.0})
        else:
            r = mt.step(fe, m["hsv"], nongrass, M, players, balls, m["field"])
            for t in targets:
                res[t["id"]].append(r[t["id"]])
        if has_ball:
            btr = next(tr for tr in mt.trackers if tr.kind == "ball")
            d_now = btr.ball_d * btr.zoom
            ball_cands.append(ballpath.collect(seg, fe, m, players, M, btr._scaled_template(), d_now))
            ball_d.append(d_now)
        t4 = time.perf_counter()
        times["enhance"].append((t1 - t0) * 1000)
        times["segment"].append((t2 - t1) * 1000)
        times["camera"].append((t3 - t2) * 1000)
        times["track"].append((t4 - t3) * 1000)
        Ms.append(M)
        frames_idx.append(fi)
        shots.append(shot)
        if k % 5 == 0:
            el = time.time() - t_start
            eta = el / (k + 1) * (n_total - k - 1)
            progress(f"Tracking frame {k + 1} of {n_total}", 8 + 62 * (k + 1) / n_total, eta)

    K = len(frames_idx)
    if K == 0:
        raise RuntimeError("No frames could be read in the selected range.")
    if has_ball:
        # replace the frame-by-frame ball answers by the most plausible whole-clip path
        progress("Finding the ball path", 70)
        for t, (n, b0) in zip(targets, targets_px):
            if t["kind"] != "ball":
                continue
            c0 = (b0[0] + b0[2] / 2.0, b0[1] + b0[3] / 2.0)
            start_ref = apply(np.linalg.inv(Ms[0]), c0)[0]
            if os.environ.get("BALLPATH_DUMP"):  # debugging aid
                import pickle
                with open(os.environ["BALLPATH_DUMP"], "wb") as fh:
                    pickle.dump({"cands": ball_cands, "d": ball_d, "shots": shots, "start": start_ref,
                                 "fps": fps_eff, "Ms": Ms}, fh)
            path = ballpath.solve(ball_cands, ball_d, shots, start_ref, fps_eff)
            R = res[t["id"]]
            for k in range(K):
                if path[k] is None:
                    R[k] = {"box": R[k]["box"], "status": PREDICTED, "conf": 0.0}
                    continue
                x, y = apply(Ms[k], path[k][:2])[0]
                dd = ball_d[k]
                R[k] = {"box": [x - dd / 2, y - dd / 2, dd, dd], "status": TRACKED,
                        "conf": round(path[k][2], 3)}
    track_time = time.time() - t_start

    # ------------------------------------------------------------ 4. post-processing
    progress("Smoothing routes", 71)
    Ms_o = [scale_matrix(M, s) for M in Ms]
    max_gap = int(3 * fps_eff)
    tracks = []
    all_heights = []
    for t in targets:
        R = res[t["id"]]
        st = [r["status"] for r in R]
        conf = [r["conf"] for r in R]
        boxes = np.array([r["box"] for r in R], np.float64) * s
        if t["kind"] == "ball":
            pts = boxes[:, :2] + boxes[:, 2:] / 2
        else:
            pts = np.stack([boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3]], 1)
        ref = np.array([apply(np.linalg.inv(Ms_o[k]), pts[k])[0] for k in range(K)])
        sizes = boxes[:, 2:].copy()
        meas = np.array([x in (TRACKED, REDETECTED) for x in st])
        if t["kind"] != "ball":
            all_heights += list(sizes[meas, 1])
        valid = meas.copy()
        mi = np.where(meas)[0]
        # the ball's points come from the whole-clip path, so every gap inside one camera
        # shot is filled: the ball then has a position on every frame
        gap_lim = K if t["kind"] == "ball" else max_gap
        for a, b in zip(mi[:-1], mi[1:]):
            if 1 < b - a <= gap_lim and shots[a] == shots[b]:
                for k in range(a + 1, b):
                    w = (k - a) / (b - a)
                    ref[k] = (1 - w) * ref[a] + w * ref[b]
                    sizes[k] = (1 - w) * sizes[a] + w * sizes[b]
                    valid[k] = True
        # smoothing on contiguous valid runs (Savitzky-Golay, quadratic)
        sm = ref.copy()
        win = _odd(fps_eff * (0.2 if t["kind"] == "ball" else 0.4))
        k = 0
        while k < K:
            if not valid[k]:
                k += 1
                continue
            j = k
            while j < K and valid[j] and shots[j] == shots[k]:
                j += 1
            n = j - k
            if n >= 5:
                w_ = min(win, n if n % 2 else n - 1)
                if w_ >= 5:
                    sm[k:j, 0] = savgol_filter(ref[k:j, 0], w_, 2)
                    sm[k:j, 1] = savgol_filter(ref[k:j, 1], w_, 2)
            k = j
        # gentle size smoothing
        ks = _odd(fps_eff * 0.3, 3)
        sizes_s = sizes.copy()
        for c in range(2):
            sizes_s[:, c] = np.convolve(np.pad(sizes[:, c], ks // 2, mode="edge"), np.ones(ks) / ks, mode="valid")
        tracks.append({"t": t, "status": st, "conf": conf, "raw_boxes": boxes, "ref": sm, "valid": valid,
                       "sizes": sizes_s, "meas": meas})

    if all_heights:
        med_h = float(np.median(all_heights))
    elif players0:
        med_h = float(np.median([b[3] for b in players0])) * s
    else:
        med_h = H * 0.08
    m_per_px = PLAYER_HEIGHT_M / max(med_h, 1)

    # ------------------------------------------------------------ 5. render pass
    progress("Drawing routes on the video", 73)
    writer = H264Writer(os.path.join(out, "tracked.mp4"), W, H, round(fps_eff, 3))
    main_shot = shots[0]
    rmap = RouteMap([M for M, sh in zip(Ms_o, shots) if sh == main_shot], W, H)
    pano_every = max(1, K // 40)
    spot_k = set(np.unique(np.linspace(0, K - 1, min(12, K)).astype(int)).tolist())
    spot, prog_frames = [], []
    trail_n = int(fps_eff * 3) if trail == "recent" else None

    def display_box(tr, k, M):
        if tr["valid"][k]:
            p = apply(M, tr["ref"][k])[0]
            w, h = tr["sizes"][k]
            if tr["t"]["kind"] == "ball":
                return [p[0] - w / 2, p[1] - h / 2, w, h]
            return [p[0] - w / 2, p[1] - h, w, h]
        return list(tr["raw_boxes"][k])

    for k, (fi, frame) in enumerate(iter_frames(video, start, end, step)):
        if k >= K:
            break
        if cancelled():
            writer.close()
            raise Cancelled()
        M = Ms_o[k]
        if (k % pano_every == 0 or k == K - 1) and shots[k] == main_shot:
            rmap.add_frame(frame, M)
        vis = frame.copy()
        for tr in tracks:
            lo = 0 if trail_n is None else max(0, k - trail_n)
            idx = [j for j in range(lo, k + 1) if tr["valid"][j] and shots[j] == shots[k]]
            # break the route where tracking was lost for long
            segs, cur = [], []
            for j in idx:
                if cur and j - cur[-1] > 1:
                    segs.append(cur)
                    cur = []
                cur.append(j)
            if cur:
                segs.append(cur)
            for sg in segs:
                if len(sg) > 1:
                    draw_route(vis, apply(M, tr["ref"][sg]), tr["t"]["bgr"], 3)
        for tr in tracks:
            stt = tr["status"][k]
            if stt == LOST and not tr["valid"][k]:
                continue
            shown = stt if not tr["valid"][k] or stt != PREDICTED else PREDICTED
            draw_target(vis, display_box(tr, k, M), tr["t"]["bgr"], tr["t"]["name"], shown, tr["t"]["kind"])
        writer.write(vis)
        if k in spot_k:
            small = cv2.resize(vis, (720, int(H * 720 / W)), interpolation=cv2.INTER_AREA)
            name = f"spot_{len(spot):02d}.jpg"
            cv2.imwrite(os.path.join(out, "spot", name), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            spot.append({"file": name, "frame": int(fi), "time": round(fi / fps, 2),
                         "targets": {tr["t"]["id"]: tr["status"][k] for tr in tracks}})
            prog_frames.append(analysis.titled(small, f"t = {fi / fps:.1f} s  (frame {fi})"))
        if k % 10 == 0:
            progress(f"Drawing frame {k + 1} of {K}", 73 + 20 * (k + 1) / K)
    writer.close()

    # ------------------------------------------------------------ 6. outputs
    progress("Building route map and report", 94)
    routes = []
    for tr in tracks:
        v = tr["valid"]
        v = v & (np.array(shots) == main_shot)
        routes.append((tr["t"]["name"], tr["t"]["bgr"], tr["ref"][v] if v.any() else None))
    cv2.imwrite(os.path.join(out, "route_map.jpg"), rmap.render(routes), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if prog_frames:
        sel = prog_frames[:: max(1, len(prog_frames) // 6)][:6]
        cv2.imwrite(os.path.join(out, "stages", "15_tracking_progression.jpg"), analysis.grid(sel, 3, 480),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        stage_items.append({"file": "15_tracking_progression.jpg", "caption": "Frame sequence showing tracking progression"})

    frame_times = [fi / fps for fi in frames_idx]
    tmetrics, chart_targets = [], []
    for tr in tracks:
        st = tr["status"]
        n = len(st)
        cnt = {k_: st.count(k_) for k_ in (TRACKED, REDETECTED, PREDICTED, LOST)}
        gaps, g = [], 0
        for x in tr["meas"]:
            if x:
                if g:
                    gaps.append(g)
                g = 0
            else:
                g += 1
        if g:
            gaps.append(g)
        v = tr["valid"]
        ref = tr["ref"]
        step_d = np.zeros(n)
        for k_ in range(1, n):
            if v[k_] and v[k_ - 1] and shots[k_] == shots[k_ - 1]:
                step_d[k_] = np.linalg.norm(ref[k_] - ref[k_ - 1])
        dist_px = float(step_d.sum())
        win = max(1, int(round(fps_eff)))
        sp = np.convolve(step_d, np.ones(win) / win, mode="same") * fps_eff * m_per_px * 3.6
        speed = [float(x) if v[i] else None for i, x in enumerate(sp)]
        vs = [x for x in speed if x is not None]
        redet = sum(1 for k_ in range(1, n) if st[k_] == REDETECTED)
        meas_conf = [c for c, x in zip(tr["conf"], tr["meas"]) if x]
        tm = {
            "id": tr["t"]["id"], "name": tr["t"]["name"], "kind": tr["t"]["kind"], "color": tr["t"]["color"],
            "frames": n,
            "tracked_pct": round(100 * cnt[TRACKED] / n, 1),
            "redetected_pct": round(100 * cnt[REDETECTED] / n, 1),
            "predicted_pct": round(100 * cnt[PREDICTED] / n, 1),
            "lost_pct": round(100 * cnt[LOST] / n, 1),
            "consistency_pct": round(100 * (cnt[TRACKED] + cnt[REDETECTED]) / n, 1),
            "route_coverage_pct": round(100 * v.sum() / n, 1),
            "redetections": redet,
            "longest_gap_s": round(max(gaps) / fps_eff, 2) if gaps else 0.0,
            "mean_confidence": round(float(np.mean(meas_conf)), 3) if meas_conf else 0.0,
            "distance_px": round(dist_px, 1),
            "distance_m": round(dist_px * m_per_px, 1),
            "avg_speed_kmh": round(float(np.mean(vs)), 1) if vs else None,
            "max_speed_kmh": round(float(np.max(vs)), 1) if vs else None,
        }
        tmetrics.append(tm)
        chart_targets.append({"name": tm["name"], "color": tm["color"], "status": st, "conf": tr["conf"],
                              "speed_kmh": [x if x is not None else np.nan for x in speed]})

    ch = os.path.join(out, "charts")
    analysis.chart_processing_time(os.path.join(ch, "processing_time.png"), times)
    analysis.chart_status(os.path.join(ch, "tracking_status.png"), chart_targets, frame_times)
    analysis.chart_confidence(os.path.join(ch, "confidence.png"), chart_targets, frame_times)
    analysis.chart_speed(os.path.join(ch, "speed.png"), chart_targets, frame_times)
    analysis.flowchart(os.path.join(ch, "pipeline_flowchart.png"))

    with open(os.path.join(out, "tracks.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "target_id", "name", "kind", "status", "x", "y", "w", "h",
                     "point_x", "point_y", "route_x", "route_y", "on_route", "confidence"])
        for tr in tracks:
            for k_ in range(K):
                M = Ms_o[k_]
                b = display_box(tr, k_, M)
                p = apply(M, tr["ref"][k_])[0]
                wr.writerow([frames_idx[k_], round(frames_idx[k_] / fps, 3), tr["t"]["id"], tr["t"]["name"],
                             tr["t"]["kind"], tr["status"][k_], *[round(float(x), 1) for x in b],
                             round(float(p[0]), 1), round(float(p[1]), 1),
                             round(float(tr["ref"][k_][0]), 1), round(float(tr["ref"][k_][1]), 1),
                             int(tr["valid"][k_]), tr["conf"][k_]])

    ms = {k_: float(np.mean(v_)) for k_, v_ in times.items()}
    total_ms = float(np.mean(np.sum([times[k_] for k_ in times], axis=0)))
    metrics = {
        "video": dict(meta),
        "settings": {"start_frame": start, "end_frame": frames_idx[-1], "frame_step": step,
                     "enhancement": use_enh, "trail": trail, "processing_width": pw, "output_fps": round(fps_eff, 3)},
        "enhancement": enhancer.params(),
        "segmentation": seg.params(),
        "enhancement_eval": enh_eval,
        "processing": {
            "frames": K,
            "tracking_time_s": round(track_time, 2),
            "avg_ms_per_frame": round(total_ms, 2),
            "processing_fps": round(1000 / total_ms, 2) if total_ms else None,
            "stage_ms": {k_: round(v_, 2) for k_, v_ in ms.items()},
            "players_detected_first_frame": len(players0),
            "camera_cuts": cut_frames,
            "shots": shot + 1,
            "overlay_masked": overlay is not None,
        },
        "scale": {"metres_per_pixel": round(m_per_px, 5), "median_player_height_px": round(med_h, 1),
                  "note": "Approximate: assumes 1.8 m player height; perspective makes far players smaller."},
        "targets": tmetrics,
        "stages": stage_items,
        "samples": samples,
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out, "spot.json"), "w") as f:
        json.dump(spot, f, indent=2)
    progress("Done", 100)
    return metrics
