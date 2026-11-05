"""Whole-clip ball trajectory: collect ball candidates on every frame, then find the most
plausible path through them with dynamic programming (Viterbi).

Why not decide frame by frame? At the moment of a kick the kicker's boot, a white sock
or a white spot on the pitch can look more like the ball than the (blurred, half hidden)
ball itself. A frame-by-frame tracker that picks the wrong one follows it and never
comes back. The whole video is processed before anything is drawn, so the decision can
use the future as well as the past: the true ball is the candidate sequence that is
ball-like on many frames AND moves like a ball (no faster than ~30 m/s) between them.

Model
  * nodes: up to K candidates per frame, each with a ball-likeness score s in [0, 1]
    (blob shape / colour / contrast from PitchSegmenter.detect_ball, and normalised
    cross-correlation with the user's ball template; candidates on a player's upper
    body are down-weighted, ones touching his feet are kept as weak evidence)
  * node cost   1 - s
  * skipping a frame (ball hidden, out of shot, blurred) costs MISS per frame, so a
    candidate is only used if it is more ball-like than "not visible"
  * moving between two candidates g frames apart costs w * (D / (vmax*g + slack))^2,
    with a hard limit D <= vmax*g + slack (positions in camera-compensated coordinates)
  * after a camera cut, or after more than GAP_S seconds without any usable candidate,
    the path may restart anywhere (fixed penalty)
The path starts at the user's ball on the first frame. Frames between path nodes are
filled by the pipeline's gap interpolation, so the ball has a position on every frame
where it can be inferred.
"""
import cv2
import numpy as np

from .camera_motion import apply

K_MAX = 10       # candidates kept per frame
MISS = 0.42      # cost of a frame without the ball on the path
W_MOTION = 0.6   # weight of the motion cost
GAP_S = 3.0      # longest gap bridged with a motion constraint (seconds)
RESTART = 2.5    # cost of restarting the path (camera cut / long loss)


def _ncc_map(gray, tmpl, field):
    th, tw = tmpl.shape
    res = cv2.matchTemplate(gray.astype(np.float32), tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    full = np.full(gray.shape, -1.0, np.float32)
    full[th // 2:th // 2 + res.shape[0], tw // 2:tw // 2 + res.shape[1]] = res
    if field is not None:
        full[field == 0] = -1.0
    return full


def _peaks(score, d, k, min_v):
    s = score.copy()
    out = []
    sup = max(3, int(round(d)))
    for _ in range(k):
        _, mv, _, ml = cv2.minMaxLoc(s)
        if mv < min_v:
            break
        out.append((float(ml[0]), float(ml[1]), float(mv)))
        s[max(0, ml[1] - sup):ml[1] + sup + 1, max(0, ml[0] - sup):ml[0] + sup + 1] = -1
    return out


def collect(seg, fe, m, players, M, tmpl, d):
    """Ball candidates of one (processing-size) frame: [(x_ref, y_ref, score)]."""
    gray = cv2.cvtColor(fe, cv2.COLOR_BGR2GRAY)
    ncc = _ncc_map(gray, tmpl, m["field"]) if tmpl is not None else None
    cands = []
    for b, sc in seg.detect_ball(fe, m, players, size_hint=d, max_out=K_MAX):
        c = (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
        n = 0.0
        if ncc is not None:
            x, y = int(round(c[0])), int(round(c[1]))
            r = max(1, int(d / 2))
            win = ncc[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            n = float(win.max()) if win.size else 0.0
        cands.append((c[0], c[1], 0.55 * sc + 0.45 * max(n, 0.0)))
    if ncc is not None:
        for x, y, v in _peaks(ncc, d, K_MAX, 0.55):
            if any(np.hypot(x - a, y - b) < d for a, b, _ in cands):
                continue
            # template-only evidence (e.g. the ball touching a boot): weaker
            cands.append((x, y, 0.75 * v))
    out = []
    for x, y, s in cands:
        for px, py, pw, ph in players:
            if px - 1 <= x <= px + pw + 1 and py - 1 <= y <= py + 0.75 * ph:
                s *= 0.5  # head / shirt / shorts of a player
                break
        p = apply(np.linalg.inv(M), (x, y))[0]
        out.append((float(p[0]), float(p[1]), float(np.clip(s, 0, 1))))
    out.sort(key=lambda t: -t[2])
    return out[:K_MAX]


def solve(cands, d_per_frame, shots, start_ref, fps):
    """cands[t] = [(x_ref, y_ref, score)], start_ref = user's ball (ref coords) on frame 0.
    Returns, per frame, None or (x_ref, y_ref, score) of the chosen candidate."""
    T = len(cands)
    G = max(2, int(GAP_S * fps))
    P = [np.array([(c[0], c[1]) for c in cs], np.float64).reshape(-1, 2) for cs in cands]
    S = [np.array([c[2] for c in cs], np.float64) for cs in cands]
    # frame 0: only the user's ball
    P[0] = np.array([start_ref], np.float64)
    S[0] = np.array([1.0])
    best = [None] * T
    back = [None] * T
    best[0] = np.zeros(1)
    back[0] = [(-1, -1)]
    # best restart source so far (cost adjusted for the misses up to "now")
    for t in range(1, T):
        n = len(P[t])
        if n == 0:
            best[t] = np.zeros(0)
            back[t] = []
            continue
        cost = np.full(n, np.inf)
        bk = [(-1, -1)] * n
        node = 1.0 - S[t]
        for g in range(1, min(G, t) + 1):
            tp = t - g
            if len(P[tp]) == 0 or not np.isfinite(best[tp]).any():
                continue
            if shots[tp] != shots[t]:
                continue
            vmax = 30.0 * (max(d_per_frame[t], 2.0) / 0.2 / 1.8) / fps
            lim = vmax * g + 2.0 * d_per_frame[t]
            D = np.linalg.norm(P[tp][:, None, :] - P[t][None, :, :], axis=2)
            mc = np.where(D <= lim, W_MOTION * (D / lim) ** 2, np.inf)
            tot = best[tp][:, None] + MISS * (g - 1) + mc
            i = np.argmin(tot, axis=0)
            v = tot[i, np.arange(n)]
            upd = v < cost
            cost = np.where(upd, v, cost)
            for j in np.where(upd)[0]:
                bk[j] = (tp, int(i[j]))
        # restart: from the best end state of any earlier frame (camera cut or long loss)
        rs, rsrc = np.inf, (-1, -1)
        for tp in range(t - 1, -1, -1):
            if len(P[tp]) == 0 or not np.isfinite(best[tp]).any():
                continue
            if shots[tp] == shots[t] and t - tp <= G:
                continue
            i = int(np.argmin(best[tp]))
            v = best[tp][i] + MISS * (t - tp - 1) + RESTART
            if v < rs:
                rs, rsrc = v, (tp, i)
            if shots[tp] != shots[t]:
                break  # an older restart source is never better than the newest one
        upd = rs < cost
        cost = np.where(upd, rs, cost)
        for j in np.where(upd)[0]:
            bk[j] = rsrc
        best[t] = cost + node
        back[t] = bk
    # end: any frame, plus the misses up to the end of the clip
    end, ev = (0, 0), np.inf
    for t in range(T):
        if len(best[t]) == 0:
            continue
        i = int(np.argmin(best[t]))
        v = best[t][i] + MISS * (T - 1 - t)
        if v < ev:
            end, ev = (t, i), v
    out = [None] * T
    t, i = end
    while t >= 0:
        out[t] = (float(P[t][i][0]), float(P[t][i][1]), float(S[t][i]))
        t, i = back[t][i]
    return out
