"""Hybrid multi-target tracker for football players and the ball.

Each selected target is followed by a combination of four ideas:

  1. Appearance tracker  - OpenCV CSRT (discriminative correlation filter with channel and
                           spatial reliability) gives a box every frame.
  2. Appearance model    - a 2-D Hue-Saturation histogram of the target's kit, computed
                           only on non-grass pixels. Every proposed box is verified with the
                           Bhattacharyya similarity  sim = 1 - d_B(h_model, h_box).
  3. Motion model        - a constant-velocity Kalman filter, run in camera-compensated
                           reference coordinates, predicts where the target should be.
                           It gates implausible jumps and bridges occlusions.
  4. Re-detection        - when CSRT drifts or fails, segmentation blobs near the
                           prediction are scored by appearance + distance + size and the
                           best one re-initialises CSRT.

Close encounters between players (crossings, shoulder to shoulder, one behind another):
  5. Neighbour motion    - every detected player blob gets a light-weight motion track
                           (BlobTracks, Hungarian assignment). When another player comes
                           close, the target enters "crowd mode": its velocity is frozen
                           (limited to ~9 m/s) and a measurement is accepted only if it is
                           clearly closer to the target's own extrapolated path than to every
                           same-kit neighbour's path. While players overlap, the target
                           coasts instead of guessing.

The ball (no CSRT; it is weak on tiny, fast objects):
  * fixed size from the user's (refined) ball box and a grey-level template of the ball,
  * candidates from small round or motion-blurred blobs and from normalised
    cross-correlation peaks, which also find the ball touching a player's boot,
  * matches on or touching a player are "weak" (could be a sock) and never beat a clear,
    isolated ball; candidates far from the prediction must be confirmed on two frames and
    their motion must trace back to where the ball was last seen,
  * the search grows the longer the ball is unseen, up to the whole pitch.

Per-frame status: tracked | redetected | predicted (occluded, Kalman only) | lost.
"""
from collections import deque

import cv2
import numpy as np

from .camera_motion import apply

TRACKED, REDETECTED, PREDICTED, LOST = "tracked", "redetected", "predicted", "lost"


def clip_box(b, W, H):
    x, y, w, h = b
    x = max(0, min(int(round(x)), W - 2))
    y = max(0, min(int(round(y)), H - 2))
    w = max(2, min(int(round(w)), W - x))
    h = max(2, min(int(round(h)), H - y))
    return [x, y, w, h]


def center(b):
    return np.array([b[0] + b[2] / 2.0, b[1] + b[3] / 2.0])


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def kit_hist(hsv, nongrass, box):
    x, y, w, h = box
    roi = hsv[y:y + h, x:x + w]
    m = nongrass[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    # kit pixels only (grass excluded); if the kit is mostly grass-coloured (e.g. a green
    # goalkeeper) fall back to the whole box
    if cv2.countNonZero(m) < max(6, 0.15 * w * h):
        m = None
    hist = cv2.calcHist([roi], [0, 1], m, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0, cv2.NORM_L1)
    return hist


def shirt_colour(lab, nongrass, box):
    """Median CIE-Lab colour of the shirt: non-grass pixels in the upper part of the box
    (15-60 % of its height, below the head). Returned as (L, a, b) with a, b centred on 0.

    The kit histogram alone separates kits poorly when one team wears grass-coloured
    (lime) shirts; the chroma (a, b) of the shirt separates them clearly, and it changes
    little in shadow (shadow mostly lowers L)."""
    x, y, w, h = box
    y1, y2 = y + int(0.15 * h), y + max(int(0.6 * h), int(0.15 * h) + 2)
    roi = lab[y1:y2, x:x + w].reshape(-1, 3)
    m = (nongrass[y1:y2, x:x + w] > 0).ravel()
    if m.sum() < 6:
        return None
    return np.median(roi[m].astype(np.float32), axis=0) - np.array([0, 128, 128], np.float32)


def shirt_distance(c1, c2):
    if c1 is None or c2 is None:
        return 0.0
    d = c1 - c2
    return float(np.sqrt(d[1] ** 2 + d[2] ** 2 + (0.15 * d[0]) ** 2))


SHIRT_TOL = 18.0  # Lab units; lime vs white kits are ~50 apart, the same kit in shadow ~5-15


def similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return float(1.0 - cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def make_kalman(x, y, dt=1.0, q_vel=4.0):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.diag([1, 1, q_vel, q_vel]).astype(np.float32) * 0.5
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 10
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf


def refine_ball_box(frame, nongrass, box):
    """Tighten a hand-drawn ball box onto the ball itself (the non-grass blob nearest
    the box centre), so the size and template describe the ball and not the grass."""
    H, W = frame.shape[:2]
    x, y, w, h = box
    m = nongrass[y:y + h, x:x + w]
    if m.size == 0:
        return box
    n, lab, st, cen = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    c = np.array([w / 2.0, h / 2.0])
    best, bd = None, 1e9
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 3:
            continue
        d = float(np.linalg.norm(cen[i] - c))
        if d < bd:
            best, bd = i, d
    if best is None:
        return box
    bx, by, bw, bh = st[best, :4]
    if max(bw, bh) < 3 or bw * bh > 0.95 * w * h and max(w, h) > 14:
        return box
    side = int(max(bw, bh, 4))
    cx, cy = x + bx + bw / 2.0, y + by + bh / 2.0
    return clip_box([cx - side / 2, cy - side / 2, side, side], W, H)


def ball_template(frame, c, d):
    """Grey-level patch of the ball with a ring of grass around it (the contrast between
    ball and grass is what normalised cross-correlation locks on to)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    r = int(round(max(d, 3) * 0.5 + max(3, 0.6 * d)))
    x, y = int(round(c[0])), int(round(c[1]))
    H, W = g.shape
    if x - r < 0 or y - r < 0 or x + r + 1 > W or y + r + 1 > H:
        return None
    return g[y - r:y + r + 1, x - r:x + r + 1].astype(np.float32)


def ncc_peaks(gray, tmpl, c, radius, k=3, min_score=0.4, mask=None):
    """Normalised cross-correlation of the template in a window of the given radius
    around c. Returns up to k local peaks [(x, y, score)] in frame coordinates."""
    if tmpl is None:
        return []
    th, tw = tmpl.shape
    H, W = gray.shape
    r = int(radius)
    x0, y0 = max(0, int(c[0]) - r - tw // 2), max(0, int(c[1]) - r - th // 2)
    x1, y1 = min(W, int(c[0]) + r + tw // 2 + 1), min(H, int(c[1]) + r + th // 2 + 1)
    if x1 - x0 <= tw or y1 - y0 <= th:
        return []
    res = cv2.matchTemplate(gray[y0:y1, x0:x1].astype(np.float32), tmpl, cv2.TM_CCOEFF_NORMED)
    if mask is not None:
        mm = mask[y0 + th // 2:y0 + th // 2 + res.shape[0], x0 + tw // 2:x0 + tw // 2 + res.shape[1]]
        res = np.where(mm > 0, res, -1)
    out = []
    sup = max(3, tw // 2)
    for _ in range(k):
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv < min_score:
            break
        out.append((x0 + ml[0] + tw / 2.0, y0 + ml[1] + th / 2.0, float(mv)))
        res[max(0, ml[1] - sup):ml[1] + sup + 1, max(0, ml[0] - sup):ml[0] + sup + 1] = -1
    return out


class BlobTracks:
    """Light-weight motion tracks for EVERY detected player blob (not only the targets).

    A target tracker only knows where it expects its own player to be. When two players
    in the same kit meet, the measurement in between them fits both, and the target can
    slide onto the other player. Knowing where each *neighbour* is expected to be lets
    the target accept only measurements that are clearly closer to its own path.

    Tracks live in camera-compensated reference coordinates. Blobs are matched to
    predicted tracks with the Hungarian algorithm (scipy linear_sum_assignment). A blob
    whose size does not fit the track (two players merged into one blob, or a player
    half hidden) is NOT used to update it, so both players coast with the velocity they
    had before they met.
    """

    def __init__(self, fps):
        self.fps = fps
        self.tracks = []
        self.k = 0
        self._next = 1

    def update(self, players, Minv):
        from scipy.optimize import linear_sum_assignment
        self.k += 1
        for t in self.tracks:
            t["p"] = t["p"] + t["v"]
            t["miss"] += 1
            t["matched"] = None
            t["run"] = 0 if t["miss"] > 1 else t.get("run", 0)
        C = [apply(Minv, center(b))[0] for b in players]
        if self.tracks and players:
            cost = np.full((len(self.tracks), len(players)), 1e6)
            for i, t in enumerate(self.tracks):
                gate = max(6.0, 0.45 * t["size"][1]) * (1 + 0.25 * min(t["miss"] - 1, 4))
                for j, b in enumerate(players):
                    d = float(np.linalg.norm(C[j] - t["p"]))
                    r = (b[2] * b[3]) / max(t["size"][0] * t["size"][1], 1)
                    if d < gate and 0.55 < r < 1.7:
                        cost[i, j] = d / gate
            ri, ci = linear_sum_assignment(cost)
            used = set()
            for i, j in zip(ri, ci):
                if cost[i, j] >= 1e6:
                    continue
                t, b = self.tracks[i], players[j]
                vel = (C[j] - t["last"]) / max(1, self.k - t["last_k"])
                t["v"] = vel if t["hits"] < 2 else 0.7 * t["v"] + 0.3 * vel
                t["p"] = C[j]
                t["last"], t["last_k"] = C[j], self.k
                t["size"] = 0.8 * t["size"] + 0.2 * np.array([b[2], b[3]], np.float64)
                t["miss"], t["hits"], t["matched"] = 0, t["hits"] + 1, j
                t["run"] = t.get("run", 0) + 1
                used.add(j)
        else:
            used = set()
        for t in self.tracks:
            if t["miss"] > 3:
                t["v"] = t["v"] * 0.97  # coasting for long: slowly forget the velocity
        self.tracks = [t for t in self.tracks if t["miss"] <= self.fps * 1.2]
        for j, b in enumerate(players):
            if j in used:
                continue
            # (a merged pair also starts a new track here; it is younger than any crowd
            # it belongs to, so targets ignore it as a neighbour)
            self.tracks.append({"id": self._next, "p": C[j], "v": np.zeros(2), "last": C[j], "last_k": self.k,
                                "size": np.array([b[2], b[3]], np.float64), "miss": 0, "hits": 1,
                                "birth": self.k, "matched": j, "run": 1})
            self._next += 1

    def nearest(self, p_ref, max_d):
        best, bd = None, max_d
        for t in self.tracks:
            d = float(np.linalg.norm(t["p"] - p_ref))
            if d < bd:
                best, bd = t, d
        return best


def fit_velocity(hist):
    """Robust velocity (px / frame) from recent (frame, position) samples: least-squares line."""
    if len(hist) < 3:
        return None
    k = np.array([h[0] for h in hist], np.float64)
    P = np.array([h[1] for h in hist], np.float64)
    k = k - k.mean()
    den = float((k * k).sum())
    if den <= 0:
        return None
    return (k[:, None] * (P - P.mean(axis=0))).sum(axis=0) / den


def new_csrt():
    try:
        p = cv2.TrackerCSRT_Params()
        p.use_segmentation = False  # ~25% faster; our own segmentation verifies boxes instead
        return cv2.TrackerCSRT_create(p)
    except Exception:  # pragma: no cover - older builds
        return cv2.TrackerCSRT_create()


class TargetTracker:
    def __init__(self, tid, name, kind, box, frame, hsv, nongrass, M, fps):
        H, W = frame.shape[:2]
        self.id, self.name, self.kind = tid, name, kind
        self.W, self.H = W, H
        box = clip_box(box, W, H)
        if kind == "ball":
            box = refine_ball_box(frame, nongrass, box)
        self.box = box
        self.size = np.array([box[2], box[3]], np.float64)
        # players: the expected size follows the camera zoom (broadcast cameras zoom in and
        # out); measurements may only adapt it within limits around size0 * zoom
        self.size0 = self.size.copy()
        self.pz0 = float(M[0, 0]) if abs(M[0, 0]) > 1e-6 else 1.0
        self.model = kit_hist(hsv, nongrass, box)
        self._lab, self._ng = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), nongrass
        self.shirt0 = shirt_colour(self._lab, nongrass, box) if kind != "ball" else None
        self.shirt = None if self.shirt0 is None else self.shirt0.copy()
        self.init_model = self.model.copy() if self.model is not None else None
        x, y, w, h = box
        roi = hsv[y:y + h, x:x + w].reshape(-1, 3).astype(np.float32)
        self.mean_hsv = roi.mean(axis=0) if len(roi) else np.array([0, 0, 255], np.float32)
        self.csrt, self.csrt_ok = None, False
        if kind != "ball":
            self._init_csrt(frame, box)
        c_ref = apply(np.linalg.inv(M), center(box))[0]
        # the ball changes velocity abruptly (kicks, deflections): larger velocity noise
        self.kf = make_kalman(*c_ref, q_vel=20.0 if kind == "ball" else 4.0)
        self.updates = 0
        self.lost = 0
        self.base = 0.6  # running self-similarity of the target -> adaptive thresholds
        self.max_predict = int(fps * 1.5)
        self.fps = fps
        # crowd handling (see _crowd_measure)
        self.k = 0
        self.hist = deque([(0, c_ref)], maxlen=max(5, int(round(fps * 0.6))))
        self.crowd, self.calm, self.crowd_start = False, 0, 0
        self.v0, self.p0, self.k0 = np.zeros(2), c_ref, 0
        self.own = None
        self.own_ids = set()
        self.nb = {}  # neighbour snapshots during a crowd: id -> position, velocity, kit histogram
        self.cut = False
        self.k_acc = 0
        if kind == "ball":
            # fixed ball size (the user's ball, refined) and a grey-level template of it
            self.ball_d = float(max(box[2], box[3]))  # at the zoom of the start frame
            self.tmpl0 = ball_template(frame, center(box), self.ball_d)
            self.tmpl = self.tmpl0.copy() if self.tmpl0 is not None else None
            # broadcast cameras zoom: ball size and template follow the camera zoom
            self.z0 = float(M[0, 0]) if abs(M[0, 0]) > 1e-6 else 1.0
            self.zoom = 1.0
            self._tcache = (None, None)
            self.pending = None  # unconfirmed re-detection while the ball is lost
            self.weak = 0  # consecutive frames measured only on/at a player (uncertain)
            self.last_meas = (0, c_ref)

    def _init_csrt(self, frame, box):
        if self.kind == "ball":  # the ball is measured by template matching, not CSRT
            self.csrt, self.csrt_ok = None, False
            return
        b = list(box)
        try:
            self.csrt = new_csrt()
            self.csrt.init(frame, tuple(int(v) for v in b))
            self.csrt_ok = True
        except Exception:
            self.csrt_ok = False

    def _box_at(self, c):
        w, h = self.size
        return clip_box([c[0] - w / 2, c[1] - h / 2, w, h], self.W, self.H)

    def _one_player(self, b, h_lo=0.6, h_hi=1.45):
        """Is blob b the size of ONE player like this one? Height is the stable measure
        (width changes with the running pose); two merged players are ~2x as wide or tall."""
        hr = b[3] / max(self.size[1], 1)
        wr = b[2] / max(self.size[0], 1)
        return h_lo <= hr <= h_hi and 0.35 <= wr <= 1.7

    def _kit_ok(self, box, strict=False):
        """Same team? The candidate's shirt colour must match this player's.
        strict: a box whose shirt colour cannot be measured (mostly grass, or a lime shirt
        hidden in the grass mask) does not pass - used in crowds and when re-finding a
        player, where accepting the wrong one is what swaps identities."""
        if self.shirt is None:
            return True
        sc = shirt_colour(self._lab, self._ng, box)
        if sc is None:
            return not strict
        return shirt_distance(self.shirt, sc) <= SHIRT_TOL

    def _clamp_speed(self, v):
        vmax = 9.0 * (self.size[1] / 1.8) / self.fps  # ~9 m/s, pixels per metre from height
        sp = float(np.hypot(*v))
        return v * (vmax / sp) if sp > vmax else np.asarray(v, np.float64)

    def reset_after_cut(self):
        """New camera shot: the target must be re-identified anywhere in the frame."""
        self.lost = self.max_predict + 1
        self.cut = True
        self.crowd = False
        self.hist.clear()

    def _update_neighbours(self, neigh, players, hsv, nongrass, my_ref, diag):
        """Follow each neighbour on ITS OWN extrapolated path.

        The shared blob tracks can themselves swap when players merge (a neighbour's track
        may grab this target's blob). A track update is therefore only believed when it
        stays close to that neighbour's constant-velocity extrapolation; otherwise the
        extrapolation is used. Returns [(position_ref, kit_hist), ...].
        """
        for t in neigh:
            if t["id"] not in self.nb:
                h = None
                if t["matched"] is not None and t["matched"] < len(players):
                    h = kit_hist(hsv, nongrass, players[t["matched"]])
                self.nb[t["id"]] = {"p": t["p"].copy(), "v": t["v"].copy(), "k": self.k, "hist": h,
                                    "h": float(t["size"][1])}
        alive = {t["id"]: t for t in neigh}
        out = []
        for i, n in list(self.nb.items()):
            dk = self.k - n["k"]
            e = n["p"] + n["v"] * dk
            t = alive.get(i)
            if t is not None and t["miss"] == 0 and np.linalg.norm(t["p"] - e) < max(5.0, 0.25 * n["h"]) + 0.5 * dk:
                if dk > 0:
                    n["v"] = 0.8 * n["v"] + 0.2 * (t["p"] - n["p"]) / dk
                n["p"], n["k"], e = t["p"].copy(), self.k, t["p"].copy()
            if np.linalg.norm(e - my_ref) > 3 * diag:
                del self.nb[i]
                continue
            out.append((e, n["hist"]))
        return out

    def _neighbours(self, bt, p_ref, radius):
        """Other players' motion tracks near p_ref that existed before the encounter began."""
        if bt is None:
            return []
        born_before = self.crowd_start if self.crowd else bt.k - 2
        out = []
        for t in bt.tracks:
            if t["id"] == self.own or t["id"] in self.own_ids or t["birth"] >= born_before or t["hits"] < 3:
                continue
            if np.linalg.norm(t["p"] - p_ref) < radius:
                out.append(t)
        return out

    def step(self, frame, hsv, nongrass, M, players, balls, claimed, bt=None, field=None, lab=None):
        self.k += 1
        Minv = np.linalg.inv(M)
        if self.kind != "ball":
            self._lab, self._ng = lab if lab is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), nongrass
            zoom = float(np.clip(M[0, 0] / self.pz0, 0.25, 4.0))
            ref = self.size0 * zoom
            if self.lost > 0:  # unseen: drift back towards the zoom-predicted size
                self.size = 0.9 * self.size + 0.1 * ref
            self.size = np.clip(self.size, 0.65 * ref, 1.5 * ref)
        prev_ref = self.kf.statePost[:2, 0].astype(np.float64).copy()
        pred_ref = self.kf.predict()[:2, 0].astype(np.float64)
        last_c = center(self.box)
        diag = float(np.hypot(*self.size))

        # ---- crowd state (players only): another player close by, or blobs merged
        neigh = []
        if self.kind != "ball":
            area = self.size[0] * self.size[1]
            neigh = self._neighbours(bt, pred_ref, (1.6 if self.crowd else 1.1) * diag)
            merged = any(b[2] * b[3] > 1.7 * area and iou(b, self.box) > 0.2 for b in players)
            crowded = bool(neigh) or merged
            if crowded and not self.crowd and self.lost <= self.max_predict:
                # entering: freeze the velocity the player had while still on his own,
                # limited to what a player can physically run (~9 m/s)
                v = fit_velocity(self.hist) if len(self.hist) >= 5 else None
                v = v if v is not None else self.kf.statePost[2:, 0].astype(np.float64)
                self.v0 = self._clamp_speed(v)
                self.p0, self.k0 = prev_ref, self.k - 1
                self.crowd, self.calm, self.crowd_start = True, 0, (bt.k - 1 if bt else 0)
                # every motion track sitting on this player right now is "self", not a neighbour
                self.own_ids = {t["id"] for t in (bt.tracks if bt else [])
                                if np.linalg.norm(t["p"] - prev_ref) < max(4.0, 0.4 * self.size[0])}
                self.nb = {}
                neigh = self._neighbours(bt, pred_ref, 1.6 * diag)
            elif self.crowd:
                self.calm = 0 if crowded else self.calm + 1
            if self.crowd:
                # constant-velocity extrapolation from the last accepted position; the
                # Kalman velocity is not trusted while the players are entangled
                pred_ref = self.p0 + self.v0 * (self.k - self.k0)
                self.kf.statePost[:2, 0] = pred_ref.astype(np.float32)
                self.kf.statePost[2:, 0] = self.v0.astype(np.float32)
                self.kf.statePre[:] = self.kf.statePost
            if self.crowd:
                nb_pos = self._update_neighbours(neigh, players, hsv, nongrass, pred_ref, diag)
        pred = apply(M, pred_ref)[0]
        ref_point = pred if (self.updates >= 3 or self.crowd) else last_c

        # ---- 1. appearance tracker proposal
        csrt_box, csrt_sim = None, 0.0
        if self.csrt_ok:
            try:
                ok, b = self.csrt.update(frame)
            except Exception:
                ok = False
            if ok:
                b = list(b)
                # CSRT estimates scale too and can stretch onto two players (one behind the
                # other). Keep its position, but use this player's own size; reject a box
                # that has grown or shrunk far beyond it.
                if 0.65 < b[3] / max(self.size[1], 1) < 1.5 and 0.5 < b[2] / max(self.size[0], 1) < 1.8:
                    cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
                    if 0.8 < b[3] / max(self.size[1], 1) < 1.25:
                        w, h = b[2], b[3]  # normal: CSRT's own scale (follows the zoom)
                    else:
                        w, h = self.size
                    csrt_box = clip_box([cx - w / 2, cy - h / 2, w, h], self.W, self.H)
                    csrt_sim = similarity(self.model, kit_hist(hsv, nongrass, csrt_box))
                else:
                    self._init_csrt(frame, self.box)  # it has drifted: restart on the last box

        clear = True
        if self.kind == "ball":
            box, status, conf = self._ball_measure(frame, hsv, ref_point, balls, players, field, M)
        elif self.crowd:
            box, status, conf, clear = self._crowd_measure(hsv, nongrass, ref_point, csrt_box, csrt_sim,
                                                           players, claimed, diag, nb_pos, M, bt)
        else:
            box, status, conf = self._player_measure(hsv, nongrass, ref_point, csrt_box, csrt_sim,
                                                     players, claimed, diag, bt, M)

        # ---- 4. update state
        weak_only = self.kind == "ball" and box is not None and self.weak > self.fps
        if box is not None:
            c_ref = apply(Minv, center(box))[0]
            if self.kind != "ball" and self.crowd:
                if clear and self.k > self.k0:  # clean, unambiguous measurement: adapt velocity a little
                    v_inst = (c_ref - self.p0) / (self.k - self.k0)
                    self.v0 = self._clamp_speed(0.85 * self.v0 + 0.15 * v_inst)
                self.p0, self.k0 = c_ref, self.k
                self.kf.statePost[:2, 0] = c_ref.astype(np.float32)
                self.kf.statePost[2:, 0] = self.v0.astype(np.float32)
            else:
                if self.kind == "ball":
                    # a weak answer (ball at a player's feet, or a sock) is noisy: trust it less
                    r_ = 4.0 if self.weak == 0 else 30.0
                    self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_
                self.kf.correct(np.array([[c_ref[0]], [c_ref[1]]], np.float32))
            self.updates += 1
            if weak_only:
                # only weak matches (on a player's feet / legs) for over a second: the ball
                # may well be gone. Keep it with the player, but count it as unseen so the
                # wider search for a clear ball keeps running
                self.lost += 1
                status, conf = PREDICTED, 0.0
            else:
                self.lost = 0
            self.cut = False
            self.k_acc = self.k
            if self.kind != "ball" and not self.crowd:
                if status in (TRACKED, REDETECTED) and conf > 0:
                    self.base = float(np.clip(0.95 * self.base + 0.05 * conf, 0.3, 0.9))
                # adapt the size only to boxes of about this player's size: a box that has
                # grown onto two players (a duel) must not become the new normal
                if 0.75 < box[3] / max(self.size[1], 1) < 1.33 and 0.5 < box[2] / max(self.size[0], 1) < 1.6:
                    self.size = 0.85 * self.size + 0.15 * np.array([box[2], box[3]])
                if conf > 0.85 * self.base and self.model is not None:
                    hb = kit_hist(hsv, nongrass, box)
                    if hb is not None:  # slow adaptation, anchored to the initial model
                        self.model = 0.9 * self.model + 0.05 * hb + 0.05 * self.init_model
                    sc = shirt_colour(self._lab, nongrass, box)
                    if self.shirt is not None and sc is not None and shirt_distance(self.shirt, sc) < 12:
                        self.shirt = 0.95 * self.shirt + 0.025 * sc + 0.025 * self.shirt0
                self.hist.append((self.k, c_ref))
                if bt is not None:  # remember which motion track is this player's own
                    t = bt.nearest(c_ref, max(4.0, 0.35 * self.size[0]))
                    self.own = t["id"] if t is not None else self.own
            elif self.kind == "ball":
                k_last, p_last = self.last_meas
                if self.k - k_last > 2 and self.weak == 0:
                    # re-found after a gap (e.g. a pass while hidden): restart the velocity
                    v = (c_ref - p_last) / (self.k - k_last)
                    self.kf.statePost[:2, 0] = c_ref.astype(np.float32)
                    self.kf.statePost[2:, 0] = v.astype(np.float32)
                if self.weak == 0:  # last STRONG sighting: what a pass is traced back to
                    self.last_meas = (self.k, c_ref)
                vmax = 30.0 * (self.ball_d * self.zoom / 0.2 / 1.8) / self.fps
                v = self.kf.statePost[2:, 0]
                sp = float(np.hypot(*v))
                if sp > vmax:
                    self.kf.statePost[2:, 0] = v * (vmax / sp)
            if status == REDETECTED:
                self._init_csrt(frame, box)
            self.box = box
        else:
            self.lost += 1
            if not self.crowd:
                # damp velocity while unseen so predictions do not run away (a passed
                # ball is usually stopped by a player, so it slows down faster)
                self.kf.statePost[2:] *= 0.85 if self.kind == "ball" else 0.92
            box = self._box_at(pred)
            self.box = box
            status = PREDICTED if self.lost <= self.max_predict else LOST
            conf = 0.0
            if self.kind == "ball":
                box = self._box_at(pred)
                self.box = box
            elif self.lost % 15 == 0 and not self.crowd:  # occasionally re-centre CSRT on the prediction
                self._init_csrt(frame, box)

        # ---- leaving the crowd: players clearly apart for a few frames
        if self.kind != "ball" and self.crowd and (self.calm >= 3 or self.lost > self.max_predict):
            self.crowd = False
            self.hist.clear()
            self.hist.append((self.k, apply(Minv, center(self.box))[0]))
            if status in (TRACKED, REDETECTED):
                self._init_csrt(frame, self.box)  # CSRT may have followed the other player
        return {"box": [int(v) for v in box], "status": status, "conf": round(float(conf), 3)}

    # ------------------------------------------------------------------
    def _crowd_measure(self, hsv, nongrass, ref_point, csrt_box, csrt_sim, players, claimed, diag, neigh, M, bt=None):
        """Measurement while another player is close (crossing, shoulder to shoulder, merged).

        Team-mates in the same kit cannot be told apart by colour, so a candidate box is
        accepted only if it
          * is close to this player's own constant-velocity extrapolation, and
          * is clearly closer to it than to every neighbour's extrapolated position.
        When the players overlap, every candidate is ambiguous and the target coasts on
        its extrapolated path until they separate.
        """
        area = self.size[0] * self.size[1]
        n = self.k - self.k_acc  # frames since the last accepted measurement
        v = float(np.hypot(*self.v0))
        tight = min(0.6 * diag, max(6.0, 0.15 * diag + 0.08 * n * max(v, 1.0) + 3.0))
        nb = [(apply(M, p)[0], h) for p, h in neigh]

        def taken(b):
            return any(iou(b, c) > 0.45 for c in claimed)

        cands = []
        if csrt_box is not None and not taken(csrt_box):
            r = csrt_box[2] * csrt_box[3] / max(area, 1)
            if 0.5 < r < 1.7 and csrt_sim >= 0.5 * self.base and self._kit_ok(csrt_box, strict=True):
                cands.append((csrt_box, csrt_sim, True))
        for b in players:
            if taken(b):
                continue
            if not self._one_player(b):
                continue
            if np.linalg.norm(center(b) - ref_point) > tight:
                continue
            s = similarity(self.model, kit_hist(hsv, nongrass, b))
            if s >= 0.6 * self.base and self._kit_ok(b, strict=True):
                cands.append((b, s, False))
        best, best_score, best_clear = None, 1e9, False
        for b, s, is_csrt in cands:
            c = center(b)
            dm = float(np.linalg.norm(c - ref_point))
            if dm > tight:
                continue
            # neighbours that could explain this candidate: close to it AND wearing a
            # similar kit (an opponent in another kit is told apart by colour)
            hb = kit_hist(hsv, nongrass, b)
            dn = 1e9
            for q, h in nb:
                if h is not None and s - similarity(h, hb) > 0.15:
                    continue
                dn = min(dn, float(np.linalg.norm(c - q)))
            if dm > 0.6 * dn:  # as well explained by a neighbour -> ambiguous
                continue
            score = dm / tight + 0.5 * (1 - s) + (0.1 if is_csrt else 0.0)
            if score < best_score:
                best, best_score, best_clear, best_s, best_csrt = b, score, dm < 0.35 * dn, s, is_csrt
        if best is None:
            return self._isolated_candidate(hsv, nongrass, ref_point, players, claimed, diag, M, bt)
        status = TRACKED if best_csrt or (csrt_box is not None and iou(best, csrt_box) > 0.3) else REDETECTED
        return best, status, best_s, best_clear

    def _isolated_candidate(self, hsv, nongrass, ref_point, players, claimed, diag, M, bt):
        """Fallback after a long encounter, when this player has drifted off his
        extrapolated path (he turned while hidden): a single-player blob near him that no
        other player's live motion track accounts for must be him - but only if it is the
        ONLY such blob, otherwise the situation is still ambiguous and he keeps coasting."""
        n = self.k - self.k_acc
        if bt is None or n < max(3, int(0.3 * self.fps)):
            return None, None, 0.0, False
        area = self.size[0] * self.size[1]
        # how far he can have left his extrapolated path in that time
        vmax = 9.0 * (self.size[1] / 1.8) / self.fps
        radius = min(1.3 * diag, 0.3 * diag + 0.5 * vmax * n)
        # he may simply be hidden inside a merged blob at his expected position (behind
        # another player): then the lone player nearby is somebody else
        mx = 0.25 * self.size[0]
        for b in players:
            if b[2] * b[3] > 1.5 * area and b[0] - mx <= ref_point[0] <= b[0] + b[2] + mx \
                    and b[1] - mx <= ref_point[1] <= b[1] + b[3] + mx:
                return None, None, 0.0, False
        # only a track matched without a break for a while is trusted to be "someone
        # else" (tracks can swap while players overlap)
        born_before = bt.k - n
        live = [apply(M, t["p"])[0] for t in bt.tracks
                if t["miss"] == 0 and t.get("run", 0) >= 8 and t["id"] not in self.own_ids and t["id"] != self.own
                and t["birth"] < born_before]
        free = []
        for b in players:
            if any(iou(b, c) > 0.45 for c in claimed):
                continue
            if not self._one_player(b):
                continue
            c = center(b)
            if np.linalg.norm(c - ref_point) > radius:
                continue
            if any(np.linalg.norm(c - q) < 0.3 * diag for q in live):
                continue  # this blob is another player's
            s = similarity(self.model, kit_hist(hsv, nongrass, b))
            if s >= 0.7 * self.base:
                if self._kit_ok(b, strict=True):
                    free.append((b, s))
        if len(free) != 1:
            return None, None, 0.0, False
        return free[0][0], REDETECTED, free[0][1], False

    def _player_measure(self, hsv, nongrass, ref_point, csrt_box, csrt_sim, players, claimed, diag, bt=None, M=None):
        gate = diag * (1.2 + 0.4 * min(self.lost, 10))
        area = self.size[0] * self.size[1]

        def taken(b):
            return any(iou(b, c) > 0.45 for c in claimed)

        # after coasting through an occlusion CSRT may be locked on the wrong player,
        # so re-acquire from segmentation blobs near the prediction first
        if csrt_box is not None and not taken(csrt_box) and self.lost == 0:
            d = np.linalg.norm(center(csrt_box) - ref_point)
            a = csrt_box[2] * csrt_box[3]
            if csrt_sim >= 0.75 * self.base and d <= gate and 0.45 < a / max(area, 1) < 1.8 and self._kit_ok(csrt_box):
                return csrt_box, TRACKED, csrt_sim

        # blobs that clearly belong to another player's motion track (seen recently, much
        # closer to that player's path than to ours) are not candidates while the target
        # was only briefly unseen
        others = []
        if bt is not None and M is not None and 0 < self.lost <= self.fps:
            # (a track born while this target was unmeasured may well be the target itself)
            born_before = bt.k - (self.k - self.k_acc)
            others = [apply(M, t["p"])[0] for t in bt.tracks
                      if t["id"] != self.own and t["id"] not in self.own_ids and t["hits"] >= 5
                      and t["miss"] <= 2 and t["birth"] < born_before]

        # ---- re-detection among segmentation blobs
        # A player cannot be further away than he can run in the time he was unseen
        # (~9 m/s; pixels per metre from his height, assumed 1.8 m). Only after a camera
        # cut is the whole frame searched.
        long_lost = self.lost > self.max_predict
        w_dist = 0.3
        if long_lost and not self.cut:
            px_per_m = self.size[1] / 1.8
            search = diag + 9.0 * px_per_m * self.lost / self.fps
            long_lost = False
            w_dist = 0.6  # team-mates look alike: the nearest plausible player is most likely
        else:
            search = gate
        best, best_score, best_sim, second = None, -1, 0, -1
        for b in players:
            if taken(b):
                continue
            a = b[2] * b[3]
            ratio = a / max(area, 1)
            if not self._one_player(b, 0.55, 1.5):  # (two merged players are not one player)
                continue
            d = np.linalg.norm(center(b) - ref_point)
            if d > search and not long_lost:
                continue
            if others and d > 0.3 * diag:
                dn = min(float(np.linalg.norm(center(b) - q)) for q in others)
                if dn < 0.25 * d:
                    continue
            if not self._kit_ok(b, strict=self.lost > 0):
                continue  # other team's (or a mixed) shirt
            s = similarity(self.model, kit_hist(hsv, nongrass, b))
            dist_term = max(0.0, 1 - d / max(search, 1)) if not long_lost else 0.5
            size_term = 1 - min(abs(np.log(ratio)), 1)
            score = 0.6 * s + w_dist * dist_term + 0.1 * size_term
            if score > best_score:
                second = best_score
                best, best_score, best_sim = b, score, s
            elif score > second:
                second = score
        need = (0.85 if self.lost > self.max_predict else 0.7) * self.base
        # after a long loss / camera cut the whole frame is searched, where team-mates in
        # the same kit look alike: only accept a clear, unambiguous winner
        if self.lost > self.max_predict and best is not None and best_score - second < 0.06:
            best = None
        if best is not None and best_sim >= need:
            if csrt_box is not None and iou(best, csrt_box) > 0.3:
                return best, TRACKED, best_sim
            return best, REDETECTED, best_sim

        # ---- weaker CSRT answer (e.g. partly occluded)
        if csrt_box is not None and not taken(csrt_box) and self.lost == 0:
            d = np.linalg.norm(center(csrt_box) - ref_point)
            if csrt_sim >= 0.5 * self.base and d <= gate and self._kit_ok(csrt_box):
                return csrt_box, TRACKED, csrt_sim
        return None, None, 0.0

    def _scaled_template(self):
        """The ball template resized to the current camera zoom (cached)."""
        if self.tmpl is None:
            return None
        key = round(self.zoom, 2)
        if self._tcache[0] == key:
            return self._tcache[1]
        if abs(self.zoom - 1.0) < 0.03:
            t = self.tmpl
        else:
            n = max(5, int(round(self.tmpl.shape[0] * self.zoom)) | 1)
            t = cv2.resize(self.tmpl, (n, n), interpolation=cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_LINEAR)
        self._tcache = (key, t)
        return t

    @staticmethod
    def _contrast(gray, x, y, d):
        """Brightest pixel of the candidate minus the median of a ring around it."""
        H, W = gray.shape
        r = max(1, int(round(d / 3)))
        R0, R1 = max(2, int(round(0.9 * d))), max(3, int(round(1.5 * d)))
        if x - R1 < 0 or y - R1 < 0 or x + R1 >= W or y + R1 >= H:
            return 255.0
        core = gray[y - r:y + r + 1, x - r:x + r + 1]
        win = gray[y - R1:y + R1 + 1, x - R1:x + R1 + 1].astype(np.float32)
        yy, xx = np.mgrid[-R1:R1 + 1, -R1:R1 + 1]
        rr = np.hypot(xx, yy)
        ring = win[(rr >= R0) & (rr <= R1)]
        return float(core.max()) - float(np.median(ring))

    def _confirm_far(self, c, d, vmax, M):
        """A ball candidate far from the prediction is accepted only when seen in two
        consecutive frames AND its motion traces back to where the ball was last
        measured (a passed ball moves away from the passer; a static white spot on the
        pitch does not). After a long loss only the two-frame confirmation is needed."""
        p = self.pending
        self.pending = (c, self.k)
        if p is None or self.k - p[1] > 2:
            return False
        dk = self.k - p[1]
        if np.linalg.norm(c - p[0]) > 2.5 * d + vmax * dk:
            return False
        v = (c - p[0]) / dk
        k_last, p_last_ref = self.last_meas
        if self.k - k_last > self.fps * 1.5:
            return True
        p_last = apply(M, p_last_ref)[0]
        back = p[0] - v * (p[1] - k_last)
        gap = float(np.linalg.norm(p[0] - p_last))
        return float(np.linalg.norm(back - p_last)) <= max(3.0 * d, 0.35 * gap)

    def _ball_measure(self, frame, hsv, ref_point, balls, players, field=None, M=None):
        """Locate the ball near its predicted position.

        Candidates come from three sources:
          * small round (or motion-blurred, elongated) segmentation blobs,
          * peaks of normalised cross-correlation with the ball template, which also find
            the ball when it touches a player's boot and is part of the player's blob,
        Each candidate is refined to its local template peak and scored by distance to
        the prediction, template similarity, colour and whether it sits on a player's
        upper body (heads, white shirts and socks look like a ball). The search radius
        grows the longer the ball is unseen, up to the whole pitch; far from the
        prediction a candidate must be confirmed in two consecutive frames.
        """
        if M is not None:
            self.zoom = float(np.clip(M[0, 0] / self.z0, 0.25, 4.0))
        d = self.ball_d * self.zoom
        tmpl = self._scaled_template()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # fastest plausible ball movement per processed frame (~30 m/s); pixels per metre
        # follow from the ball size (a player is ~ 5 ball diameters tall per 1.8 m)
        vmax = 30.0 * (d / 0.2 / 1.8) / self.fps
        unsure = min(self.lost, 60) + min(self.weak, 12)
        gate = 2.5 * d + vmax * (0.6 + 0.6 * unsure)
        far = self.lost > 3 or self.weak > 3
        cands = [(center(b), "blob") for b in balls if np.linalg.norm(center(b) - ref_point) <= gate]
        min_ncc = 0.55 if far else 0.42
        for x, y, sc in ncc_peaks(gray, tmpl, ref_point, min(gate, self.W), k=4 if far else 3,
                                  min_score=min_ncc, mask=field):
            cands.append((np.array([x, y]), "ncc"))
        # while unseen, the ball is often simply where it was last seen (stopped by a
        # player): search around that point as well as around the prediction
        anchors = [ref_point]
        if self.lost > 0 and M is not None:
            p_last = apply(M, self.last_meas[1])[0]
            anchors.append(p_last)
            for x, y, sc in ncc_peaks(gray, tmpl, p_last, 2.5 * d + vmax * 0.6, k=2,
                                      min_score=0.42, mask=field):
                cands.append((np.array([x, y]), "ncc"))
        # sudden kicks are handled by the far-candidate confirmation, so the "near" window
        # only has to cover normal rolling / dribbling motion
        # "near" = where the ball can plausibly be without any doubt. Anything further away
        # (a pass, or the ball re-appearing after it was lost) must be confirmed on two
        # frames and trace back to the last clear sighting (_confirm_far), so a wrong
        # prediction can never pull the ball onto a white spot nearby.
        near_gate = 2.5 * d + vmax * (0.4 + 0.3 * min(self.lost, 3))
        scored = []
        for c, src in cands:
            pk = ncc_peaks(gray, tmpl, c, max(2, d * 0.5), k=1, min_score=-1)
            if pk:
                c2, ncc = np.array(pk[0][:2]), pk[0][2]
                if np.linalg.norm(c2 - c) <= d:
                    c = c2
            else:
                ncc = 0.0
            dist = min(float(np.linalg.norm(c - a_)) for a_ in anchors)
            if dist > gate:
                continue
            x, y = int(round(c[0])), int(round(c[1]))
            if self._contrast(gray, x, y, d) < 18:
                continue  # not brighter than the grass around it: a grass patch, not the ball
            r = max(1, int(d / 3))
            roi = hsv[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3).astype(np.float32)
            if len(roi) == 0:
                continue
            col = np.abs(roi.mean(axis=0) - self.mean_hsv) / np.array([90, 255, 255])
            col_d = float(min(1.0, col[1] + col[2] + 0.5 * min(col[0], 1)))
            body = any(px - 1 <= c[0] <= px + pw_ + 1 and py - 1 <= c[1] <= py + 0.75 * ph_
                       for px, py, pw_, ph_ in players)
            # on or touching a player (feet, legs): could be the ball at his feet, or a
            # sock / boot. Such a "weak" measurement never beats a clear, isolated ball.
            touching = any(px - 0.3 * pw_ <= c[0] <= px + 1.3 * pw_ and py <= c[1] <= py + ph_ + 0.2 * ph_
                           for px, py, pw_, ph_ in players)
            strong = ncc >= 0.6 and not touching
            g_ = near_gate if dist <= near_gate else gate
            cost = 0.8 * dist / g_ + 1.2 * (1 - ncc) + 0.5 * col_d + (0.5 if body else 0.0)
            ok = ncc >= (min_ncc if dist <= near_gate else 0.55) or \
                (src == "blob" and ncc >= 0.3 and col_d < 0.35 and not body)
            if not touching and ncc < 0.6 and not (src == "blob" and ncc >= 0.45 and col_d < 0.35):
                # a mediocre match on open grass is a grass patch, not a hidden ball: the
                # ball can only be partly hidden where a player is
                ok = False
            if ok and cost <= 1.5:
                scored.append((c, cost, ncc, strong, dist <= near_gate))

        def pick(lst):
            if not lst:
                return None
            st = [t for t in lst if t[3]]
            return min(st or lst, key=lambda t: t[1])

        near = pick([t for t in scored if t[4]])
        far_c = pick([t for t in scored if not t[4] and t[3]])
        choice = None
        if near is not None and near[3]:
            choice = near  # clear ball close to where it was expected
        elif far_c is not None and self._confirm_far(far_c[0], d, vmax, M):
            choice = far_c  # the ball has left the player / re-appeared elsewhere
        elif near is not None:
            choice = near  # only a weak answer: probably the ball at a player's feet
        if far_c is None:
            self.pending = None
        if choice is None:
            return None, None, 0.0
        best, best_cost, best_strong = choice[0], choice[1], choice[3]
        if choice is far_c:
            self.pending = None
        self.weak = 0 if best_strong else self.weak + 1
        # (no template adaptation: a template that adapts to what it matched slowly turns
        # into a grass template after a few wrong matches; the white ball itself changes
        # little, and zoom is handled by _scaled_template)
        conf = float(np.clip(1 - best_cost / 1.5, 0, 1))
        box = clip_box([best[0] - d / 2, best[1] - d / 2, d, d], self.W, self.H)
        return box, TRACKED if self.lost == 0 else REDETECTED, conf


class MultiTracker:
    def __init__(self, targets, frame, hsv, nongrass, M, fps):
        self.trackers = [TargetTracker(t["id"], t["name"], t["kind"], t["box"], frame, hsv, nongrass, M, fps)
                         for t in targets]
        self.blobs = BlobTracks(fps)

    def ball_size_hint(self):
        sizes = [t.ball_d * t.zoom for t in self.trackers if t.kind == "ball"]
        return float(np.median(sizes)) if sizes else None

    def boxes(self):
        return [t.box for t in self.trackers]

    def step(self, frame, hsv, nongrass, M, players, balls, field=None):
        # confident targets first, so they claim their blobs before uncertain ones
        self.blobs.update(players, np.linalg.inv(M))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        order = sorted(self.trackers, key=lambda t: t.lost)
        claimed, out = [], {}
        for t in order:
            r = t.step(frame, hsv, nongrass, M, players, balls, claimed, self.blobs, field, lab)
            if r["status"] in (TRACKED, REDETECTED) and t.kind != "ball":
                claimed.append(r["box"])
            out[t.id] = r
        return out
