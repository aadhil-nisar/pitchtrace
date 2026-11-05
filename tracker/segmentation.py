"""Segmentation: isolate players (and ball candidates) from the pitch.

Method
------
1. Grass candidates: a hue histogram (HSV) of saturated pixels is built; its dominant
   peak inside the green band (H in [25, 95], OpenCV 0-179 scale) and the interval
   around it (bins above 8% of the peak) select candidate grass pixels.
2. Statistical grass colour model: hue alone is not enough on real footage - a team in
   lime / neon-green kits has exactly the grass hue but is much brighter. So the
   candidates are modelled in CIE-Lab as a Gaussian (mean mu, covariance Sigma), fitted
   robustly (2-98% trimming + two re-weighting passes so kit pixels are rejected).
       grass(x,y) = 1  if  (c - mu)^T Sigma^-1 (c - mu) < T      (Mahalanobis distance)
   with T = 1.6 x the 97th percentile of the training distances (data-driven).
   The test is precomputed into a 64^3 Lab look-up table, so it costs one table lookup
   per pixel. The model is refreshed every ~2 s with samples from newer frames, so
   shadows and lighting changes during camera pans are learnt.
3. Field region: the grass mask is closed with a large kernel; for every column the
   top-most and bottom-most rows of the main grass region give the field boundary, and
   both boundary profiles are median-filtered across columns (window = W/12). Players
   standing on the touchline stay inside, while crowd, stands, advertising boards and
   dugouts are excluded.
4. Foreground = field AND NOT grass. Pitch lines, players and the ball remain.
   Pitch markings are removed: white (low S, high V) pixels that disappear under a 7x7
   opening are "thin"; thin connected components longer than 5% of the frame width are
   lines, while short ones (white shorts, socks) are kept.
5. Morphology: an opening (3x3 ellipse) removes thin pitch lines and noise, then a closing
   with a tall kernel (3x7) re-joins torso, shorts and legs of the same player.
6. Connected components + shape filtering: a blob is a player when its height
   (>= 1.7% of frame height), aspect ratio (0.8 <= h / w <= 6) and fill ratio
   (area / box area >= 0.18) are plausible for a person. Wide blobs (a player merged with
   a touchline strip or shadow) get a vertical opening that keeps upright bodies only.
   Finally a perspective model  h = a*y_feet + b  is fitted to the detections and blobs
   whose height is implausible for their distance from the camera are rejected. Ball
   candidates are small, roughly circular blobs taken from the un-opened foreground.
"""
import cv2
import numpy as np


class PitchSegmenter:
    def __init__(self, reference_bgr):
        self.learn(reference_bgr)
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        self.k_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # ---------- grass colour model ----------
    def _hue_band(self, hsv):
        h, s, v = cv2.split(hsv)
        valid = ((s > 40) & (v > 30)).astype(np.uint8)
        hist = cv2.calcHist([h], [0], valid, [180], [0, 180]).ravel()
        hist = np.convolve(hist, np.ones(3) / 3, mode="same")
        band = hist.copy()
        band[:25] = 0
        band[96:] = 0
        peak = int(np.argmax(band)) if band.max() > 0 else 50
        thr = 0.08 * max(band[peak], 1)
        lo = peak
        while lo > 20 and hist[lo - 1] > thr:
            lo -= 1
        hi = peak
        while hi < 100 and hist[hi + 1] > thr:
            hi += 1
        lo, hi = min(lo, peak - 6), max(hi, peak + 6)
        return hist, peak, lo, hi, (h >= lo) & (h <= hi) & (valid > 0)

    def _samples(self, bgr, n):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist, peak, lo, hi, sel = self._hue_band(hsv)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)[sel.ravel()].astype(np.float64)
        if len(lab) > n:
            lab = lab[np.random.default_rng(0).choice(len(lab), n, replace=False)]
        return hist, peak, lo, hi, lab

    def _fit(self):
        X = np.vstack(self.pool)
        # robust fit: trim extremes, then two reweighting passes (kits that share the
        # grass hue, e.g. lime/neon green, are a small minority and get rejected)
        lo_p, hi_p = np.percentile(X, 2, axis=0), np.percentile(X, 98, axis=0)
        Y = X[np.all((X >= lo_p) & (X <= hi_p), axis=1)]
        for _ in range(2):
            mu = Y.mean(axis=0)
            cov = np.cov(Y.T) + np.eye(3) * 4.0
            inv = np.linalg.inv(cov)
            d2 = np.einsum("ij,jk,ik->i", Y - mu, inv, Y - mu)
            T = max(9.0, 1.6 * np.percentile(d2, 97))
            Y = Y[d2 < T]
        self.mu, self.inv, self.T = mu, inv, float(T)
        # 64x64x64 look-up table over quantised Lab -> grass yes/no (fast per-frame test)
        c = np.arange(64) * 4 + 2.0
        L, A, B = np.meshgrid(c, c, c, indexing="ij")
        G = np.stack([L.ravel(), A.ravel(), B.ravel()], 1) - mu
        d2 = np.einsum("ij,jk,ik->i", G, inv, G)
        self.lut = np.where(d2 < T, 255, 0).astype(np.uint8)

    def learn(self, bgr):
        hist, peak, lo, hi, lab = self._samples(bgr, 40000)
        self.h_lo, self.h_hi, self.peak, self.hue_hist = int(lo), int(hi), peak, hist
        self.base = lab
        self.recent = []
        self.pool = [lab]
        self._fit()

    def refresh(self, bgr):
        """Re-learn with samples from a newer frame (lighting / shadows change as the camera pans)."""
        hist, peak, lo, hi, lab = self._samples(bgr, 10000)
        if abs(peak - self.peak) > 10 or len(lab) < 500:
            return  # probably a close-up / replay frame: do not adapt to it
        self.recent = (self.recent + [lab])[-4:]
        self.pool = [self.base] + self.recent
        self._fit()

    def grass_mask(self, bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        idx = ((lab[..., 0] >> 2).astype(np.int32) << 12) | ((lab[..., 1] >> 2).astype(np.int32) << 6) | (lab[..., 2] >> 2)
        return self.lut[idx]

    def params(self):
        return {"hue_band": [self.h_lo, self.h_hi], "hue_peak": self.peak,
                "grass_lab_mean": [round(float(x), 1) for x in self.mu],
                "mahalanobis_threshold_d2": round(self.T, 2),
                "line_removal": "white thin components > 5% width",
                "open_kernel": "ellipse 3x3", "close_kernel": "rect 3x7"}

    def field_mask(self, grass):
        """Playing-area mask from per-column top/bottom grass boundaries.

        A convex hull would also swallow advertising boards and dugouts between grass
        corners. Instead, for every column the top-most and bottom-most grass row of the
        main grass region is found and both boundary profiles are median-filtered across
        columns: the narrow notch a player standing on the boundary makes is removed,
        while the real edge of the grass (ads, stands, track) is followed.
        """
        H, W = grass.shape
        k = max(9, (min(H, W) // 40) | 1)
        closed = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.04 * H * W]
        if not keep:
            return np.full((H, W), 255, np.uint8)
        G = np.isin(lab, keep)
        has = G.any(axis=0)
        top = np.where(has, G.argmax(axis=0), -1).astype(np.float64)
        bot = np.where(has, H - 1 - G[::-1].argmax(axis=0), -1).astype(np.float64)
        xs = np.arange(W)
        if has.sum() < 2:
            return np.full((H, W), 255, np.uint8)
        top = np.interp(xs, xs[has], top[has])
        bot = np.interp(xs, xs[has], bot[has])
        win = max(5, (W // 12) | 1)
        from scipy.ndimage import median_filter
        top = median_filter(top, size=win, mode="nearest")
        bot = median_filter(bot, size=win, mode="nearest")
        rows = np.arange(H)[:, None]
        field = ((rows >= top[None, :] + 1) & (rows <= bot[None, :] - 1) & has[None, :])
        # columns at the very left/right without grass stay outside the field
        return (field.astype(np.uint8) * 255)

    # ---------- masks ----------
    def masks(self, bgr):
        H, W = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        grass = self.grass_mask(bgr)

        field = self.field_mask(grass)
        fg_raw = cv2.bitwise_and(field, cv2.bitwise_not(grass))

        # pitch markings: thin white structures that are long (players' white shorts or
        # socks are thin too, but short). Removing them stops players merging with lines.
        white = cv2.inRange(hsv, (0, 0, 160), (180, 80, 255))
        white = cv2.bitwise_and(white, field)
        thin = cv2.bitwise_and(white, cv2.bitwise_not(cv2.morphologyEx(white, cv2.MORPH_OPEN, self.k_line)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
        lines = np.zeros_like(thin)
        long_ids = [i for i in range(1, n) if max(st[i, 2], st[i, 3]) > 0.05 * W]
        if long_ids:
            lines = np.isin(lab, long_ids).astype(np.uint8) * 255
            lines = cv2.dilate(lines, self.k_open)
        fg = cv2.bitwise_and(fg_raw, cv2.bitwise_not(lines))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)
        return {"hsv": hsv, "grass": grass, "field": field, "fg_raw": fg_raw, "lines": lines, "fg": fg}

    # ---------- detection ----------
    def detect_players(self, fg, H, W):
        min_h = max(9, int(0.017 * H))
        max_h = int(0.45 * H)

        def ok(w, h, area):
            if h < min_h or h > max_h or w > 0.25 * W:
                return False
            aspect, fill = h / max(w, 1), area / float(w * h)
            return 0.8 <= aspect <= 6 and fill >= 0.18 and area >= 25

        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        boxes, split = [], []
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(7, min_h - 2)))
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if ok(w, h, area):
                boxes.append([int(x), int(y), int(w), int(h)])
            elif h >= min_h and w > 0.8 * h and area >= 60:
                # wide blob: a player merged with a touchline strip / shadow / ad edge.
                # A vertical opening keeps upright bodies and removes horizontal strips.
                roi = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
                roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kv)
                m, _, st2, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
                for j in range(1, m):
                    x2, y2, w2, h2, a2 = st2[j]
                    if ok(w2, h2, a2):
                        split.append([int(x + x2), int(y + y2), int(w2), int(h2)])
        return self._perspective_filter(boxes, split)

    @staticmethod
    def _perspective_filter(boxes, split):
        """Players further away look smaller: fit  height = a * y_feet + b  on the detections
        (robustly, two passes) and drop blobs far too small or too large for their position."""
        if len(boxes) < 5:
            return boxes + split
        B = np.array(boxes, np.float64)
        yb, hh = B[:, 1] + B[:, 3], B[:, 3]
        keep = np.ones(len(B), bool)
        for _ in range(2):
            a, b = np.polyfit(yb[keep], hh[keep], 1)
            exp = np.maximum(a * yb + b, 4)
            keep = (hh > 0.5 * exp) & (hh < 2.0 * exp)
            if keep.sum() < 4:
                return boxes + split
        out = []
        for x, y, w, h in boxes + split:
            e = max(a * (y + h) + b, 4)
            if 0.5 * e < h < 2.2 * e:
                out.append([x, y, w, h])
        return out

    def detect_ball_candidates(self, fg_raw, field, size_hint=None):
        """Small round blobs. size_hint = expected ball diameter in pixels (from the user's box)."""
        H, W = fg_raw.shape
        m = cv2.morphologyEx(fg_raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        if size_hint:
            d = max(size_hint, 3)
            a_min, a_max = 0.15 * d * d, 2.5 * d * d
        else:
            d = None
            a_min, a_max = 4, 0.0004 * H * W + 30
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            asp = w / max(h, 1)
            fill = area / float(w * h)
            if a_min <= area <= a_max and 0.4 <= asp <= 2.5 and fill >= 0.35:
                out.append([int(x), int(y), int(w), int(h)])
            elif d is not None and a_min <= area <= 4.5 * d * d and min(w, h) <= 1.6 * d \
                    and max(w, h) <= 6 * d and fill >= 0.25:
                # a fast pass is blurred into a short streak: the ball is at its centre
                out.append([int(x), int(y), int(w), int(h)])
        return out

    @staticmethod
    def ball_size_model(players, H):
        """Expected ball diameter (px) as a function of image row.

        Players further from the camera look smaller, so the expected ball size comes from
        the perspective line  h = a*y_feet + b  fitted to the detected players: the ball is
        ~0.2 x the height of a player standing at the same row (broadcast blur makes the
        22 cm ball look slightly bigger than its true 0.12 ratio)."""
        if len(players) >= 5:
            B = np.array(players, np.float64)
            yb, hh = B[:, 1] + B[:, 3], B[:, 3]
            try:
                a, b = np.polyfit(yb, hh, 1)
                r = np.abs(hh - (a * yb + b))
                keep = r <= np.percentile(r, 75) + 1e-6
                if keep.sum() >= 4:
                    a, b = np.polyfit(yb[keep], hh[keep], 1)
                lo, hi = np.percentile(hh, 5), np.percentile(hh, 95) * 1.3
                return lambda y: float(np.clip(0.2 * np.clip(a * y + b, lo, hi), 3, 0.06 * H))
            except (np.linalg.LinAlgError, ValueError):
                pass
        if len(players) >= 1:
            med = float(np.median([b_[3] for b_ in players]))
            return lambda y: float(np.clip(0.2 * med, 3, 0.06 * H))
        return lambda y: max(4.0, 0.012 * H)

    def detect_ball(self, bgr, m, players, size_hint=None, max_out=5):
        """Rank ball-like blobs on a frame: [(box, score)], best first (score 0..1).

        Used for the automatic ball suggestion on the Select screen and for re-finding the
        ball while tracking. Each small non-grass blob inside the field is scored by
          * size      - diameter close to the expected ball size
          * shape     - roughly round and compact (a disk fills ~78 % of its box)
          * colour    - bright and unsaturated (white ball), using its brightest pixels so
                        the dark shadow under the ball does not count against it
          * contrast  - clearly brighter than the grass ring around it
        Blobs in the upper 80 % of a player box (heads, shirts, socks) are rejected.
        """
        H, W = m["fg_raw"].shape
        size_at = (lambda y: float(size_hint)) if size_hint else self.ball_size_model(players, H)
        fg = cv2.morphologyEx(m["fg_raw"], cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
        hsv = m["hsv"]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            d = max(w, h)
            d_exp = size_at(y + h / 2.0)
            if d < 0.45 * d_exp or d > 2.2 * d_exp or area < 4:
                continue
            asp = w / max(h, 1)
            if asp < 0.45 or asp > 2.2:
                continue
            fill = area / float(w * h)
            if fill < 0.35:
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            on_body = False
            for px, py, pw_, ph_ in players:
                if px - 2 <= cx <= px + pw_ + 2 and py - 2 <= cy <= py + 0.8 * ph_:
                    on_body = True
                    break
            if on_body:
                continue
            sel = labels[y:y + h, x:x + w] == i
            # pieces of pitch line that survived line removal lie on (or next to) a line
            rr = int(max(2, 0.5 * d))
            if m["lines"][max(0, y - rr):y + h + rr, max(0, x - rr):x + w + rr].any():
                continue
            v = hsv[y:y + h, x:x + w, 2][sel].astype(np.float32)
            s = hsv[y:y + h, x:x + w, 1][sel].astype(np.float32)
            top = v >= np.percentile(v, 50)
            vb, sb = float(v[top].mean()), float(s[top].mean())
            r = int(max(3, d))
            x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(W, x + w + r), min(H, y + h + r)
            ring = (labels[y0:y1, x0:x1] == 0) & (m["grass"][y0:y1, x0:x1] > 0)
            g_ring = float(gray[y0:y1, x0:x1][ring].mean()) if ring.any() else 100.0
            g_ball = float(np.percentile(gray[y:y + h, x:x + w][sel], 75))
            s_size = 1.0 - min(abs(np.log(d / d_exp)) / np.log(2.4), 1.0)
            s_shape = (1.0 - min(abs(fill - 0.78) / 0.5, 1.0)) * (1.0 - min(abs(np.log(asp)) / np.log(2.2), 1.0))
            s_col = float(np.clip((vb - 110) / 100, 0, 1) * np.clip((130 - sb) / 90, 0, 1))
            s_con = float(np.clip((g_ball - g_ring) / 60, 0, 1))
            score = 0.25 * s_size + 0.2 * s_shape + 0.3 * s_col + 0.25 * s_con
            out.append(([int(x), int(y), int(w), int(h)], round(score, 3)))
        out = _suppress(out)
        out.sort(key=lambda t: -t[1])
        return out[:max_out]

    def detection_overlay(self, bgr, boxes, color=(0, 255, 255)):
        vis = bgr.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        return vis


def _suppress(cands):
    """Drop candidates whose centre lies inside a better-scoring candidate."""
    cands = sorted(cands, key=lambda t: -t[1])
    keep = []
    for b, s in cands:
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        if any(k[0] - 2 <= cx <= k[0] + k[2] + 2 and k[1] - 2 <= cy <= k[1] + k[3] + 2 for k, _ in keep):
            continue
        keep.append((b, s))
    return keep
