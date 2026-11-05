"""Drawing helpers: boxes, labels, routes and the stitched route map."""
import cv2
import numpy as np

from .camera_motion import apply


def hex_to_bgr(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _text_color(bgr):
    lum = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    return (20, 20, 20) if lum > 140 else (255, 255, 255)


def label(img, text, x, y, color, scale=0.55):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    pad = 5
    x = int(max(0, min(x, img.shape[1] - tw - 2 * pad)))
    y = int(max(th + 2 * pad, y))
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y), color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + pad, y - pad), font, scale, _text_color(color), 1, cv2.LINE_AA)


def dashed_rect(img, p1, p2, color, thickness=2, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for (a, b) in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        a, b = np.array(a, float), np.array(b, float)
        L = np.linalg.norm(b - a)
        n = max(1, int(L // dash))
        for i in range(0, n, 2):
            s = a + (b - a) * (i / n)
            e = a + (b - a) * (min(i + 1, n) / n)
            cv2.line(img, tuple(s.astype(int)), tuple(e.astype(int)), color, thickness, cv2.LINE_AA)


def draw_route(img, pts, color, thickness=3):
    """Route with a dark outline, older parts thinner and more transparent."""
    if len(pts) < 2:
        return
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    overlay = img.copy()
    p = pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [p], False, (15, 15, 15), thickness + 3, cv2.LINE_AA)
    cv2.polylines(overlay, [p], False, color, thickness, cv2.LINE_AA)
    # older route faded, the most recent ~1/3 fully opaque
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    recent = p[int(n * 0.66):]
    if len(recent) >= 2:
        cv2.polylines(img, [recent], False, color, thickness, cv2.LINE_AA)


def draw_target(img, box, color, name, status, kind):
    x, y, w, h = [int(v) for v in box]
    if kind == "ball":
        c = (x + w // 2, y + h // 2)
        r = int(max(w, h) * 0.8) + 4
        if status == "predicted":
            cv2.circle(img, c, r, color, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, r + 2, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.circle(img, c, r, color, 2, cv2.LINE_AA)
        label(img, name if status != "predicted" else f"{name} (hidden)", c[0] + r + 4, c[1] - r, color)
        return
    if status == "predicted":
        dashed_rect(img, (x, y), (x + w, y + h), color, 2)
        label(img, f"{name} (occluded)", x, y - 4, color)
    else:
        cv2.rectangle(img, (x - 1, y - 1), (x + w + 1, y + h + 1), (15, 15, 15), 4, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        label(img, name, x, y - 4, color)
    # ground marker (where the route is measured from)
    gx, gy = x + w // 2, y + h
    cv2.ellipse(img, (gx, gy), (max(6, w // 2), max(3, w // 6)), 0, 0, 360, color, 2, cv2.LINE_AA)


def legend(img, items):
    """items = [(name, bgr_color)]"""
    font = cv2.FONT_HERSHEY_DUPLEX
    lh = 24
    wmax = max([cv2.getTextSize(n, font, 0.55, 1)[0][0] for n, _ in items] + [60])
    box_w, box_h = wmax + 50, lh * len(items) + 14
    ov = img.copy()
    cv2.rectangle(ov, (12, 12), (12 + box_w, 12 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
    for i, (n, c) in enumerate(items):
        y = 12 + 7 + lh * i + lh // 2
        cv2.line(img, (24, y), (48, y), c, 4, cv2.LINE_AA)
        cv2.putText(img, n, (56, y + 6), font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


class RouteMap:
    """Stitches frames into one reference image (mosaic) and draws the complete routes."""

    def __init__(self, Ms, W, H, max_side=3200):
        corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
        pts = [apply(np.linalg.inv(M), corners) for M in Ms]
        allp = np.vstack(pts)
        lo = np.maximum(allp.min(axis=0), [-2.0 * W, -2.0 * H])
        hi = np.minimum(allp.max(axis=0), [3.0 * W, 3.0 * H])
        size = hi - lo
        self.sc = float(min(1.0, max_side / max(size.max(), 1)))
        self.off = lo
        self.cw, self.ch = int(size[0] * self.sc) + 1, int(size[1] * self.sc) + 1
        self.canvas = np.zeros((self.ch, self.cw, 3), np.uint8)
        self.filled = np.zeros((self.ch, self.cw), np.uint8)

    def ref_to_canvas(self, pts):
        return (np.asarray(pts, np.float64) - self.off) * self.sc

    def add_frame(self, frame, M):
        T = np.array([[self.sc, 0, -self.off[0] * self.sc], [0, self.sc, -self.off[1] * self.sc], [0, 0, 1]])
        P = (T @ np.linalg.inv(M))[:2]
        warped = cv2.warpAffine(frame, P, (self.cw, self.ch))
        m = cv2.warpAffine(np.full(frame.shape[:2], 255, np.uint8), P, (self.cw, self.ch))
        m = cv2.erode(m, np.ones((5, 5), np.uint8))
        new = (m > 0) & (self.filled == 0)
        self.canvas[new] = warped[new]
        self.filled[new] = 255

    def render(self, routes):
        """routes = [(name, bgr, ref_points (N,2) or None)]"""
        img = self.canvas.copy()
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 0.55, gray, 0.25, 0)
        img[self.filled == 0] = (28, 28, 28)
        th = max(3, int(round(4 * max(self.cw, self.ch) / 1600)))
        for name, col, pts in routes:
            if pts is None or len(pts) < 2:
                continue
            c = self.ref_to_canvas(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [c], False, (10, 10, 10), th + 4, cv2.LINE_AA)
            cv2.polylines(img, [c], False, col, th, cv2.LINE_AA)
            s, e = tuple(c[0, 0]), tuple(c[-1, 0])
            cv2.circle(img, s, th * 3, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(img, s, th * 3 - 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, s, th * 2 - 1, col, -1, cv2.LINE_AA)
            # arrow head at the end
            if len(c) > 6:
                p0 = c[-6, 0].astype(float)
                d = np.array(e, float) - p0
                if np.linalg.norm(d) > 1:
                    d /= np.linalg.norm(d)
                    nrm = np.array([-d[1], d[0]])
                    L = th * 5
                    tip = np.array(e, float) + d * L * 0.5
                    pa = tip - d * L + nrm * L * 0.6
                    pb = tip - d * L - nrm * L * 0.6
                    tri = np.array([tip, pa, pb], np.int32)
                    cv2.fillConvexPoly(img, tri, col, cv2.LINE_AA)
                    cv2.polylines(img, [tri], True, (10, 10, 10), 2, cv2.LINE_AA)
            label(img, name, e[0] + 10, e[1] - 10, col, 0.6)
        legend(img, [(n, c) for n, c, p in routes] + [("o = start   > = end", (200, 200, 200))])
        return img
