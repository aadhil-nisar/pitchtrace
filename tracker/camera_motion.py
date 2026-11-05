"""Camera-motion compensation.

Broadcast cameras pan and zoom constantly, so a player's pixel position mixes their own
movement with camera movement. We estimate the frame-to-frame camera transform from
background feature points and remove it, so routes are drawn in a stable "reference"
coordinate system (the first tracked frame).

  * Shi-Tomasi corners are detected on the previous frame, with tracked players masked out
    (they move independently of the camera).
  * Pyramidal Lucas-Kanade optical flow finds the corners in the current frame.
  * A similarity transform is fitted with RANSAC (rejecting remaining moving points as
    outliers); only zoom (uniform scale) and pan/tilt (translation) are kept.
  * Transforms are chained:  M_t = F_t . M_(t-1),  M_t maps reference coords -> frame t.
"""
import cv2
import numpy as np


class CameraMotion:
    def __init__(self):
        self.prev = None
        self.prev_mask = None
        self.M = np.eye(3)
        self.last_inliers = 0
        self.static_mask = None  # 0 where burnt-in graphics are (score bug, logo, watermark)

    def update(self, gray, exclude_boxes=()):
        mask = np.full(gray.shape, 255, np.uint8) if self.static_mask is None else self.static_mask.copy()
        for x, y, w, h in exclude_boxes:
            pad = int(0.3 * max(w, h))
            cv2.rectangle(mask, (int(x - pad), int(y - pad)), (int(x + w + pad), int(y + h + pad)), 0, -1)
        if self.prev is None:
            self.prev, self.prev_mask = gray, mask
            return self.M
        F = np.eye(3)
        pts = cv2.goodFeaturesToTrack(self.prev, maxCorners=500, qualityLevel=0.01,
                                      minDistance=7, mask=self.prev_mask, blockSize=7)
        if pts is not None and len(pts) >= 10:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, pts, None,
                                                  winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            p0, p1 = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
            if len(p0) >= 8:
                A, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.0, maxIters=2000)
                if A is not None:
                    scale = np.sqrt(abs(np.linalg.det(A[:, :2])))
                    self.last_inliers = int(inl.sum()) if inl is not None else 0
                    if 0.9 < scale < 1.1 and self.last_inliers >= 6:
                        # Broadcast cameras pan, tilt and zoom but practically never roll,
                        # so the rotation is dropped (it would otherwise accumulate as drift)
                        # and the translation is re-fitted on the RANSAC inliers.
                        m = inl.ravel().astype(bool)
                        t = (p1[m] - scale * p0[m]).mean(axis=0)
                        F = np.array([[scale, 0, t[0]], [0, scale, t[1]], [0, 0, 1]])
        self.M = F @ self.M
        self.prev, self.prev_mask = gray, mask
        return self.M


def apply(M, pts):
    """Apply a 3x3 affine matrix to an (N,2) array of points."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return pts @ M[:2, :2].T + M[:2, 2]


def scale_matrix(M, s):
    """Convert a transform estimated at processing resolution to original resolution."""
    S = np.diag([s, s, 1.0])
    return S @ M @ np.linalg.inv(S)
