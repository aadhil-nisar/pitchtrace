"""Image enhancement pipeline.

Stages (each one targets a quality problem typical of broadcast football footage):

  1. Noise reduction   - edge-preserving bilateral filter. Strength is chosen from the
                         estimated noise level (Immerkaer's method), skipped for clean video.
  2. Illumination      - CLAHE on the L channel of CIE-Lab. Fixes stadium shadows and
                         uneven floodlight without shifting kit colours.
  3. Gamma correction  - global power-law curve; gamma is computed once per clip so the
                         mean brightness moves toward mid-grey (no frame-to-frame flicker).
  4. Sharpening        - unsharp masking to recover edges lost to motion blur; the amount
                         grows when the clip is blurry (low variance of the Laplacian).

Parameters are estimated once from a reference frame and then kept fixed for the whole
clip, which keeps colours stable for the colour-histogram tracker.
"""
import cv2
import numpy as np


def estimate_noise(gray):
    """Immerkaer (1996) fast noise standard deviation estimate."""
    g = gray.astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(g, -1, kernel)[1:-1, 1:-1]
    h, w = g.shape
    return float(np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


def sharpness(gray):
    """Variance of the Laplacian: low values mean a blurry frame."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def quality_stats(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": round(float(gray.mean()), 2),
        "contrast_rms": round(float(gray.std()), 2),
        "entropy": round(entropy(gray), 3),
        "sharpness": round(sharpness(gray), 1),
        "noise_sigma": round(estimate_noise(gray), 3),
    }


class Enhancer:
    def __init__(self, reference_bgr, enabled=True):
        self.enabled = enabled
        gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        self.noise = estimate_noise(gray)
        self.blur = sharpness(gray)

        # 1. denoise strength from noise level
        self.denoise = self.noise > 1.5
        self.bilateral_sigma = float(np.clip(8 + 6 * self.noise, 10, 60))

        # 2. CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 3. gamma: output = input^gamma with mean^gamma = 0.5  ->  gamma = ln(0.5)/ln(mean)
        #    (gamma < 1 brightens a dark clip, gamma > 1 darkens a washed-out clip).
        #    Only a partial correction is applied (halfway in log space) to stay natural.
        mean = float(np.clip(gray.mean() / 255.0, 0.05, 0.95))
        g = np.log(0.5) / np.log(mean)
        self.gamma = float(np.clip(np.sqrt(g), 0.7, 1.4))
        self.lut = np.array([((i / 255.0) ** self.gamma) * 255 for i in range(256)], np.uint8)

        # 4. unsharp amount: more sharpening for blurrier footage
        self.sharp_amount = float(np.clip(0.4 + 300.0 / (self.blur + 300.0), 0.4, 1.2))
        self.sharp_sigma = 1.2

    def params(self):
        return {
            "enabled": self.enabled,
            "noise_sigma": round(self.noise, 3),
            "denoise_applied": bool(self.denoise),
            "bilateral": {"d": 5, "sigmaColor": round(self.bilateral_sigma, 1), "sigmaSpace": 5},
            "clahe": {"clipLimit": 2.0, "tileGridSize": [8, 8], "space": "Lab L-channel"},
            "gamma": round(self.gamma, 3),
            "unsharp": {"sigma": self.sharp_sigma, "amount": round(self.sharp_amount, 3)},
            "laplacian_variance": round(self.blur, 1),
        }

    def enhance(self, bgr, stages=False):
        if not self.enabled:
            return (bgr, {}) if stages else bgr
        out = {}
        x = bgr
        if self.denoise:
            x = cv2.bilateralFilter(x, 5, self.bilateral_sigma, 5)
        out["1_denoised"] = x
        lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge((self.clahe.apply(l), a, b))
        x = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        out["2_clahe"] = x
        x = cv2.LUT(x, self.lut)
        out["3_gamma"] = x
        blurred = cv2.GaussianBlur(x, (0, 0), self.sharp_sigma)
        x = cv2.addWeighted(x, 1 + self.sharp_amount, blurred, -self.sharp_amount, 0)
        out["4_sharpened"] = x
        return (x, out) if stages else x
