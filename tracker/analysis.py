"""Report material: stage images, charts and the pipeline flowchart."""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

STATUS_COLORS = {"tracked": "#2bb673", "redetected": "#3b82f6", "predicted": "#f59e0b", "lost": "#9ca3af"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.bbox": "tight"})


def titled(img, text):
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(out, text, (10, 21), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def grid(images, cols=3, width=480):
    tiles = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h = int(im.shape[0] * width / im.shape[1])
        tiles.append(cv2.resize(im, (width, h)))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack([cv2.copyMakeBorder(t, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                       for t in tiles[i:i + cols]]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def save_stage_images(outdir, frame, enh_stages, enhanced, seg, masks, players, targets_px):
    """Writes the enhancement + segmentation intermediate images. Returns list of (file, caption)."""
    os.makedirs(outdir, exist_ok=True)
    items = []

    def w(name, img, caption):
        p = os.path.join(outdir, name)
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        items.append({"file": name, "caption": caption})

    w("01_original.jpg", frame, "Original frame")
    labels = {"1_denoised": "Stage 1 - Bilateral noise reduction", "2_clahe": "Stage 2 - CLAHE (Lab L-channel)",
              "3_gamma": "Stage 3 - Gamma correction", "4_sharpened": "Stage 4 - Unsharp masking"}
    for i, (k, cap) in enumerate(labels.items()):
        if k in enh_stages:
            w(f"0{i + 2}_{k[2:]}.jpg", enh_stages[k], cap)
    ba = np.hstack([titled(frame, "Before enhancement"), titled(enhanced, "After enhancement")])
    w("06_before_after.jpg", ba, "Before vs after enhancement")

    # hue histogram with the learned grass interval
    fig, ax = plt.subplots(figsize=(7, 2.8))
    hh = seg.hue_hist
    ax.bar(np.arange(180), hh, width=1.0, color="#9aa5b1")
    ax.axvspan(seg.h_lo, seg.h_hi, color="#2bb673", alpha=0.25, label=f"candidate hue band [{seg.h_lo}, {seg.h_hi}]")
    ax.axvline(seg.peak, color="#2bb673", lw=1.5)
    ax.set_xlabel("Hue (OpenCV 0-179)")
    ax.set_ylabel("pixels")
    ax.set_title("Grass candidates: hue histogram (then Lab Mahalanobis model)")
    ax.legend(frameon=False)
    p = os.path.join(outdir, "07_hue_histogram.png")
    fig.savefig(p)
    plt.close(fig)
    items.append({"file": "07_hue_histogram.png", "caption": "Hue histogram and learned grass range"})

    w("08_grass_mask.jpg", masks["grass"], "Grass mask (Lab Mahalanobis colour model)")
    w("09_field_mask.jpg", masks["field"], "Field region (median-filtered grass boundary)")
    w("10_foreground_raw.jpg", masks["fg_raw"], "Foreground = field AND NOT grass")
    w("10b_pitch_lines.jpg", masks["lines"], "Detected pitch markings (removed)")
    w("11_foreground_clean.jpg", masks["fg"], "Lines removed, opening (3x3) + closing (3x7)")
    seg_players = cv2.bitwise_and(enhanced, enhanced, mask=masks["fg"])
    w("12_segmented_players.jpg", seg_players, "Segmented players (foreground pixels only)")
    det = seg.detection_overlay(enhanced, players)
    w("13_detections.jpg", det, f"Player detections after shape filtering ({len(players)} blobs)")

    crops = []
    for name, (x, y, ww, hh_) in targets_px:
        pad = int(0.4 * max(ww, hh_)) + 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(frame.shape[1], x + ww + pad), min(frame.shape[0], y + hh_ + pad)
        c = enhanced[y0:y1, x0:x1]
        m = masks["fg_raw"][y0:y1, x0:x1]
        if c.size == 0:
            continue
        sc = 160 / c.shape[0]
        c = cv2.resize(c, (max(1, int(c.shape[1] * sc)), 160), interpolation=cv2.INTER_NEAREST)
        m = cv2.resize(cv2.bitwise_and(enhanced[y0:y1, x0:x1], enhanced[y0:y1, x0:x1], mask=m),
                       (c.shape[1], 160), interpolation=cv2.INTER_NEAREST)
        crops.append(np.hstack([c, np.full((160, 4, 3), 255, np.uint8), m]))
        crops.append(np.full((160, 12, 3), 255, np.uint8))
    if crops:
        w("14_target_crops.jpg", np.hstack(crops), "Selected targets: crop and segmented crop")
    return items


def chart_processing_time(path, times):
    """times = dict stage -> list of ms per frame."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    keys = [k for k in ["enhance", "segment", "camera", "track"] if k in times]
    n = len(times[keys[0]])
    x = np.arange(n)
    cols = ["#8b5cf6", "#2bb673", "#f59e0b", "#3b82f6"]
    ax.stackplot(x, *[times[k] for k in keys], labels=keys, colors=cols[:len(keys)], alpha=0.85)
    ax.set_xlabel("processed frame")
    ax.set_ylabel("ms / frame")
    ax.set_title("Processing time per frame by stage")
    ax.legend(loc="upper right", frameon=False, ncol=4)
    fig.savefig(path)
    plt.close(fig)


def chart_status(path, targets, frame_times):
    fig, ax = plt.subplots(figsize=(8, 0.7 + 0.55 * len(targets)))
    dt = frame_times[1] - frame_times[0] if len(frame_times) > 1 else 0.04
    for i, t in enumerate(targets):
        st = t["status"]
        start = 0
        for k in range(1, len(st) + 1):
            if k == len(st) or st[k] != st[start]:
                ax.barh(i, frame_times[k - 1] - frame_times[start] + dt, left=frame_times[start],
                        color=STATUS_COLORS.get(st[start], "#999"), height=0.6)
                start = k
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([t["name"] for t in targets])
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    ax.set_title("Detection consistency across frames")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in STATUS_COLORS.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.35 if len(targets) < 3 else -0.15), ncol=4, frameon=False)
    fig.savefig(path)
    plt.close(fig)


def chart_confidence(path, targets, frame_times):
    fig, ax = plt.subplots(figsize=(8, 3))
    for t in targets:
        ax.plot(frame_times, t["conf"], label=t["name"], color=t["color"], lw=1.4)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("appearance similarity")
    ax.set_title("Tracking confidence (Bhattacharyya similarity to the target's kit model)")
    ax.legend(frameon=False, ncol=4)
    fig.savefig(path)
    plt.close(fig)


def chart_speed(path, targets, frame_times):
    fig, ax = plt.subplots(figsize=(8, 3))
    any_ = False
    for t in targets:
        if t.get("speed_kmh") is None:
            continue
        any_ = True
        ax.plot(frame_times, t["speed_kmh"], label=t["name"], color=t["color"], lw=1.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("km/h (approx.)")
    ax.set_title("Estimated speed (1 s smoothing, scale from player height)")
    if any_:
        ax.legend(frameon=False, ncol=4)
    fig.savefig(path)
    plt.close(fig)


def chart_enhancement(path, ev):
    keys = ["brightness", "contrast_rms", "entropy", "sharpness"]
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.8))
    for ax, k in zip(axes, keys):
        ax.bar(["original", "enhanced"], [ev["original"][k], ev["enhanced"][k]], color=["#9aa5b1", "#2bb673"])
        ax.set_title(k.replace("_", " "))
    ax = axes[-1]
    ax.bar(["original", "enhanced"], [ev["detections_original"], ev["detections_enhanced"]], color=["#9aa5b1", "#2bb673"])
    ax.set_title("players detected / frame")
    fig.suptitle("Enhancement effectiveness (mean over sample frames)", y=1.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def flowchart(path):
    steps = [
        ("Input video", "upload, FFmpeg -> H.264 CFR MP4"),
        ("Frame extraction", "every frame / every n-th frame"),
        ("Enhancement", "bilateral -> CLAHE -> gamma -> unsharp"),
        ("Segmentation", "HSV grass model -> field hull -> morphology"),
        ("Detection", "connected components + shape filter"),
        ("Camera motion", "LK flow + RANSAC similarity"),
        ("Tracking", "CSRT + kit histogram + Kalman + re-detection"),
        ("Post-processing", "gap interpolation + Savitzky-Golay"),
        ("Outputs", "video, route map, CSV, charts"),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 9.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps))
    ax.axis("off")
    for i, (t, s) in enumerate(steps):
        y = len(steps) - i - 0.85
        col = "#2bb673" if t in ("Enhancement", "Segmentation", "Tracking") else "#e5e7eb"
        ax.add_patch(FancyBboxPatch((0.08, y), 0.84, 0.66, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=col, ec="#374151", lw=1))
        ax.text(0.5, y + 0.43, t, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(0.5, y + 0.18, s, ha="center", va="center", fontsize=8.2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.12), xytext=(0.5, y), arrowprops=dict(arrowstyle="->", lw=1.3))
    fig.savefig(path)
    plt.close(fig)
