"""Video preparation: probing, format conversion, frame access and sampling.

Every uploaded clip is normalised with FFmpeg into a constant-frame-rate H.264 MP4
(max 1080p, no audio). This removes the problems of YouTube downloads
(variable frame rate, VP9/AV1 codecs OpenCV may not decode, inaccurate seeking).
"""
import os
import shutil
import subprocess

import cv2
import numpy as np

MAX_HEIGHT = 1080


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def probe(path):
    """Return basic metadata for a video, or None if OpenCV cannot read it."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 1 or fps > 240:
        fps = 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:  # some containers do not store a frame count: count manually
        n = 1
        while cap.grab():
            n += 1
    h, w = frame.shape[:2]
    cap.release()
    return {"width": w, "height": h, "fps": round(float(fps), 3), "frames": n,
            "duration": round(n / fps, 2)}


def normalise(src, dst, log=None, start=None, end=None):
    """Convert any input video into a seek-friendly H.264 MP4 (format conversion step).

    start / end (seconds) cut the chosen part of the video at the same time.
    """
    if not ffmpeg_available():
        shutil.copyfile(src, dst)
        return probe(dst)
    cut_in, cut_out = [], []
    if start is not None and start > 0:
        cut_in = ["-ss", f"{start:.3f}"]
    if end is not None and end > (start or 0):
        cut_out = ["-t", f"{end - (start or 0):.3f}"]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", *cut_in, "-i", src, *cut_out,
        "-an", "-vf", f"scale=-2:'min({MAX_HEIGHT},ih)'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-movflags", "+faststart", dst,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(dst):
        # fall back to the original file if conversion is impossible
        if log:
            log(f"ffmpeg conversion failed: {res.stderr[-400:]}")
        shutil.copyfile(src, dst)
    return probe(dst)


def read_frame(path, index):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def iter_frames(path, start, end, step=1):
    """Yield (frame_index, frame) from start..end (inclusive) keeping every `step`-th frame.

    Frame sampling strategy: step=1 processes every frame (best accuracy),
    step=2/3 processes every 2nd/3rd frame (faster, larger motion between samples).
    """
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    idx = int(start)
    while idx <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % step == 0:
            yield idx, frame
        idx += 1
    cap.release()


class H264Writer:
    """Writes browser-playable MP4 by piping raw frames into FFmpeg (fallback: OpenCV)."""

    def __init__(self, path, width, height, fps):
        self.path = path
        self.proc = None
        self.cv = None
        if ffmpeg_available():
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.cv:
            self.cv.release()
