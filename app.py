"""Football Route Tracker - web application.

Run locally:   python app.py            (http://localhost:7860)
Production:    gunicorn -w 1 --threads 8 -b 0.0.0.0:$PORT app:app
"""
import io
import json
import os
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from tracker import pipeline
from tracker.segmentation import PitchSegmenter
from tracker.enhancement import Enhancer
from tracker.shots import analyse
from tracker.video_io import normalise, read_frame

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(BASE, "data"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
MAX_SECONDS = float(os.environ.get("MAX_TRACK_SECONDS", "120"))  # longest range tracked per job
MAX_CLIP_SECONDS = float(os.environ.get("MAX_CLIP_SECONDS", "300"))  # longest clip cut from a video / link
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "24"))
BALL_SUGGEST_MIN = 0.6  # minimum ball-likeness score for an automatic ball suggestion
WORKERS = int(os.environ.get("WORKERS", "1"))
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".3gp"}

os.makedirs(DATA, exist_ok=True)
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
executor = ThreadPoolExecutor(max_workers=WORKERS)
analysis_pool = ThreadPoolExecutor(max_workers=2)  # camera-cut + overlay scan after upload
lock = threading.Lock()
state = {}  # job_id -> runtime status (progress etc.)
prep = {}  # job_id -> status of a video being fetched from a link


# ---------------------------------------------------------------- helpers
def jdir(jid):
    if not jid or not all(c in "0123456789abcdef" for c in jid):
        abort(404)
    d = os.path.join(DATA, jid)
    if not os.path.isdir(d):
        abort(404)
    return d


def load_job(jid):
    with open(os.path.join(jdir(jid), "job.json")) as f:
        return json.load(f)


def save_job(jid, job):
    p = os.path.join(DATA, jid, "job.json")
    with open(p + ".tmp", "w") as f:
        json.dump(job, f)
    os.replace(p + ".tmp", p)


def set_state(jid, **kw):
    with lock:
        state.setdefault(jid, {}).update(kw)


def cleanup_old():
    now = time.time()
    for d in os.listdir(DATA):
        p = os.path.join(DATA, d)
        try:
            if os.path.isdir(p) and now - os.path.getmtime(p) > JOB_TTL_HOURS * 3600 and \
                    state.get(d, {}).get("status") != "running":
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def jpeg(img, q=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE, "static"), "index.html")


@app.errorhandler(413)
def too_big(_):
    return jsonify(error=f"That file is larger than {MAX_UPLOAD_MB} MB. Please trim the clip first."), 413


# ---------------------------------------------------------------- API
def _times(form):
    """Read optional start / end seconds and check the clip length."""
    def num(k):
        v = form.get(k)
        if v in (None, "", "null"):
            return None
        try:
            return max(0.0, float(v))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {k} time.")
    start, end = num("start"), num("end")
    if start is not None and end is not None:
        if end - start < 1:
            raise ValueError("The end time must be at least 1 second after the start time.")
        if end - start > MAX_CLIP_SECONDS + 0.5:
            raise ValueError(f"Please choose a part of at most {int(MAX_CLIP_SECONDS // 60)} minutes.")
    return start, end


def _start_scan(d):
    def scan():
        try:
            analyse(os.path.join(d, "input.mp4"), os.path.join(d, "analysis.tmp.json"), os.path.join(d, "overlay.png"))
            os.replace(os.path.join(d, "analysis.tmp.json"), os.path.join(d, "analysis.json"))
        except Exception:  # analysis is optional: tracking still detects cuts itself
            traceback.print_exc()
    analysis_pool.submit(scan)


@app.post("/api/upload")
def upload():
    cleanup_old()
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="No video received."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error="Please upload a video file (MP4, MOV, AVI, MKV or WebM)."), 400
    try:
        start, end = _times(request.form)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    jid = uuid.uuid4().hex
    d = os.path.join(DATA, jid)
    os.makedirs(d)
    raw = os.path.join(d, "upload" + ext)
    f.save(raw)
    meta = normalise(raw, os.path.join(d, "input.mp4"), start=start, end=end)
    if not meta:
        shutil.rmtree(d, ignore_errors=True)
        return jsonify(error="This video could not be read. Try exporting it as MP4 (H.264)."), 400
    os.remove(raw)
    meta["source"] = {"type": "upload", "file": f.filename, "start": start, "end": end}
    job = {"id": jid, "name": f.filename, "meta": meta, "created": time.time(), "runs": 0}
    save_job(jid, job)
    _start_scan(d)
    return jsonify(id=jid, name=f.filename, meta=meta, max_seconds=MAX_SECONDS)


# ---------------------------------------------------------------- video from a link (YouTube)
def _check_url(url):
    url = (url or "").strip()
    if not url.lower().startswith(("http://", "https://")) or len(url) > 2000:
        raise ValueError("Please paste a full link starting with https://")
    return url


def _ydl_error(e):
    msg = str(e)
    if "Sign in to confirm" in msg or "bot" in msg.lower():
        return ("YouTube refused the download from this server (it blocks many cloud servers). "
                "Run the site on your own computer, or download the clip and upload it as a file.")
    if "Private video" in msg or "unavailable" in msg.lower():
        return "This video is private or unavailable."
    if "403" in msg or "Unable to download" in msg or "urlopen" in msg or "timed out" in msg.lower():
        return ("This server could not reach the video (YouTube refused the connection or the network blocks it). "
                "Run the site on your own computer, or download the clip and use 'Upload a video'.")
    if "Unsupported URL" in msg:
        return "That link is not a video page. Copy the link of a YouTube video."
    return "Could not read that link. " + msg.split("\n")[0][-200:]


@app.post("/api/link/info")
def link_info():
    try:
        import yt_dlp
    except ImportError:
        return jsonify(error="YouTube support is not installed on this server (pip install yt-dlp)."), 500
    try:
        url = _check_url((request.get_json(silent=True) or {}).get("url"))
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True,
                               "skip_download": True}) as y:
            info = y.extract_info(url, download=False)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:  # yt-dlp raises many kinds of errors
        return jsonify(error=_ydl_error(e)), 400
    if info.get("_type") == "playlist":
        return jsonify(error="That is a playlist. Open one video and copy its link."), 400
    if info.get("is_live"):
        return jsonify(error="Live streams cannot be used. Wait until the recording is available."), 400
    is_yt = info.get("extractor_key", "").lower().startswith("youtube")
    return jsonify(url=url, title=info.get("title") or "Video", duration=info.get("duration"),
                   uploader=info.get("uploader") or info.get("channel"), thumbnail=info.get("thumbnail"),
                   youtube_id=info.get("id") if is_yt else None, height=info.get("height"),
                   webpage_url=info.get("webpage_url") or url, max_clip=MAX_CLIP_SECONDS)


@app.post("/api/link/fetch")
def link_fetch():
    body = request.get_json(silent=True) or {}
    try:
        url = _check_url(body.get("url"))
        start, end = _times(body)
        if start is None or end is None:
            raise ValueError("Choose a start and an end time.")
    except ValueError as e:
        return jsonify(error=str(e)), 400
    cleanup_old()
    jid = uuid.uuid4().hex
    d = os.path.join(DATA, jid)
    os.makedirs(d)
    title = str(body.get("title") or "YouTube clip")[:120]
    prep[jid] = {"status": "running", "stage": "Connecting", "progress": 0}

    def hook(dl):
        if dl.get("status") == "downloading":
            tot = dl.get("total_bytes") or dl.get("total_bytes_estimate")
            pct = 100.0 * dl.get("downloaded_bytes", 0) / tot if tot else None
            prep[jid].update(stage="Downloading the selected part",
                             progress=round(min(pct, 99), 1) if pct is not None else prep[jid].get("progress", 0))

    def work():
        try:
            import yt_dlp
            from yt_dlp.utils import download_range_func
            opts = {
                "outtmpl": os.path.join(d, "source.%(ext)s"), "quiet": True, "no_warnings": True,
                "noplaylist": True, "progress_hooks": [hook],
                "format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b[height<=1080]/bv*/b",
                "download_ranges": download_range_func(None, [(start, end)]),
                "force_keyframes_at_cuts": True,
            }
            with yt_dlp.YoutubeDL(opts) as y:
                y.download([url])
            src = next((os.path.join(d, f) for f in os.listdir(d) if f.startswith("source.")
                        and not f.endswith((".part", ".ytdl"))), None)
            if not src:
                raise RuntimeError("The download produced no video.")
            prep[jid].update(stage="Converting to a standard format", progress=100)
            meta = normalise(src, os.path.join(d, "input.mp4"))
            if not meta:
                raise RuntimeError("The downloaded video could not be read.")
            os.remove(src)
            meta["source"] = {"type": "link", "url": url, "title": title, "start": start, "end": end}
            name = f"{title} ({_mmss(start)}-{_mmss(end)})"
            save_job(jid, {"id": jid, "name": name, "meta": meta, "created": time.time(), "runs": 0})
            _start_scan(d)
            prep[jid].update(status="done", stage="Ready")
        except Exception as e:
            traceback.print_exc()
            prep[jid].update(status="error", error=_ydl_error(e) if "yt_dlp" in type(e).__module__ else str(e))
            shutil.rmtree(d, ignore_errors=True)

    analysis_pool.submit(work)
    return jsonify(id=jid)


@app.get("/api/link/<jid>/status")
def link_status(jid):
    if jid not in prep:
        abort(404)
    return jsonify(prep[jid])


def _mmss(t):
    t = int(round(t or 0))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


@app.get("/api/jobs/<jid>")
def job_info(jid):
    job = load_job(jid)
    st = state.get(jid, {})
    ap = os.path.join(DATA, jid, "analysis.json")
    analysis = json.load(open(ap)) if os.path.exists(ap) else None
    return jsonify(id=jid, name=job["name"], meta=job["meta"], max_seconds=MAX_SECONDS, analysis=analysis,
                   has_result=os.path.exists(os.path.join(DATA, jid, "output", "metrics.json")) and
                   st.get("status") != "running",
                   status=st.get("status"), targets=job.get("targets"), settings=job.get("settings"))


@app.get("/api/jobs/<jid>/frame/<int:idx>")
def frame(jid, idx):
    d = jdir(jid)
    img = read_frame(os.path.join(d, "input.mp4"), idx)
    if img is None:
        abort(404)
    return jpeg(img, 88)


@app.get("/api/jobs/<jid>/detect/<int:idx>")
def detect(jid, idx):
    """Auto-detect players and the ball on a frame (enhancement + segmentation) for click-to-select."""
    d = jdir(jid)
    img = read_frame(os.path.join(d, "input.mp4"), idx)
    if img is None:
        abort(404)
    H, W = img.shape[:2]
    pw = min(pipeline.PROC_WIDTH, W)
    s = W / pw
    small = cv2.resize(img, (pw, int(round(H / s))), interpolation=cv2.INTER_AREA) if pw != W else img
    enh = Enhancer(small).enhance(small)
    seg = PitchSegmenter(enh)
    m = seg.masks(enh)
    boxes = seg.detect_players(m["fg"], small.shape[0], pw)
    out = [[int(round(v * s)) for v in b] for b in boxes]
    # ball suggestions: only confident ones (a wrong suggestion is worse than none; the
    # ball can always be selected by drawing a box)
    balls = [{"box": [int(round(v * s)) for v in b], "score": float(sc)}
             for b, sc in seg.detect_ball(enh, m, boxes, max_out=3) if sc >= BALL_SUGGEST_MIN]
    return jsonify(players=out, balls=balls)


@app.post("/api/jobs/<jid>/track")
def track(jid):
    d = jdir(jid)
    job = load_job(jid)
    if state.get(jid, {}).get("status") == "running":
        return jsonify(error="This video is already being tracked."), 409
    body = request.get_json(force=True, silent=True) or {}
    meta = job["meta"]
    targets = []
    for i, t in enumerate(body.get("targets", [])[:12]):
        try:
            box = [int(round(float(v))) for v in t["box"]]
        except (KeyError, TypeError, ValueError):
            continue
        if box[2] < 3 or box[3] < 3:
            continue
        color = str(t.get("color", "#ff3b30"))
        if not (color.startswith("#") and len(color) in (4, 7)):
            color = "#ff3b30"
        targets.append({"id": f"t{i + 1}", "name": str(t.get("name") or f"Target {i + 1}")[:24],
                        "kind": "ball" if t.get("kind") == "ball" else "player", "color": color, "box": box})
    if not targets:
        return jsonify(error="Select at least one player or the ball first."), 400
    start = max(0, min(int(body.get("start", 0)), meta["frames"] - 1))
    end = int(body.get("end", meta["frames"] - 1))
    end = max(start + 1, min(end, meta["frames"] - 1, start + int(MAX_SECONDS * meta["fps"])))
    settings = {"start": start, "end": end, "step": int(body.get("step", 1)) if body.get("step") in (1, 2, 3) else 1,
                "enhance": bool(body.get("enhance", True)),
                "trail": "recent" if body.get("trail") == "recent" else "full"}
    job.update(targets=targets, settings=settings, runs=job.get("runs", 0) + 1)
    save_job(jid, job)
    shutil.rmtree(os.path.join(d, "output"), ignore_errors=True)
    set_state(jid, status="queued", stage="Waiting for a free worker", progress=0, eta=None, error=None,
              cancel=False, started=time.time())

    def progress(stage, pct, eta=None):
        set_state(jid, status="running", stage=stage, progress=round(pct, 1),
                  eta=None if eta is None else round(eta))

    def work():
        set_state(jid, status="running", stage="Starting", progress=0)
        try:
            pipeline.run(d, os.path.join(d, "input.mp4"), meta, settings,
                         [dict(t) for t in targets], progress, lambda: state[jid].get("cancel"))
            set_state(jid, status="done", stage="Done", progress=100, eta=0)
        except pipeline.Cancelled:
            set_state(jid, status="cancelled", stage="Cancelled", progress=0)
        except Exception as e:  # report a readable error to the page
            traceback.print_exc()
            set_state(jid, status="error", stage="Failed", error=str(e) or e.__class__.__name__)

    executor.submit(work)
    return jsonify(ok=True)


@app.get("/api/jobs/<jid>/status")
def status(jid):
    jdir(jid)
    st = dict(state.get(jid, {}))
    st.pop("cancel", None)
    if not st:
        done = os.path.exists(os.path.join(DATA, jid, "output", "metrics.json"))
        st = {"status": "done" if done else "idle", "progress": 100 if done else 0}
    if st.get("status") == "queued":
        ahead = sum(1 for k, v in state.items() if v.get("status") in ("running", "queued")
                    and v.get("started", 0) < st.get("started", 0))
        st["stage"] = f"Waiting in queue ({ahead} ahead)" if ahead else "Starting"
    return jsonify(st)


@app.post("/api/jobs/<jid>/cancel")
def cancel(jid):
    jdir(jid)
    set_state(jid, cancel=True)
    return jsonify(ok=True)


@app.get("/api/jobs/<jid>/result")
def result(jid):
    d = jdir(jid)
    p = os.path.join(d, "output", "metrics.json")
    if not os.path.exists(p):
        abort(404)
    with open(p) as f:
        m = json.load(f)
    sp = os.path.join(d, "output", "spot.json")
    m["spot"] = json.load(open(sp)) if os.path.exists(sp) else []
    vp = os.path.join(d, "output", "verification.json")
    m["verification"] = json.load(open(vp)) if os.path.exists(vp) else None
    m["job"] = {"name": load_job(jid)["name"]}
    return jsonify(m)


@app.post("/api/jobs/<jid>/verify")
def verify(jid):
    """Store the user's spot-check answers -> tracking accuracy measured by a human."""
    d = jdir(jid)
    body = request.get_json(force=True, silent=True) or {}
    marks = body.get("marks", {})
    metrics = json.load(open(os.path.join(d, "output", "metrics.json")))
    summary = []
    for t in metrics["targets"]:
        vals = [v for k, v in marks.items() if k.endswith("|" + t["id"]) and v in ("ok", "bad")]
        n_ok = vals.count("ok")
        summary.append({"id": t["id"], "name": t["name"], "checked": len(vals), "correct": n_ok,
                        "accuracy_pct": round(100 * n_ok / len(vals), 1) if vals else None})
    out = {"marks": marks, "summary": summary, "saved": time.time()}
    with open(os.path.join(d, "output", "verification.json"), "w") as f:
        json.dump(out, f, indent=2)
    return jsonify(out)


@app.get("/api/jobs/<jid>/files/<path:name>")
def files(jid, name):
    d = os.path.join(jdir(jid), "output")
    full = os.path.realpath(os.path.join(d, name))
    if not full.startswith(os.path.realpath(d) + os.sep) or not os.path.isfile(full):
        abort(404)
    dl = request.args.get("download")
    if dl:
        return send_file(full, as_attachment=True, download_name=os.path.basename(dl))
    return send_file(full, conditional=True)


@app.get("/api/jobs/<jid>/report.zip")
def report_zip(jid):
    d = os.path.join(jdir(jid), "output")
    if not os.path.isdir(d):
        abort(404)
    base = os.path.splitext(load_job(jid)["name"])[0][:40] or "match"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(d):
            for fn in fs:
                if fn.endswith(".tmp"):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, d)
                z.write(full, rel, compress_type=zipfile.ZIP_STORED if fn.endswith(".mp4") else zipfile.ZIP_DEFLATED)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"{base}_tracking_results.zip")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), threaded=True)
