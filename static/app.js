(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const PALETTE = ["#ff3b30", "#ffd60a", "#00c7ff", "#ff2dd1", "#7cff4f", "#ff8a00", "#ffffff", "#8b5cf6"];
  const api = (p) => `/api/jobs/${S.id}${p}`;

  const S = {
    id: null, meta: null, maxSeconds: 120,
    frame: 0, end: 0, step: 1, trail: "full",
    img: null, players: [], balls: [], targets: [], mode: "click",
    hover: -1, hoverBall: -1, drag: null, detectCtl: null, poll: null, nextId: 1, cuts: null, cutPoll: null,
  };

  // ------------------------------------------------------------ helpers
  function toast(msg, ms = 2600) {
    const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
    clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), ms);
  }
  function fmt(sec) {
    const m = Math.floor(sec / 60), s = sec - m * 60;
    return `${m}:${s.toFixed(1).padStart(4, "0")}`;
  }
  function dur(sec) {
    if (sec == null) return "";
    sec = Math.max(0, Math.round(sec));
    return sec < 60 ? `${sec}s` : `${Math.floor(sec / 60)}m ${sec % 60}s`;
  }
  function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  async function getJSON(url, opt) {
    const r = await fetch(url, opt);
    let j = null; try { j = await r.json(); } catch (e) { /* ignore */ }
    if (!r.ok) throw new Error((j && j.error) || `Request failed (${r.status})`);
    return j;
  }
  function show(view) {
    ["upload", "select", "track", "results"].forEach((v) => $("view-" + v).classList.toggle("hidden", v !== view));
    const order = ["upload", "select", "track", "results"], i = order.indexOf(view);
    document.querySelectorAll("#steps li").forEach((li, k) => {
      li.classList.toggle("on", k === i); li.classList.toggle("done", k < i);
    });
    window.scrollTo({ top: 0 });
  }
  function setHash() { if (S.id) history.replaceState(null, "", "#job-" + S.id); }
  function iou(a, b) {
    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
    const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return i / (a[2] * a[3] + b[2] * b[3] - i || 1);
  }

  // ------------------------------------------------------------ time helpers
  function parseTime(v) {
    v = String(v || "").trim().replace(",", ".");
    if (!v) return null;
    if (!/^\d+(:\d{1,2}){0,2}(\.\d+)?$/.test(v)) return NaN;
    return v.split(":").reduce((acc, p) => acc * 60 + parseFloat(p), 0);
  }
  function clock(t) {
    if (t == null || isNaN(t)) return "";
    t = Math.max(0, t);
    const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), sec = Math.floor(t % 60);
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
  }

  /* Start/end selector: two time boxes ("Use current time" buttons when a player is
     available) and, when the length is known, a two-handle slider. */
  function makeRange(el, o) {
    const dur = o.duration && isFinite(o.duration) ? o.duration : null, max = o.max;
    let s = 0, e = dur ? Math.min(dur, o.defaultLen || 60) : null;
    el.innerHTML = `
      <div class="times ${o.getCurrent ? "" : "nobtn"}">
        <label>Start</label><input class="t" data-k="s" inputmode="decimal" placeholder="0:00" aria-label="Start time">
        <button type="button" data-now="s" class="${o.getCurrent ? "" : "hidden"}">Use current time</button>
        <label>End</label><input class="t" data-k="e" inputmode="decimal" placeholder="${dur ? clock(dur) : "end of video"}" aria-label="End time">
        <button type="button" data-now="e" class="${o.getCurrent ? "" : "hidden"}">Use current time</button>
      </div>
      ${dur ? `<div class="dual"><div class="track"></div><div class="sel"></div>
        <input type="range" data-k="s" min="0" max="${dur}" step="0.1" aria-label="Start">
        <input type="range" data-k="e" min="0" max="${dur}" step="0.1" aria-label="End"></div>` : ""}
      <div class="span-info"><span class="len"></span><span class="warn"></span></div>`;
    const tS = el.querySelector('.t[data-k="s"]'), tE = el.querySelector('.t[data-k="e"]');
    const rS = el.querySelector('input[type=range][data-k="s"]'), rE = el.querySelector('input[type=range][data-k="e"]');
    function problem() {
      if (isNaN(s) || (e !== null && isNaN(e))) return "Use a time like 1:30 or 0:01:30.";
      if (dur && s >= dur) return `Start must be before the end of the video (${clock(dur)}).`;
      if (e !== null && e - s < 1) return "End must be at least 1 second after start.";
      if (e !== null && e - s > max) return `Choose at most ${clock(max)} (the site tracks up to ${clock(o.trackMax)} at a time).`;
      if (e === null && dur && dur - s > max) return `Choose an end time: at most ${clock(max)} can be used.`;
      if (e === null && o.requireEnd) return "Choose an end time.";
      return "";
    }
    function paint(fromBox) {
      if (!fromBox) { tS.value = clock(s); tE.value = e === null ? "" : clock(e); }
      if (rS) {
        rS.value = s; rE.value = e === null ? dur : e;
        const a = (s / dur) * 100, b = ((e === null ? dur : e) / dur) * 100;
        el.querySelector(".sel").style.cssText = `left:${a}%;width:${Math.max(0, b - a)}%`;
      }
      const len = (e === null ? (dur || NaN) : e) - s;
      el.querySelector(".len").innerHTML = isNaN(len) ? "From the start time to the end of the video"
        : `Selected <b>${clock(len)}</b> (${clock(s)} → ${clock(e === null ? dur : e)})`;
      const pr = problem();
      el.querySelector(".warn").textContent = pr;
      tS.classList.toggle("bad", isNaN(s)); tE.classList.toggle("bad", e !== null && isNaN(e));
      o.onChange && o.onChange(!pr);
    }
    [tS, tE].forEach((inp) => inp.addEventListener("input", () => {
      const v = parseTime(inp.value);
      if (inp === tS) s = v === null ? 0 : v; else e = v;
      paint(true);
      if (inp === tS && !isNaN(s) && o.seek) o.seek(s);
    }));
    [tS, tE].forEach((inp) => inp.addEventListener("blur", () => paint(false)));
    if (rS) {
      rS.addEventListener("input", () => {
        s = Math.min(+rS.value, (e === null ? dur : e) - 1);
        if (e !== null && e - s > max) e = Math.min(dur, s + max);
        paint(); o.seek && o.seek(s);
      });
      rE.addEventListener("input", () => {
        e = Math.max(+rE.value, s + 1);
        if (e - s > max) s = Math.max(0, e - max);
        paint(); o.seek && o.seek(e);
      });
    }
    el.querySelectorAll("[data-now]").forEach((b) => b.onclick = () => {
      const t = o.getCurrent && o.getCurrent();
      if (t == null || isNaN(t)) return toast("Play or move the video to a moment first.");
      if (b.dataset.now === "s") { s = t; if (e !== null && e <= s + 1) e = dur ? Math.min(dur, s + Math.min(max, 30)) : s + 30; }
      else { e = t; if (e <= s + 1) s = Math.max(0, e - Math.min(max, 30)); }
      paint();
    });
    paint();
    return { get: () => ({ start: s, end: e }), ok: () => !problem() };
  }

  // ------------------------------------------------------------ busy panel
  function busy(on, label, sub) {
    $("busy").classList.toggle("hidden", !on);
    if (label) $("busyLabel").textContent = label;
    if (sub !== undefined) $("busySub").textContent = sub;
  }
  function busyBar(p) {
    const bar = $("busyBar").parentElement;
    bar.classList.toggle("indeterminate", p == null);
    if (p != null) $("busyBar").style.width = p + "%";
  }
  function showError(msg) { $("uploadError").textContent = msg; $("uploadError").classList.toggle("hidden", !msg); }

  // source tabs
  $("srcSeg").querySelectorAll("button").forEach((b) => b.onclick = () => {
    $("srcSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    $("srcFile").classList.toggle("hidden", b.dataset.src !== "file");
    $("srcLink").classList.toggle("hidden", b.dataset.src !== "link");
    showError("");
  });

  // ------------------------------------------------------------ 1a. upload a file + choose the part
  const drop = $("drop"), fileInput = $("fileInput"), preview = $("filePreview");
  let chosen = null, fileRange = null, previewURL = null;
  const MAX_CLIP = 300, TRACK_MAX = 120;
  fileInput.addEventListener("change", () => fileInput.files[0] && chooseFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (ev) => { const f = ev.dataTransfer.files[0]; if (f) chooseFile(f); });

  function chooseFile(file) {
    showError("");
    chosen = file;
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = URL.createObjectURL(file);
    drop.classList.add("hidden"); $("fileClip").classList.remove("hidden");
    $("fileName").textContent = file.name;
    $("fileInfo").textContent = `${(file.size / 1048576).toFixed(1)} MB · reading…`;
    $("noPreview").classList.add("hidden"); preview.classList.remove("hidden");
    $("uploadBtn").disabled = true;
    let done = false;
    const setup = (dur) => {
      if (done) return; done = true;
      $("fileInfo").textContent = `${(file.size / 1048576).toFixed(1)} MB` + (dur ? ` · ${clock(dur)} long` : "");
      fileRange = makeRange($("fileRange"), {
        duration: dur, max: MAX_CLIP, trackMax: TRACK_MAX, defaultLen: 60,
        getCurrent: dur ? () => preview.currentTime : null,
        seek: dur ? (t) => { try { preview.currentTime = t; } catch (e) { /* ignore */ } } : null,
        onChange: (ok) => { $("uploadBtn").disabled = !ok; },
      });
    };
    preview.onloadedmetadata = () => setup(preview.duration);
    preview.onerror = () => { preview.classList.add("hidden"); $("noPreview").classList.remove("hidden"); setup(null); };
    preview.src = previewURL;
    setTimeout(() => { if (!done && !(preview.duration > 0)) preview.onerror(); }, 6000);
  }
  $("otherFile").onclick = () => {
    preview.pause(); preview.removeAttribute("src"); preview.load();
    $("fileClip").classList.add("hidden"); drop.classList.remove("hidden"); fileInput.value = ""; chosen = null;
  };

  $("uploadBtn").onclick = () => {
    if (!chosen || !fileRange || !fileRange.ok()) return;
    preview.pause();
    const { start, end } = fileRange.get();
    const fd = new FormData();
    fd.append("video", chosen);
    if (start) fd.append("start", start);
    if (end !== null) fd.append("end", end);
    $("fileClip").classList.add("hidden"); showError("");
    busy(true, `Uploading ${chosen.name}`, "Keep this tab open"); busyBar(0);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const p = (e.loaded / e.total) * 100;
      busyBar(p);
      $("busySub").textContent = `${(e.loaded / 1048576).toFixed(1)} of ${(e.total / 1048576).toFixed(1)} MB`;
      if (p >= 100) { busy(true, "Cutting the selected part…", "Converting it to a standard format and reading frames"); busyBar(null); }
    };
    xhr.onload = () => {
      let j = null; try { j = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
      if (xhr.status !== 200 || !j) return fileFail((j && j.error) || "Upload failed. Please try again.");
      busy(false); S.cuts = null; openJob(j.id, j.meta, j.max_seconds);
      $("fileClip").classList.add("hidden"); drop.classList.remove("hidden"); fileInput.value = "";
    };
    xhr.onerror = () => fileFail("Connection lost during upload. Please try again.");
    xhr.send(fd);
  };
  function fileFail(msg) { busy(false); $("fileClip").classList.remove("hidden"); showError(msg); }

  // ------------------------------------------------------------ 1b. YouTube link + choose the part
  let linkInfo = null, linkRange = null, ytPlayer = null, ytApi = null;
  function loadYT() {
    if (ytApi) return ytApi;
    ytApi = new Promise((resolve, reject) => {
      if (window.YT && window.YT.Player) return resolve(window.YT);
      window.onYouTubeIframeAPIReady = () => resolve(window.YT);
      const sc = document.createElement("script");
      sc.src = "https://www.youtube.com/iframe_api"; sc.onerror = reject;
      document.head.appendChild(sc);
      setTimeout(() => reject(new Error("timeout")), 8000);
    });
    return ytApi;
  }

  $("linkForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    showError(""); $("linkClip").classList.add("hidden");
    const url = $("linkInput").value.trim();
    $("linkCheck").disabled = true; $("linkCheck").textContent = "Checking…";
    try {
      linkInfo = await getJSON("/api/link/info", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
      showLink(linkInfo);
    } catch (e) { showError(e.message); }
    finally { $("linkCheck").disabled = false; $("linkCheck").textContent = "Check link"; }
  });

  function showLink(info) {
    $("linkClip").classList.remove("hidden");
    $("linkTitle").textContent = info.title;
    $("linkInfo").textContent = [info.uploader, info.duration ? `${clock(info.duration)} long` : null, info.height ? `${info.height}p` : null].filter(Boolean).join(" · ");
    $("fetchBtn").disabled = true;
    const thumb = $("ytThumb");
    if (ytPlayer && ytPlayer.destroy) { try { ytPlayer.destroy(); } catch (e) { /* ignore */ } ytPlayer = null; }
    const wrap = document.querySelector(".yt-wrap");
    wrap.querySelectorAll("#ytPlayer, iframe").forEach((n) => n.remove());
    wrap.prepend(Object.assign(document.createElement("div"), { id: "ytPlayer" }));
    thumb.classList.add("hidden"); $("noLinkPreview").classList.add("hidden");
    const range = (withPlayer) => {
      linkRange = makeRange($("linkRange"), {
        duration: info.duration, max: info.max_clip || MAX_CLIP, trackMax: TRACK_MAX, defaultLen: 30, requireEnd: true,
        getCurrent: withPlayer ? () => ytPlayer && ytPlayer.getCurrentTime && ytPlayer.getCurrentTime() : null,
        seek: withPlayer ? (t) => ytPlayer && ytPlayer.seekTo && ytPlayer.seekTo(t, true) : null,
        onChange: (ok) => { $("fetchBtn").disabled = !ok; },
      });
    };
    if (info.youtube_id) {
      range(true);
      loadYT().then((YT) => {
        ytPlayer = new YT.Player("ytPlayer", { videoId: info.youtube_id, playerVars: { rel: 0, modestbranding: 1, playsinline: 1 } });
      }).catch(() => {
        if (info.thumbnail) { thumb.src = info.thumbnail; thumb.classList.remove("hidden"); } else $("noLinkPreview").classList.remove("hidden");
        range(false);
      });
    } else {
      if (info.thumbnail) { thumb.src = info.thumbnail; thumb.classList.remove("hidden"); } else $("noLinkPreview").classList.remove("hidden");
      range(false);
    }
  }

  $("fetchBtn").onclick = async () => {
    if (!linkInfo || !linkRange || !linkRange.ok()) return;
    if (ytPlayer && ytPlayer.pauseVideo) try { ytPlayer.pauseVideo(); } catch (e) { /* ignore */ }
    const { start, end } = linkRange.get();
    showError(""); $("linkClip").classList.add("hidden");
    busy(true, "Getting the video", `${linkInfo.title} · ${clock(start)} → ${clock(end)}`); busyBar(null);
    try {
      const j = await getJSON("/api/link/fetch", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: linkInfo.webpage_url || linkInfo.url, title: linkInfo.title, start, end }) });
      const poll = async () => {
        let st;
        try { st = await getJSON(`/api/link/${j.id}/status`); } catch (e) { return setTimeout(poll, 1500); }
        if (st.status === "error") { busy(false); $("linkClip").classList.remove("hidden"); return showError(st.error || "Download failed."); }
        if (st.status === "done") {
          const info = await getJSON(`/api/jobs/${j.id}`);
          busy(false); S.cuts = null; openJob(info.id, info.meta, info.max_seconds); return;
        }
        $("busyLabel").textContent = st.stage || "Working…";
        busyBar(st.progress ? st.progress : null);
        setTimeout(poll, 1000);
      };
      poll();
    } catch (e) { busy(false); $("linkClip").classList.remove("hidden"); showError(e.message); }
  };

  // ------------------------------------------------------------ selection view
  const canvas = $("canvas"), ctx = canvas.getContext("2d"), stage = $("stage");

  function openJob(id, meta, maxSeconds, keepTargets) {
    S.id = id; S.meta = meta; S.maxSeconds = maxSeconds || 120; setHash();
    stage.style.aspectRatio = `${meta.width} / ${meta.height}`;
    const fs = $("frameSlider"); fs.max = meta.frames - 1;
    if (!keepTargets) { S.targets = []; S.frame = 0; S.nextId = 1; }
    fs.value = S.frame;
    show("select");
    renderTargets(); updateEnd(true); loadFrame(S.frame);
    watchAnalysis();
  }

  // camera cuts (replays, close-ups) are found in the background after upload
  function watchAnalysis() {
    clearTimeout(S.cutPoll);
    if (S.cuts) return;
    const id = S.id;
    const check = async () => {
      if (S.id !== id) return;
      try {
        const j = await getJSON(`/api/jobs/${id}`);
        if (j.analysis) { S.cuts = j.analysis.cuts || []; updateEnd(true); return; }
      } catch (e) { return; }
      S.cutPoll = setTimeout(check, 1500);
    };
    check();
  }
  function srcStart() { return (S.meta && S.meta.source && S.meta.source.start) || 0; }
  function nextCut(f) { return (S.cuts || []).find((c) => c > f); }

  function loadFrame(idx) {
    S.frame = idx;
    $("frameTime").textContent = fmt(idx / S.meta.fps);
    $("frameTime").title = srcStart() ? `${clock(srcStart() + idx / S.meta.fps)} in the original video` : "";
    $("frameLoading").classList.remove("hidden");
    const img = new Image();
    img.onload = () => {
      if (S.frame !== idx) return;
      S.img = img; S.players = []; S.balls = []; draw(); detect(idx);
    };
    img.onerror = () => { $("frameLoading").classList.add("hidden"); toast("Could not load this frame."); };
    img.src = api(`/frame/${idx}`);
  }

  async function detect(idx) {
    if (S.detectCtl) S.detectCtl.abort();
    const ctl = new AbortController(); S.detectCtl = ctl;
    try {
      const j = await getJSON(api(`/detect/${idx}`), { signal: ctl.signal });
      if (S.frame !== idx) return;
      // the most likely ball, plus any other candidate that is nearly as likely
      const bl = j.balls || [];
      S.players = j.players; S.balls = bl.filter((b) => b.score >= bl[0].score - 0.06); draw();
      if (S.mode === "click") stageMsg(S.players.length ? "" : "No players found on this frame. Try another frame or draw a box.");
      modeHint();
    } catch (e) { if (e.name !== "AbortError") stageMsg("Player detection failed. You can still draw boxes."); }
    finally { if (S.detectCtl === ctl) { S.detectCtl = null; $("frameLoading").classList.add("hidden"); modeHint(); } }
  }

  function stageMsg(t) { const m = $("stageMsg"); m.textContent = t; m.classList.toggle("hidden", !t); }

  function fit() {
    const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(r.width * dpr)) { canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr); }
    return { k: canvas.width / S.meta.width, dpr };
  }

  function draw() {
    if (!S.img || !S.meta) return;
    const { k, dpr } = fit();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(S.img, 0, 0, canvas.width, canvas.height);
    const R = (b) => [b[0] * k, b[1] * k, b[2] * k, b[3] * k];
    if (S.mode === "click") {
      S.players.forEach((b, i) => {
        if (S.targets.some((t) => iou(t.box, b) > 0.5)) return;
        const [x, y, w, h] = R(b);
        ctx.setLineDash(i === S.hover ? [] : [5 * dpr, 4 * dpr]);
        ctx.lineWidth = (i === S.hover ? 2.5 : 1.5) * dpr;
        ctx.strokeStyle = i === S.hover ? "#c6f432" : "rgba(255,255,255,.75)";
        ctx.strokeRect(x - 2 * dpr, y - 2 * dpr, w + 4 * dpr, h + 4 * dpr);
        if (i === S.hover) { ctx.fillStyle = "rgba(198,244,50,.15)"; ctx.fillRect(x - 2 * dpr, y - 2 * dpr, w + 4 * dpr, h + 4 * dpr); }
      });
      // ball suggestions: a dashed ring, the most likely one labelled "Ball?"
      if (!S.targets.some((t) => t.kind === "ball")) {
        S.balls.forEach((bb, i) => {
          const [x, y, w, h] = R(bb.box), on = i === S.hoverBall;
          const r = Math.max(w, h) / 2 + 6 * dpr;
          ctx.setLineDash(on ? [] : [4 * dpr, 3 * dpr]);
          ctx.lineWidth = (on ? 2.5 : i === 0 ? 2 : 1.3) * dpr;
          ctx.strokeStyle = on ? "#c6f432" : i === 0 ? "rgba(255,214,10,.95)" : "rgba(255,214,10,.55)";
          ctx.beginPath(); ctx.arc(x + w / 2, y + h / 2, r, 0, 7); ctx.stroke();
          if (on) { ctx.fillStyle = "rgba(198,244,50,.18)"; ctx.fill(); }
          if (i === 0 || on) {
            ctx.setLineDash([]);
            ctx.font = `600 ${11 * dpr}px Inter, sans-serif`;
            const label = "Ball?", tw = ctx.measureText(label).width + 8 * dpr, th = 16 * dpr;
            const lx = Math.min(x + w / 2 + r + 3 * dpr, canvas.width - tw), ly = Math.max(0, y + h / 2 - r - th);
            ctx.fillStyle = on ? "#c6f432" : "rgba(255,214,10,.95)"; ctx.fillRect(lx, ly, tw, th);
            ctx.fillStyle = "#111"; ctx.fillText(label, lx + 4 * dpr, ly + 12 * dpr);
          }
        });
      }
      ctx.setLineDash([]);
    }
    S.targets.forEach((t) => {
      const [x, y, w, h] = R(t.box);
      ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "rgba(0,0,0,.6)";
      if (t.kind === "ball") {
        const r = Math.max(w, h) / 2 + 5 * dpr;
        ctx.beginPath(); ctx.arc(x + w / 2, y + h / 2, r + 1.5 * dpr, 0, 7); ctx.stroke();
        ctx.strokeStyle = t.color; ctx.lineWidth = 2.5 * dpr; ctx.beginPath(); ctx.arc(x + w / 2, y + h / 2, r, 0, 7); ctx.stroke();
      } else {
        ctx.strokeRect(x - 1, y - 1, w + 2, h + 2);
        ctx.strokeStyle = t.color; ctx.lineWidth = 2.5 * dpr; ctx.strokeRect(x, y, w, h);
      }
      ctx.font = `600 ${12 * dpr}px Inter, sans-serif`;
      const tw = ctx.measureText(t.name).width + 10 * dpr, th = 18 * dpr;
      const lx = Math.min(x, canvas.width - tw), ly = Math.max(0, y - th - 4 * dpr);
      ctx.fillStyle = t.color; ctx.fillRect(lx, ly, tw, th);
      ctx.fillStyle = lum(t.color) > 150 ? "#111" : "#fff"; ctx.fillText(t.name, lx + 5 * dpr, ly + 13 * dpr);
    });
    if (S.drag) {
      const [x, y, w, h] = R(normBox(S.drag));
      ctx.setLineDash([6 * dpr, 4 * dpr]); ctx.lineWidth = 2 * dpr; ctx.strokeStyle = "#c6f432";
      ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
    }
  }
  function lum(hex) {
    const h = hex.replace("#", ""); const f = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
    const n = parseInt(f, 16); return 0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255);
  }
  function normBox(d) {
    const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
    return [x, y, Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0)];
  }
  function toVideo(ev) {
    const r = canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) * S.meta.width / r.width, (ev.clientY - r.top) * S.meta.height / r.height];
  }
  function playerAt(p) {
    let best = -1, area = 1e12;
    const pad = S.meta.width * 0.004;
    S.players.forEach((b, i) => {
      if (p[0] >= b[0] - pad && p[0] <= b[0] + b[2] + pad && p[1] >= b[1] - pad && p[1] <= b[1] + b[3] + pad && b[2] * b[3] < area) {
        if (!S.targets.some((t) => iou(t.box, b) > 0.5)) { best = i; area = b[2] * b[3]; }
      }
    });
    return best;
  }

  function ballAt(p) {
    // the ball is tiny: accept clicks within ~14 screen pixels of it
    if (S.targets.some((t) => t.kind === "ball")) return -1;
    const r = canvas.getBoundingClientRect(), px = 14 * S.meta.width / Math.max(1, r.width);
    let best = -1, bd = 1e12;
    S.balls.forEach((bb, i) => {
      const b = bb.box, d = Math.hypot(p[0] - (b[0] + b[2] / 2), p[1] - (b[1] + b[3] / 2));
      if (d <= Math.max(px, b[2], b[3]) && d < bd) { best = i; bd = d; }
    });
    return best;
  }

  canvas.addEventListener("pointerdown", (ev) => {
    if (!S.img) return;
    const p = toVideo(ev);
    if (S.mode === "click") {
      const bi = ballAt(p);
      if (bi >= 0) { addTarget(S.balls[bi].box.slice(), "ball"); S.hoverBall = -1; return; }
      const i = playerAt(p);
      if (i >= 0) addTarget(S.players[i].slice(), "player");
      else toast("Nothing detected there. Switch to “Draw a box” to select it manually.");
      return;
    }
    canvas.setPointerCapture(ev.pointerId);
    S.drag = { x0: p[0], y0: p[1], x1: p[0], y1: p[1] };
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!S.img) return;
    const p = toVideo(ev);
    if (S.drag) { S.drag.x1 = p[0]; S.drag.y1 = p[1]; draw(); return; }
    if (S.mode === "click") {
      const hb = ballAt(p), h = hb >= 0 ? -1 : playerAt(p);
      if (h !== S.hover || hb !== S.hoverBall) { S.hover = h; S.hoverBall = hb; draw(); }
    }
  });
  canvas.addEventListener("pointerleave", () => { if (S.hover !== -1 || S.hoverBall !== -1) { S.hover = -1; S.hoverBall = -1; draw(); } });
  canvas.addEventListener("pointerup", () => {
    if (!S.drag) return;
    const b = normBox(S.drag).map(Math.round); S.drag = null;
    const minSide = Math.max(4, S.meta.height * 0.006);
    if (b[2] < minSide || b[3] < minSide) { draw(); toast("Drag to draw a box around the player or the ball."); return; }
    const small = Math.max(b[2], b[3]) < S.meta.height * 0.04;
    addTarget(b, small ? "ball" : "player");
  });
  window.addEventListener("resize", () => draw());

  function modeHint() {
    let t;
    if (S.mode !== "click") t = "Drag a box around the ball or any player. Draw it tightly around them.";
    else if (S.targets.some((x) => x.kind === "ball") || !S.img || S.detectCtl) t = "Players found on this frame are outlined. Click one to track it.";
    else if (S.balls.length) t = "Players and the ball (yellow ring) found on this frame are outlined. Click one to track it.";
    else t = "Players found on this frame are outlined. The ball wasn’t found here: draw a box around it, or try another frame.";
    $("modeHint").textContent = t;
  }

  function setMode(m) {
    S.mode = m; S.hover = -1; S.hoverBall = -1;
    $("modeClick").classList.toggle("on", m === "click"); $("modeDraw").classList.toggle("on", m === "draw");
    stage.classList.toggle("click", m === "click");
    modeHint();
    stageMsg(m === "click" && S.img && !S.players.length && !S.detectCtl ? "No players found on this frame." : "");
    draw();
  }
  $("modeClick").onclick = () => setMode("click");
  $("modeDraw").onclick = () => setMode("draw");

  function addTarget(box, kind) {
    if (S.targets.length >= 12) return toast("You can track up to 12 targets at once.");
    if (S.targets.some((t) => iou(t.box, box) > 0.6)) return toast("That one is already selected.");
    if (kind === "ball" && S.targets.some((t) => t.kind === "ball")) kind = "player";
    const used = S.targets.map((t) => t.color);
    const color = PALETTE.find((c) => !used.includes(c)) || PALETTE[S.targets.length % PALETTE.length];
    const nPlayers = S.targets.filter((t) => t.kind === "player").length;
    S.targets.push({ uid: S.nextId++, box, kind, color, name: kind === "ball" ? "Ball" : `Player ${nPlayers + 1}` });
    renderTargets(); draw();
  }

  function renderTargets() {
    const list = $("targets"); list.innerHTML = "";
    S.targets.forEach((t) => {
      const el = document.createElement("div");
      el.className = "target"; el.style.setProperty("--c", t.color);
      el.innerHTML = `
        <input class="name" maxlength="24" value="${esc(t.name)}" aria-label="Name">
        <button class="rm" title="Remove" aria-label="Remove">×</button>
        <div class="row">
          <div class="swatches">${PALETTE.map((c) => `<button style="--s:${c}" data-c="${c}" class="${c === t.color ? "on" : ""}" aria-label="Route colour ${c}"></button>`).join("")}</div>
          <label class="custom" title="Custom colour"><input type="color" value="${t.color.length === 7 ? t.color : "#ff3b30"}"></label>
          <div class="seg small kind"><button data-k="player" class="${t.kind === "player" ? "on" : ""}">Player</button><button data-k="ball" class="${t.kind === "ball" ? "on" : ""}">Ball</button></div>
        </div>`;
      el.querySelector(".name").addEventListener("input", (e) => { t.name = e.target.value || (t.kind === "ball" ? "Ball" : "Player"); draw(); });
      el.querySelector(".rm").onclick = () => { S.targets = S.targets.filter((x) => x !== t); renderTargets(); draw(); };
      el.querySelectorAll(".swatches button").forEach((b) => b.onclick = () => { t.color = b.dataset.c; renderTargets(); draw(); });
      el.querySelector(".custom input").addEventListener("input", (e) => { t.color = e.target.value; el.style.setProperty("--c", t.color); el.querySelectorAll(".swatches button").forEach((b) => b.classList.remove("on")); draw(); });
      el.querySelectorAll(".kind button").forEach((b) => b.onclick = () => {
        t.kind = b.dataset.k;
        if (t.kind === "ball" && /^Player \d+$/.test(t.name)) t.name = "Ball";
        renderTargets(); draw();
      });
      list.appendChild(el);
    });
    const has = S.targets.length > 0;
    modeHint();
    $("targetsEmpty").classList.toggle("hidden", has);
    $("clearAll").classList.toggle("hidden", !has);
    $("startBtn").disabled = !has;
    $("startBtn").textContent = has ? `Start tracking ${S.targets.length} target${S.targets.length > 1 ? "s" : ""}` : "Start tracking";
    // start frame is locked while targets exist (they belong to this frame)
    ["frameSlider", "prevFrame", "nextFrame"].forEach((id) => $(id).disabled = has);
    $("scrubNote").textContent = has
      ? "The start frame is locked because your selection belongs to it. Clear the selection to pick another frame."
      : "Slide to a frame where your player or the ball is clearly visible. Tracking starts from this frame.";
    estimate();
  }
  $("clearAll").onclick = () => { S.targets = []; renderTargets(); draw(); };

  let scrubT;
  $("frameSlider").addEventListener("input", (e) => {
    const v = +e.target.value; $("frameTime").textContent = fmt(v / S.meta.fps);
    clearTimeout(scrubT); scrubT = setTimeout(() => { loadFrame(v); updateEnd(); }, 120);
  });
  $("prevFrame").onclick = () => step(-1);
  $("nextFrame").onclick = () => step(1);
  function step(d) {
    const v = Math.max(0, Math.min(S.meta.frames - 1, S.frame + d));
    $("frameSlider").value = v; loadFrame(v); updateEnd();
  }

  function updateEnd(reset) {
    const es = $("endSlider"), m = S.meta;
    const maxEnd = Math.min(m.frames - 1, S.frame + Math.floor(S.maxSeconds * m.fps));
    es.min = Math.min(S.frame + 1, m.frames - 1); es.max = maxEnd;
    const nc = nextCut(S.frame);
    const def = nc !== undefined ? Math.max(+es.min, Math.min(maxEnd, nc - 1)) : maxEnd;
    if (reset || +es.value > maxEnd || +es.value <= S.frame) es.value = def;
    S.end = +es.value; endLabel();
  }
  function endLabel() {
    const secs = (S.end - S.frame) / S.meta.fps;
    $("endLabel").textContent = `${fmt(S.end / S.meta.fps)} · ${secs.toFixed(1)} s tracked` +
      (S.end - S.frame >= Math.floor(S.maxSeconds * S.meta.fps) ? ` (max ${S.maxSeconds}s per run)` : "");
    const nc = nextCut(S.frame);
    $("cutNote").textContent = S.cuts === null ? "Checking the video for camera cuts…"
      : nc === undefined ? (S.cuts.length ? "No camera cut after this point." : "No camera cuts found: one continuous shot.")
      : S.end < nc ? `The camera cuts at ${fmt(nc / S.meta.fps)} (replay or another camera), so tracking stops just before it by default.`
      : `The camera cuts at ${fmt(nc / S.meta.fps)}. Routes will be split there and targets re-identified after the cut.`;
    estimate();
  }
  $("endSlider").addEventListener("input", (e) => { S.end = +e.target.value; endLabel(); });

  function segBind(id, key, cb) {
    $(id).querySelectorAll("button").forEach((b) => b.onclick = () => {
      $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      cb(b.dataset[key]);
    });
  }
  segBind("speedSeg", "step", (v) => {
    S.step = +v;
    $("speedHint").textContent = ["", "Every frame is analysed.", "Every 2nd frame is analysed (about 2× faster).", "Every 3rd frame is analysed (about 3× faster, less precise on fast moves)."][S.step];
    estimate();
  });
  segBind("trailSeg", "trail", (v) => { S.trail = v; });

  function estimate() {
    if (!S.meta || !S.targets.length) { $("estimate").textContent = ""; return; }
    const n = (S.end - S.frame) / S.step;
    const per = 0.05 + 0.04 * S.targets.length + 0.02 * (S.meta.width * S.meta.height) / (1280 * 720);
    $("estimate").textContent = `Estimated processing time: about ${dur(n * per + 6)}`;
  }

  $("startBtn").onclick = async () => {
    $("startBtn").disabled = true;
    try {
      await getJSON(api("/track"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start: S.frame, end: S.end, step: S.step, trail: S.trail, enhance: $("enhance").checked,
          targets: S.targets.map((t) => ({ name: t.name, kind: t.kind, color: t.color, box: t.box })),
        }),
      });
      startPolling();
    } catch (e) { toast(e.message); $("startBtn").disabled = false; }
  };

  // ------------------------------------------------------------ tracking progress
  function startPolling() {
    show("track");
    $("trackError").classList.add("hidden"); $("backFromError").classList.add("hidden"); $("cancelBtn").classList.remove("hidden");
    $("trackChips").innerHTML = S.targets.map((t) => `<span class="chip" style="--c:${t.color}"><i></i>${esc(t.name)}</span>`).join("");
    clearInterval(S.poll);
    const tick = async () => {
      let st;
      try { st = await getJSON(api("/status")); } catch (e) { return; }
      $("trackStage").textContent = st.stage || "Working…";
      $("trackBar").style.width = (st.progress || 0) + "%";
      $("trackPct").textContent = `${Math.round(st.progress || 0)}%`;
      $("trackEta").textContent = st.eta ? `about ${dur(st.eta)} left` : "";
      if (st.status === "done") { clearInterval(S.poll); loadResults(); }
      else if (st.status === "error") {
        clearInterval(S.poll);
        $("trackStage").textContent = "Tracking failed";
        $("trackError").textContent = st.error || "Something went wrong."; $("trackError").classList.remove("hidden");
        $("cancelBtn").classList.add("hidden"); $("backFromError").classList.remove("hidden");
      } else if (st.status === "cancelled" || st.status === "idle") { clearInterval(S.poll); backToSelect(); }
    };
    tick(); S.poll = setInterval(tick, 1000);
  }
  $("cancelBtn").onclick = async () => { await fetch(api("/cancel"), { method: "POST" }); $("trackStage").textContent = "Cancelling…"; };
  $("backFromError").onclick = () => backToSelect();
  function backToSelect() { openJob(S.id, S.meta, S.maxSeconds, true); }

  // ------------------------------------------------------------ results
  async function loadResults() {
    let R;
    try { R = await getJSON(api("/result")); } catch (e) { toast(e.message); return; }
    S.result = R; show("results");
    const T = R.targets;
    $("resTitle").textContent = R.job.name.replace(/\.[^.]+$/, "") || "Results";
    const src = R.video && R.video.source;
    $("resSource").innerHTML = !src ? "" : src.type === "link"
      ? `Source: <a href="${esc(src.url)}" target="_blank" rel="noopener">${esc(src.title || src.url)}</a> · ${clock(src.start)} → ${clock(src.end)}`
      : `Source: ${esc(src.file)}${src.start || src.end ? ` · ${clock(src.start || 0)} → ${src.end ? clock(src.end) : "end"}` : ""}`;
    $("resChips").innerHTML = T.map((t) => `<span class="chip" style="--c:${t.color}"><i></i>${esc(t.name)}</span>`).join("");
    const f = (n) => api(`/files/${n}`), base = (R.job.name.replace(/\.[^.]+$/, "") || "match").replace(/[^\w-]+/g, "_");
    const v = $("resVideo"); v.src = f("tracked.mp4") + `?v=${Date.now()}`;
    $("mapImg").src = f("route_map.jpg") + `?v=${Date.now()}`; $("mapLink").href = f("route_map.jpg");
    $("dlVideo").href = f(`tracked.mp4?download=${base}_tracked.mp4`);
    $("dlMap").href = f(`route_map.jpg?download=${base}_route_map.jpg`);
    $("dlCsv").href = f(`tracks.csv?download=${base}_positions.csv`);
    $("dlZip").href = api("/report.zip");
    const P = R.processing;
    $("quickStats").innerHTML = [
      [P.frames, "frames analysed"], [`${(P.avg_ms_per_frame || 0).toFixed(0)} ms`, "per frame"],
      [`${Math.round(T.reduce((a, t) => a + t.consistency_pct, 0) / T.length)}%`, "frames with a detection"],
      [dur(P.tracking_time_s), "tracking time"],
    ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
    renderAnalysis(R); renderPipeline(R); renderCheck(R);
    selectTab("video");
  }

  function selectTab(name) {
    document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
    ["video", "map", "analysis", "pipeline", "check"].forEach((t) => $("tab-" + t).classList.toggle("hidden", t !== name));
    if (name !== "video") $("resVideo").pause();
  }
  document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => selectTab(b.dataset.tab));

  function renderAnalysis(R) {
    const f = (n) => api(`/files/${n}`);
    const T = R.targets, P = R.processing, E = R.enhancement_eval;
    const na = (x, u = "") => (x == null ? "–" : `${x}${u}`);
    let h = `<div class="card"><h3>Tracking summary</h3>
      <p class="sub">“Detected” means the target was located in that frame by the tracker or by re-detection. “Occluded” frames are bridged by the motion model.</p>
      <div class="tablewrap"><table><thead><tr><th>Target</th><th>Detected</th><th>Re-detected</th><th>Occluded</th><th>Lost</th><th>Longest gap</th><th>Confidence</th><th>Distance*</th><th>Avg speed*</th><th>Top speed*</th></tr></thead><tbody>`;
    T.forEach((t) => {
      h += `<tr><td><span class="dot" style="--c:${t.color}"></span>${esc(t.name)}</td><td>${t.consistency_pct}%</td><td>${t.redetections}×</td>
        <td>${t.predicted_pct}%</td><td>${t.lost_pct}%</td><td>${t.longest_gap_s}s</td><td>${t.mean_confidence}</td>
        <td>${na(t.distance_m, " m")}</td><td>${na(t.avg_speed_kmh, " km/h")}</td><td>${na(t.max_speed_kmh, " km/h")}</td></tr>`;
    });
    h += `</tbody></table></div><p class="note">* Approximate, using a scale of ${R.scale.metres_per_pixel} m per pixel taken from an assumed 1.8 m player height. Perspective makes this less accurate for players far from the camera, and the ball's height above the pitch is not modelled.</p></div>`;

    h += `<div class="card"><h3>Processing performance</h3><div class="tablewrap"><table><thead><tr><th>Stage</th><th>ms / frame</th></tr></thead><tbody>
      ${Object.entries(P.stage_ms).map(([k, v]) => `<tr><td>${{ enhance: "Enhancement", segment: "Segmentation + detection", camera: "Camera motion", track: "Tracking" }[k] || k}</td><td>${v.toFixed(1)}</td></tr>`).join("")}
      <tr><td><b>Total</b></td><td><b>${P.avg_ms_per_frame.toFixed(1)}</b> (${P.processing_fps} fps)</td></tr>
      <tr><td>Camera cuts in tracked range</td><td>${(P.camera_cuts || []).length}${(P.camera_cuts || []).length ? " (routes split; route map shows the first shot)" : ""}</td></tr>
      <tr><td>Burnt-in graphics masked</td><td>${P.overlay_masked ? "yes" : "none found"}</td></tr></tbody></table></div>
      <div class="charts" style="margin-top:12px"><img src="${f("charts/processing_time.png")}" alt="Processing time chart"></div></div>`;

    h += `<div class="card"><h3>Detection consistency across frames</h3><div class="charts"><img src="${f("charts/tracking_status.png")}" alt="Status timeline"></div>
      <div class="charts" style="margin-top:12px"><img src="${f("charts/confidence.png")}" alt="Confidence over time"></div>
      <div class="charts" style="margin-top:12px"><img src="${f("charts/speed.png")}" alt="Speed over time"></div></div>`;

    if (E && E.original && E.original.brightness != null) {
      const row = (k, lbl, better) => {
        const a = E.original[k], b = E.enhanced[k];
        const good = better === "up" ? b > a : b < a;
        return `<tr><td>${lbl}</td><td>${a}</td><td class="${good ? "up" : ""}">${b}</td></tr>`;
      };
      h += `<div class="card"><h3>Enhancement effectiveness</h3><p class="sub">Average over ${E.frames} sample frames.</p><div class="tablewrap"><table>
        <thead><tr><th>Measure</th><th>Original</th><th>Enhanced</th></tr></thead><tbody>
        ${row("brightness", "Mean brightness (0-255)", "up")}${row("contrast_rms", "RMS contrast", "up")}${row("entropy", "Entropy (bits)", "up")}
        ${row("sharpness", "Sharpness (Laplacian variance)", "up")}${row("noise_sigma", "Noise σ estimate", "down")}
        <tr><td>Players detected per frame</td><td>${E.detections_original}</td><td class="${E.detections_enhanced >= E.detections_original ? "up" : ""}">${E.detections_enhanced}</td></tr>
        </tbody></table></div><div class="charts" style="margin-top:12px"><img src="${f("charts/enhancement.png")}" alt="Enhancement chart"></div></div>`;
    }
    $("tab-analysis").innerHTML = h;
    zoomable($("tab-analysis"));
  }

  function renderPipeline(R) {
    const f = (n) => api(`/files/${n}`);
    const fig = (src, cap) => `<figure class="figure"><img src="${src}" alt="${esc(cap)}" data-cap="${esc(cap)}"><figcaption>${esc(cap)}</figcaption></figure>`;
    const st = R.stages || [];
    const enh = st.filter((s) => /^0[1-6]/.test(s.file)), seg = st.filter((s) => /^(0[7-9]|1[0-4])/.test(s.file)), trk = st.filter((s) => /^15/.test(s.file));
    const kv = (o) => `<dl class="kv">${Object.entries(o).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</dd>`).join("")}</dl>`;
    let h = `<div class="card"><h3>Pipeline</h3><div style="display:grid;grid-template-columns:minmax(0,260px) 1fr;gap:18px;align-items:start" class="pipe">
        <div class="charts"><img src="${f("charts/pipeline_flowchart.png")}" alt="Pipeline flowchart" data-cap="Pipeline flowchart"></div>
        <div><h3 style="font-size:14px">Enhancement parameters (chosen automatically for this clip)</h3>${kv(R.enhancement)}
        <h3 style="font-size:14px;margin-top:14px">Segmentation parameters</h3>${kv(R.segmentation)}</div></div></div>`;
    h += `<div class="card"><h3>Sample extracted frames</h3><div class="gallery">${(R.samples || []).map((s) => fig(f("samples/" + s.file), `Frame ${s.frame} · ${s.time}s`)).join("")}</div></div>`;
    h += `<div class="card"><h3>Image enhancement stages</h3><div class="gallery">${enh.map((s) => fig(f("stages/" + s.file), s.caption)).join("")}</div></div>`;
    h += `<div class="card"><h3>Segmentation stages</h3><div class="gallery">${seg.map((s) => fig(f("stages/" + s.file), s.caption)).join("")}</div></div>`;
    if (trk.length) h += `<div class="card"><h3>Tracking progression</h3>${trk.map((s) => fig(f("stages/" + s.file), s.caption)).join("")}</div>`;
    $("tab-pipeline").innerHTML = h;
    zoomable($("tab-pipeline"));
  }

  function renderCheck(R) {
    const f = (n) => api(`/files/${n}`);
    const marks = (R.verification && R.verification.marks) || {};
    const T = R.targets;
    let h = `<div class="card"><h3>Measure tracking accuracy</h3>
      <p class="sub">These are ${R.spot.length} evenly spaced frames from the result. For each target, mark ✓ if its box is on the right player (or the ball) and ✗ if it's wrong. If a target is off-screen, leave it unmarked. Your answers give a tracking accuracy that you checked yourself, and saving adds it to the ZIP download.</p>
      <div class="spot">`;
    R.spot.forEach((s) => {
      h += `<div class="item"><img src="${f("spot/" + s.file)}" alt="Frame ${s.frame}" data-cap="Frame ${s.frame} · ${s.time}s"><div class="marks">
        <div class="mark" style="color:var(--muted)">Frame ${s.frame} · ${s.time}s</div>`;
      T.forEach((t) => {
        const key = `${s.frame}|${t.id}`, m = marks[key];
        const stt = s.targets[t.id];
        h += `<div class="mark" data-key="${key}"><span class="dot" style="--c:${t.color}"></span><span class="nm">${esc(t.name)}${stt === "lost" ? " (lost)" : stt === "predicted" ? " (occluded)" : ""}</span>
          <button class="ok ${m === "ok" ? "on" : ""}" aria-label="Correct">✓</button><button class="bad ${m === "bad" ? "on" : ""}" aria-label="Wrong">✗</button></div>`;
      });
      h += `</div></div>`;
    });
    h += `</div><div class="acc" id="accSummary"></div>
      <button class="primary" id="saveCheck" style="margin-top:14px">Save to report</button></div>`;
    $("tab-check").innerHTML = h;
    const cur = Object.assign({}, marks);
    const summary = () => {
      $("accSummary").innerHTML = T.map((t) => {
        const vals = Object.entries(cur).filter(([k, v]) => k.endsWith("|" + t.id) && v);
        const ok = vals.filter(([, v]) => v === "ok").length;
        return `<span class="chip" style="--c:${t.color}"><i></i>${esc(t.name)}: ${vals.length ? `<b>${Math.round(100 * ok / vals.length)}%</b> (${ok}/${vals.length})` : "not checked"}</span>`;
      }).join("");
    };
    $("tab-check").querySelectorAll(".mark[data-key]").forEach((row) => {
      row.querySelectorAll("button").forEach((b) => b.onclick = () => {
        const v = b.classList.contains("ok") ? "ok" : "bad", k = row.dataset.key;
        cur[k] = cur[k] === v ? null : v;
        row.querySelector(".ok").classList.toggle("on", cur[k] === "ok");
        row.querySelector(".bad").classList.toggle("on", cur[k] === "bad");
        summary();
      });
    });
    $("saveCheck").onclick = async () => {
      try {
        const clean = Object.fromEntries(Object.entries(cur).filter(([, v]) => v));
        await getJSON(api("/verify"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ marks: clean }) });
        toast("Saved. The ZIP download now includes your accuracy check.");
      } catch (e) { toast(e.message); }
    };
    summary();
    zoomable($("tab-check"));
  }

  function zoomable(root) {
    root.querySelectorAll("img").forEach((img) => img.addEventListener("click", (e) => {
      if (img.closest("a")) return;
      e.preventDefault();
      const lb = document.createElement("div"); lb.className = "lightbox";
      lb.innerHTML = `<img src="${img.src}" alt=""><p>${esc(img.dataset.cap || img.alt || "")}</p>`;
      lb.onclick = () => lb.remove(); document.body.appendChild(lb);
    }));
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") document.querySelectorAll(".lightbox").forEach((x) => x.remove()); });

  $("trackAgain").onclick = () => { $("resVideo").pause(); openJob(S.id, S.meta, S.maxSeconds, true); };
  $("newVideo").onclick = () => { $("resVideo").pause(); resetToUpload(); };
  $("homeLink").onclick = (e) => { e.preventDefault(); resetToUpload(); };
  function resetToUpload() {
    clearInterval(S.poll); clearTimeout(S.cutPoll); S.cuts = null; S.id = null; S.targets = []; history.replaceState(null, "", location.pathname);
    busy(false); showError(""); fileInput.value = "";
    $("fileClip").classList.add("hidden"); drop.classList.remove("hidden");
    show("upload");
  }

  // ------------------------------------------------------------ resume from link
  async function boot() {
    const m = location.hash.match(/^#job-([0-9a-f]{32})$/);
    if (!m) return show("upload");
    try {
      const j = await getJSON(`/api/jobs/${m[1]}`);
      S.id = j.id; S.meta = j.meta; S.maxSeconds = j.max_seconds; S.cuts = j.analysis ? j.analysis.cuts : null;
      if (j.targets && j.settings) {
        S.targets = j.targets.map((t, i) => ({ uid: i + 1, box: t.box, kind: t.kind, color: t.color, name: t.name }));
        S.nextId = S.targets.length + 1; S.frame = j.settings.start;
      }
      if (j.status === "running" || j.status === "queued") startPolling();
      else if (j.has_result) loadResults();
      else openJob(j.id, j.meta, j.max_seconds, true);
    } catch (e) { show("upload"); toast("That job has expired. Please upload the video again."); history.replaceState(null, "", location.pathname); }
  }
  boot();
})();
