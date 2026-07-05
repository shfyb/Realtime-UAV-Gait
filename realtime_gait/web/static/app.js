const $ = (id) => document.getElementById(id);

const els = {
  videoFeed: $("videoFeed"),
  videoPlaceholder: $("videoPlaceholder"),
  streamUrl: $("streamUrl"),
  enrollName: $("enrollName"),
  enrollTrackSelect: $("enrollTrackSelect"),
  enrollTrackField: $("enrollTrackField"),
  galleryPeople: $("galleryPeople"),
  btnGalleryDelete: $("btnGalleryDelete"),
  galleryListLabel: $("galleryListLabel"),
  modeBadge: $("modeBadge"),
  statusLog: $("statusLog"),
  enrollProgress: $("enrollProgress"),
  silProgress: $("silProgress"),
  spanProgress: $("spanProgress"),
  progressFill: $("progressFill"),
  readyTip: $("readyTip"),
  enrollCard: document.querySelector(".enroll-card"),
  kpiStrip: $("kpiStrip"),
  liveChips: $("liveChips"),
  timingGrid: $("timingGrid"),
  recognitionResults: $("recognitionResults"),
  recogCount: $("recogCount"),
};

let enrollUiState = { active: false, selectedTrackId: null };
let suppressTrackSelectEvent = false;
let lastStatus = { running: false, default_stream: "" };

function updateEnrollTrackOptions(enroll, tracks) {
  const sel = els.enrollTrackSelect;
  if (!sel) return;

  const active = Boolean(enroll && enroll.active);
  const candidates = (enroll && enroll.candidates) || [];
  const pool = active && candidates.length ? candidates : (tracks || []).map((t) => ({
    track_id: t.track_id,
    sil_count: t.sil_count,
    sil_span_sec: 0,
    ready: t.ready,
  }));

  const selected =
    enroll && enroll.selected_track_id != null
      ? String(enroll.selected_track_id)
      : sel.value || "";

  suppressTrackSelectEvent = true;
  sel.innerHTML = '<option value="">自动 — 轮廓最多的人</option>';
  for (const c of pool) {
    const tid = c.track_id;
    const label = active
      ? `T${tid} · 已采 ${c.sil_count} 帧 · ${Number(c.sil_span_sec || 0).toFixed(1)}s${c.ready ? " ✓" : ""}`
      : `T${tid} · 缓冲 ${c.sil_count || 0} 帧${c.ready ? " · 可注册" : ""}`;
    const opt = document.createElement("option");
    opt.value = String(tid);
    opt.textContent = label;
    sel.appendChild(opt);
  }
  if (selected && [...sel.options].some((o) => o.value === selected)) {
    sel.value = selected;
  } else if (!active) {
    sel.value = "";
  }
  suppressTrackSelectEvent = false;

  enrollUiState.active = active;
  enrollUiState.selectedTrackId =
    enroll && enroll.selected_track_id != null ? enroll.selected_track_id : null;
}

async function selectEnrollTrack(trackId) {
  if (!enrollUiState.active || trackId == null) return;
  await api("/api/enroll/select", {
    method: "POST",
    body: JSON.stringify({ track_id: Number(trackId) }),
  });
  await refreshStatus();
}

const MODE_LABELS = {
  preview: "预览",
  recognize: "识别",
  enrolling: "注册中",
};

function esc(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function kpiCard(label, value, sub = "", cls = "") {
  return `<div class="kpi-card">
    <div class="kpi-label">${esc(label)}</div>
    <div class="kpi-value ${cls}">${esc(value)}</div>
    ${sub ? `<div class="kpi-sub">${esc(sub)}</div>` : ""}
  </div>`;
}

function timingBar(label, ms, maxMs) {
  const n = ms == null || Number.isNaN(Number(ms)) ? 0 : Number(ms);
  const pct = maxMs > 0 ? Math.min(100, Math.round((n / maxMs) * 100)) : 0;
  const display = n > 0 ? String(Math.round(n)) : "—";
  return `<div class="timing-row">
    <span class="t-label">${esc(label)}</span>
    <div class="t-bar-wrap"><div class="t-bar" style="width:${pct}%"></div></div>
    <span class="t-val">${esc(display)}</span>
  </div>`;
}

function updateLiveChips(st) {
  if (!els.liveChips) return;
  const chips = [];
  if (st.running) {
    chips.push('<span class="chip chip-live">LIVE</span>');
    chips.push(`<span class="chip">${esc(st.fps != null ? `${Number(st.fps).toFixed(0)} FPS` : "—")}</span>`);
    if (st.lag_ms != null) {
      chips.push(`<span class="chip">${esc(Math.round(st.lag_ms))} ms</span>`);
    }
  } else {
    chips.push('<span class="chip chip-off">未连接</span>');
  }
  const mode = MODE_LABELS[st.mode] || st.mode;
  const modeCls = st.mode === "recognize" ? "chip-mode-ok" : st.mode === "enrolling" ? "chip-warn" : "";
  chips.push(`<span class="chip ${modeCls}">${esc(mode)}</span>`);
  els.liveChips.innerHTML = chips.join("");
}

function updateVideoDashboard(st) {
  if (!els.kpiStrip || !els.timingGrid || !els.recognitionResults) return;

  const tms = st.timings_ms || {};
  const modeLabel = MODE_LABELS[st.mode] || st.mode || "—";
  const recogOn = Boolean(st.recognition_enabled);
  const trackN = (st.tracks || []).length;

  updateLiveChips(st);

  els.kpiStrip.innerHTML = [
    kpiCard("显示帧率", st.fps != null ? Number(st.fps).toFixed(1) : "—", "FPS", st.fps > 10 ? "ok" : "muted"),
    kpiCard("流延迟", st.lag_ms != null ? Math.round(st.lag_ms) : "—", "毫秒", st.lag_ms < 100 ? "ok" : "warn"),
    kpiCard("跟踪目标", trackN, "当前画面", trackN > 0 ? "primary" : "muted"),
    kpiCard("档案库", st.gallery_count ?? 0, recogOn ? "比对已开" : "比对关闭", recogOn ? "ok" : "muted"),
  ].join("");

  const timingItems = [
    ["检测", tms.detect],
    ["跟踪", tms.track],
    ["分割", tms.segment],
    ["步态", tms.gait],
    ["合计", tms.total],
  ];
  const maxMs = Math.max(120, ...timingItems.map(([, v]) => Number(v) || 0));
  els.timingGrid.innerHTML = timingItems.map(([l, v]) => timingBar(l, v, maxMs)).join("");

  if (els.recogCount) {
    els.recogCount.textContent = `${trackN} 人`;
  }

  const tracks = st.tracks || [];
  const enroll = st.enroll || {};
  const enrollActive = Boolean(enroll.active);
  const targetTid = enroll.track_id != null ? enroll.track_id : null;

  if (!st.running) {
    els.recognitionResults.innerHTML = '<p class="dash-empty">未连接图传</p>';
    return;
  }
  if (tracks.length === 0) {
    els.recognitionResults.innerHTML = '<p class="dash-empty">画面暂无行人</p>';
    return;
  }

  els.recognitionResults.innerHTML = tracks
    .map((t) => {
      const name = t.display_name || t.gallery_id || `T${t.track_id}`;
      const matched = Boolean(t.gallery_id);
      let badge = "未识别";
      let badgeCls = "preview";
      if (enrollActive) {
        badge = targetTid === t.track_id ? "注册目标" : "点击选择";
        badgeCls = targetTid === t.track_id ? "success" : "preview";
      } else if (!recogOn) {
        badge = "预览模式";
      } else if (matched) {
        badge = "已识别";
        badgeCls = "success";
      } else if (t.ready) {
        badge = "待比对";
      } else {
        badge = "轮廓不足";
      }
      const dist =
        t.distance != null ? `距离 ${Number(t.distance).toFixed(1)}` : "无匹配";
      const meta = `T${t.track_id} · sil ${t.sil_count} · ${dist}`;
      const initial = esc(name.charAt(0) || "?");
      const targetCls =
        enrollActive && targetTid === t.track_id ? " enroll-target" : "";
      const pickCls = enrollActive ? " enroll-pickable" : "";
      return `
        <div class="recog-card ${matched ? "matched" : ""}${pickCls}${targetCls}" data-track-id="${t.track_id}">
          <div class="recog-avatar">${initial}</div>
          <div class="recog-main">
            <div class="name">${esc(name)}</div>
            <div class="meta">${esc(meta)}</div>
          </div>
          <span class="recog-badge ${badgeCls}">${esc(badge)}</span>
        </div>`;
    })
    .join("");
}

function updateGalleryList(people, count) {
  if (!els.galleryPeople || !els.galleryListLabel) return;
  const n = count ?? (people ? people.length : 0);
  const prev = els.galleryPeople.value;
  els.galleryListLabel.textContent = `库内人员（${n} 人）`;
  els.galleryPeople.innerHTML = "";
  if (!people || people.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "档案库为空 — 请先注册或重新加载";
    els.galleryPeople.appendChild(opt);
    if (els.btnGalleryDelete) els.btnGalleryDelete.disabled = true;
    return;
  }
  for (const p of people) {
    const opt = document.createElement("option");
    opt.value = p.id || "";
    opt.textContent = `${p.display_name || p.id}  ·  ${p.file || ""}`;
    els.galleryPeople.appendChild(opt);
  }
  if (prev && [...els.galleryPeople.options].some((o) => o.value === prev)) {
    els.galleryPeople.value = prev;
  }
  syncGalleryDeleteButton();
}

function syncGalleryDeleteButton() {
  if (!els.btnGalleryDelete || !els.galleryPeople) return;
  const id = els.galleryPeople.value;
  els.btnGalleryDelete.disabled = !id;
}

function selectedGalleryPersonLabel() {
  if (!els.galleryPeople) return "";
  const opt = els.galleryPeople.selectedOptions[0];
  return opt ? opt.textContent : "";
}

function setModeButtons(mode) {
  const preview = $("btnPreview");
  const recognize = $("btnRecognize");
  if (preview) preview.classList.toggle("active", mode === "preview");
  if (recognize) recognize.classList.toggle("active", mode === "recognize" || mode === "enrolling");
}

let pollTimer = null;

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail;
    const msg = typeof detail === "string" ? detail : data.message || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function setStatus(msg, isError = false) {
  if (els.statusLog) {
    els.statusLog.textContent = msg || "";
    els.statusLog.classList.toggle("error", Boolean(isError && msg));
  }
}

function prefillStreamUrl(defaultStream, currentUrl) {
  if (!els.streamUrl || els.streamUrl.value.trim()) return;
  const url = (currentUrl || defaultStream || "").trim();
  if (url) els.streamUrl.value = url;
}

function showEnrollPending(name) {
  setModeBadge("enrolling");
  if (els.enrollCard) els.enrollCard.classList.add("enrolling");
  if (els.enrollProgress) els.enrollProgress.hidden = false;
  const btnStart = $("btnEnrollStart");
  const btnFinish = $("btnEnrollFinish");
  const btnCancel = $("btnEnrollCancel");
  if (btnStart) {
    btnStart.disabled = true;
    btnStart.textContent = "采集中…";
  }
  if (btnFinish) btnFinish.disabled = false;
  if (btnCancel) btnCancel.disabled = false;
  if (els.silProgress) els.silProgress.textContent = `0 / 15（${name}）`;
  if (els.spanProgress) els.spanProgress.textContent = "0.0s / 1.5s";
  if (els.progressFill) els.progressFill.style.width = "0%";
  if (els.readyTip) els.readyTip.hidden = true;
}

function resetEnrollStartButton() {
  const btnStart = $("btnEnrollStart");
  if (btnStart) btnStart.textContent = "开始注册";
}

async function ensureStreamConnected() {
  if (lastStatus.running) return true;
  const url =
    (els.streamUrl && els.streamUrl.value.trim()) ||
    lastStatus.default_stream ||
    "";
  if (!url) {
    setStatus("请先填写 RTSP 地址，并点击「连接图传」", true);
    document.querySelector(".card:not(.enroll-card)")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return false;
  }
  if (els.streamUrl && !els.streamUrl.value.trim()) {
    els.streamUrl.value = url;
  }
  try {
    setStatus("图传未连接，正在自动连接…");
    await connectStream();
  } catch (e) {
    setStatus(e.message || "图传连接失败，请检查 RTSP 地址后重试", true);
    return false;
  }
  if (!lastStatus.running) {
    setStatus("图传连接失败，请检查 RTSP 地址后重试", true);
    return false;
  }
  return true;
}

function setModeBadge(mode) {
  const labels = {
    preview: "PREVIEW · 预览",
    recognize: "RECOGNIZE · 识别",
    enrolling: "ENROLLING · 注册中",
  };
  if (els.modeBadge) {
    els.modeBadge.textContent = labels[mode] || String(mode).toUpperCase();
    els.modeBadge.classList.toggle("enrolling", mode === "enrolling");
  }
  if (els.enrollCard) els.enrollCard.classList.toggle("enrolling", mode === "enrolling");
}

function updateEnrollUi(enroll, tracks) {
  const e = enroll || {};
  const active = Boolean(e.active);
  const btnStart = $("btnEnrollStart");
  const btnFinish = $("btnEnrollFinish");
  const btnCancel = $("btnEnrollCancel");
  if (btnStart) {
    btnStart.disabled = active;
    if (!active) resetEnrollStartButton();
    else if (btnStart.textContent === "开始注册") btnStart.textContent = "采集中…";
  }
  if (btnFinish) btnFinish.disabled = !active;
  if (btnCancel) btnCancel.disabled = !active;
  if (els.enrollProgress) els.enrollProgress.hidden = !active;
  updateEnrollTrackOptions(e, tracks);
  if (!active || !els.silProgress) return;

  const req = e.sil_required || 15;
  const count = e.sil_count || 0;
  const span = e.sil_span_sec || 0;
  const pct = Math.min(100, Math.round((count / req) * 100));
  const target = e.track_id != null ? `T${e.track_id}` : "自动";

  els.silProgress.textContent = `${count} / ${req}（${target}）`;
  if (els.spanProgress) {
    els.spanProgress.textContent = `${span.toFixed(1)}s / ${e.duration_required_sec || 1.5}s`;
  }
  if (els.progressFill) els.progressFill.style.width = `${pct}%`;
  if (els.readyTip) {
    if (e.ready) {
      els.readyTip.hidden = false;
      els.readyTip.classList.remove("warn");
      els.readyTip.textContent = `轮廓已足够（${target}），可以完成注册`;
    } else if (e.track_hint) {
      els.readyTip.hidden = false;
      els.readyTip.classList.add("warn");
      els.readyTip.textContent = e.track_hint;
    } else {
      els.readyTip.hidden = true;
      els.readyTip.classList.remove("warn");
    }
  }
}

async function refreshStatus() {
  try {
    const st = await api("/api/status");
    lastStatus = st;
    prefillStreamUrl(st.default_stream, st.stream_url);
    if (st.message) setStatus(st.message);
    setModeBadge(st.mode);
    updateVideoDashboard(st);
    updateGalleryList(st.gallery_people, st.gallery_count);
    setModeButtons(st.mode);
    updateEnrollUi(st.enroll, st.tracks);
  } catch (err) {
    setStatus(`状态刷新失败: ${err.message}`, true);
  }
}

function startVideo() {
  if (!els.videoFeed) return;
  els.videoFeed.src = `/api/video?t=${Date.now()}`;
  els.videoFeed.classList.add("active");
  if (els.videoPlaceholder) els.videoPlaceholder.style.display = "none";
}

function stopVideo() {
  if (!els.videoFeed) return;
  els.videoFeed.src = "";
  els.videoFeed.classList.remove("active");
  if (els.videoPlaceholder) els.videoPlaceholder.style.display = "flex";
}

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(refreshStatus, 500);
}

async function connectStream() {
  const url = (els.streamUrl && els.streamUrl.value.trim()) || "";
  const data = await api("/api/stream/start", {
    method: "POST",
    body: JSON.stringify({ url }),
  });
  startVideo();
  startPolling();
  setStatus(data.message);
  await refreshStatus();
}

async function disconnectStream() {
  const data = await api("/api/stream/stop", { method: "POST", body: "{}" });
  stopVideo();
  setStatus(data.message);
  await refreshStatus();
}

function bindClick(id, handler) {
  const el = $(id);
  if (!el) return;
  el.addEventListener("click", () => {
    Promise.resolve()
      .then(() => handler())
      .catch((e) => setStatus(e.message || String(e), true));
  });
}

bindClick("btnConnect", connectStream);
bindClick("btnDisconnect", disconnectStream);

bindClick("btnPreview", () =>
  api("/api/mode/preview", { method: "POST", body: "{}" }).then((d) => {
    setStatus(d.message);
    return refreshStatus();
  })
);

bindClick("btnRecognize", () =>
  api("/api/mode/recognize", { method: "POST", body: "{}" }).then((d) => {
    setStatus(d.message);
    return refreshStatus();
  })
);

bindClick("btnEnrollStart", async () => {
  const name = (els.enrollName && els.enrollName.value.trim()) || "";
  if (!name) {
    setStatus("请先填写注册姓名/ID", true);
    els.enrollName?.focus();
    return;
  }
  if (enrollUiState.active) {
    setStatus("已在注册采集中，请点「完成注册」或「取消」", true);
    return;
  }
  const ok = await ensureStreamConnected();
  if (!ok) return;

  showEnrollPending(name);
  setStatus(`正在启动注册：${name}…`);

  const body = { name };
  const trackRaw = els.enrollTrackSelect && els.enrollTrackSelect.value;
  if (trackRaw) body.track_id = Number(trackRaw);

  try {
    const d = await api("/api/enroll/start", {
      method: "POST",
      body: JSON.stringify(body),
    });
    setStatus(d.message);
    await refreshStatus();
  } catch (e) {
    resetEnrollStartButton();
    if (els.enrollProgress) els.enrollProgress.hidden = true;
    if (els.enrollCard) els.enrollCard.classList.remove("enrolling");
    setModeBadge(lastStatus.mode || "preview");
    await refreshStatus();
    setStatus(e.message || String(e), true);
  }
});

bindClick("btnEnrollFinish", () =>
  api("/api/enroll/finish", { method: "POST", body: JSON.stringify({ allow_partial: false }) }).then(
    (d) => {
      setStatus(d.message);
      return refreshStatus();
    }
  )
);

bindClick("btnEnrollCancel", () =>
  api("/api/enroll/cancel", { method: "POST", body: "{}" }).then((d) => {
    setStatus(d.message);
    return refreshStatus();
  })
);

bindClick("btnGalleryReload", () =>
  api("/api/gallery/reload", { method: "POST", body: JSON.stringify({ path: "" }) }).then((d) => {
    setStatus(d.message);
    return refreshStatus();
  })
);

bindClick("btnGalleryDelete", () => {
  const personId = els.galleryPeople && els.galleryPeople.value;
  if (!personId) {
    setStatus("请先在列表中选择要删除的人员");
    return Promise.resolve();
  }
  const label = selectedGalleryPersonLabel();
  const ok = window.confirm(`确定删除档案库人员？\n\n${label}\n\n将删除对应 pkl 与注册信息，且不可恢复。`);
  if (!ok) return Promise.resolve();
  return api("/api/gallery/delete", {
    method: "POST",
    body: JSON.stringify({ person_id: personId }),
  }).then((d) => {
    setStatus(d.message);
    return refreshStatus();
  });
});

if (els.galleryPeople) {
  els.galleryPeople.addEventListener("change", syncGalleryDeleteButton);
}

refreshStatus();
startPolling();

if (els.enrollTrackSelect) {
  els.enrollTrackSelect.addEventListener("change", () => {
    if (suppressTrackSelectEvent) return;
    const v = els.enrollTrackSelect.value;
    if (enrollUiState.active && v) {
      selectEnrollTrack(v).catch((e) => setStatus(e.message));
    }
  });
}

if (els.recognitionResults) {
  els.recognitionResults.addEventListener("click", (ev) => {
    if (!enrollUiState.active) return;
    const card = ev.target.closest(".recog-card[data-track-id]");
    if (!card) return;
    const tid = card.getAttribute("data-track-id");
    if (!tid) return;
    selectEnrollTrack(tid).catch((e) => setStatus(e.message));
  });
}
