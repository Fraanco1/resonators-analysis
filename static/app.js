"use strict";

// ── Small helpers ────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const CAT = {   // categorical slots, fixed order (validated light/dark steps)
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
  dark:  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
};
const MAX_SERIES = 8;

function isDark() {
  const t = document.documentElement.dataset.theme;
  if (t) return t === "dark";
  return matchMedia("(prefers-color-scheme: dark)").matches;
}
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function setStatus(msg, kind = "") {
  const el = $("status");
  el.textContent = msg || "";
  el.className = "status " + kind;
}

async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (_) {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

function b64ToArray(b64, Type) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(buf);
}

function fmtQ(v) {
  if (v == null || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 1 : 2) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(1) + "k";
  return v.toFixed(0);
}
function fmtPM(v, e) {
  if (v == null) return "—";
  return e != null && isFinite(e) ? `${fmtQ(v)} <span class="pm">±${fmtQ(e)}</span>` : fmtQ(v);
}
function fmtNum(v, unit = "") {
  if (v == null || !isFinite(v)) return "—";
  const a = Math.abs(v);
  let s;
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) s = v.toExponential(3);
  else s = +v.toPrecision(5) + "";
  return unit ? `${s} ${unit}` : s;
}
const GHz = (hz) => hz / 1e9;
function parseGHz(s) {
  // Accepts "5.0301", "5030.1 MHz", "5.0301e9" (Hz), "5.0301 GHz"
  if (!s) return null;
  const m = String(s).trim().match(/^([-+0-9.eE]+)\s*([kKmMgG]?)(?:[hH][zZ])?$/);
  if (!m) return null;
  let v = parseFloat(m[1]);
  if (!isFinite(v)) return null;
  const u = m[2].toLowerCase();
  if (u === "g") return v * 1e9;
  if (u === "m") return v * 1e6;
  if (u === "k") return v * 1e3;
  return v > 1e5 ? v : v * 1e9;   // bare number: Hz if huge, else GHz
}

function downloadCSV(name, rows) {
  if (!rows.length) return;
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];   // a failed first row has fewer keys
  const esc = (v) => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
  const text = [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ── State ────────────────────────────────────────────────────────────────────
const S = {
  db: null, expId: null, runs: [], runId: null, trace: 0,
  meta: null, freq: null, mag: null, phase: null,
  fits: new Map(),        // key "db|run|trace" -> array of fits
  detection: null,        // last detection on the current trace
  highlight: null,        // index of highlighted fit
  pending: null,          // selected range {f1, f2} not fitted yet
  dragmode: "select",
  batch: null,            // {rows, nRes, expId, db}
  batchOn: [],            // resonance index per color slot (null = free)
  batchAbort: false,
  loadToken: 0,
};
const fitKey = () => `${S.db}|${S.runId}|${S.trace}`;
const currentFits = () => S.fits.get(fitKey()) || [];

// ── Database / experiment / runs ─────────────────────────────────────────────
async function loadDatabases() {
  const r = await api("/api/databases");
  const sel = $("db");
  sel.innerHTML = "";
  if (!r.databases.length) {
    sel.innerHTML = `<option>No .db files under ${r.root}</option>`;
    setStatus("No databases found", "error");
    return;
  }
  for (const d of r.databases) sel.add(new Option(d.name, d.path));
  const saved = localStorageGet("db");
  if (saved && r.databases.some((d) => d.path === saved)) sel.value = saved;
  await selectDb(sel.value);
}

function localStorageGet(k) { try { return localStorage.getItem("resfit." + k); } catch (_) { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem("resfit." + k, v); } catch (_) {} }

async function selectDb(db) {
  S.db = db;
  localStorageSet("db", db);
  const exps = await api(`/api/experiments?db=${encodeURIComponent(db)}`);
  const sel = $("exp");
  sel.innerHTML = "";
  for (const e of exps) {
    const o = new Option(`${e.exp_id} · ${e.name} (${e.n_runs})${e.sample ? " — " + e.sample : ""}`, e.exp_id);
    o.disabled = e.n_runs === 0;
    sel.add(o);
  }
  const withRuns = exps.filter((e) => e.n_runs > 0);
  const saved = +localStorageGet("exp:" + db);
  const pick = withRuns.find((e) => e.exp_id === saved) || withRuns[withRuns.length - 1];
  if (pick) { sel.value = pick.exp_id; await selectExp(pick.exp_id); }
}

async function selectExp(expId) {
  S.expId = +expId;
  localStorageSet("exp:" + S.db, S.expId);
  setStatus("Loading runs…", "busy");
  S.runs = await api(`/api/runs?db=${encodeURIComponent(S.db)}&exp_id=${S.expId}`);
  renderRuns();
  setStatus("");
  if (S.runs.length) await selectRun(S.runs[0].run_id);
}

function scalarColumns(runs) {
  // Show the scalars that vary across the experiment (e.g. power, temperature).
  const vals = {};
  for (const r of runs) for (const [k, v] of Object.entries(r.scalars || {})) (vals[k] ||= { label: v.label, unit: v.unit, set: new Set() }).set.add(v.value);
  return Object.entries(vals).filter(([, v]) => v.set.size > 1).slice(0, 3).map(([k, v]) => ({ key: k, ...v }));
}

function renderRuns() {
  const cols = scalarColumns(S.runs);
  const thead = $("runs").querySelector("thead");
  thead.innerHTML = `<tr><th>run</th>${cols.map((c) => `<th title="${c.key}">${c.label}${c.unit ? ` (${c.unit})` : ""}</th>`).join("")}</tr>`;
  const tb = $("runs").querySelector("tbody");
  tb.innerHTML = "";
  for (const r of S.runs) {
    const tr = document.createElement("tr");
    tr.dataset.run = r.run_id;
    const fitted = [...S.fits.keys()].some((k) => k.startsWith(`${S.db}|${r.run_id}|`) && S.fits.get(k).length);
    tr.innerHTML = `<td class="${fitted ? "fitted" : ""}">${r.run_id}${r.completed ? "" : " ⚠"}</td>` +
      cols.map((c) => `<td>${r.scalars[c.key] ? fmtNum(r.scalars[c.key].value) : ""}</td>`).join("");
    if (r.error) tr.title = r.error;
    tr.onclick = () => selectRun(r.run_id);
    tb.appendChild(tr);
  }
  $("runCount").textContent = `${S.runs.length} runs`;
  markActiveRun();
}

function markActiveRun() {
  for (const tr of $("runs").querySelectorAll("tbody tr")) {
    const on = +tr.dataset.run === S.runId;
    tr.classList.toggle("active", on);
    if (on) tr.scrollIntoView({ block: "nearest" });
  }
}

async function selectRun(runId, trace = 0) {
  const token = ++S.loadToken;
  S.runId = runId;
  S.trace = trace;
  markActiveRun();
  setStatus(`Loading run ${runId}…`, "busy");
  try {
    const m = await api(`/api/run?db=${encodeURIComponent(S.db)}&run_id=${runId}&trace=${trace}`);
    if (token !== S.loadToken) return;   // user moved on
    // Keep the zoom when stepping to a run whose span still contains the current view.
    const view = $("plot").layout?.xaxis;
    const nf = b64ToArray(m.freq, Float64Array);
    const keepView = view && !view.autorange && view.range &&
      view.range[0] * 1e9 >= nf[0] && view.range[1] * 1e9 <= nf[nf.length - 1];
    S.meta = m;
    S.freq = nf;
    S.mag = b64ToArray(m.mag, Float32Array);
    S.phase = m.phase ? b64ToArray(m.phase, Float32Array) : null;
    S.detection = null;
    S.highlight = null;
    $("detectInfo").textContent = "";
    renderTraceSelector();
    renderParams();
    renderMainPlot(!keepView);
    renderFitTable();
    setStatus(`Run ${runId}: ${m.n_points.toLocaleString()} points` + (m.has_phase ? "" : " — no phase data, fitting disabled"), m.has_phase ? "" : "error");
  } catch (e) {
    if (token === S.loadToken) setStatus(`Run ${runId}: ${e.message}`, "error");
  }
}

function renderTraceSelector() {
  const sec = $("traceSection"), sel = $("trace");
  if (S.meta.n_traces <= 1) { sec.hidden = true; return; }
  sec.hidden = false;
  sel.innerHTML = "";
  S.meta.traces.forEach((outer, i) => {
    const desc = Object.values(outer).map((o) => `${o.label} = ${fmtNum(o.value, o.unit)}`).join(", ");
    sel.add(new Option(`${i}: ${desc}`, i));
  });
  sel.value = S.trace;
}

function renderParams() {
  const dl = $("runParams");
  const m = S.meta;
  const items = [];
  for (const v of Object.values(m.outer || {})) items.push([v.label, fmtNum(v.value, v.unit)]);
  for (const v of Object.values(m.scalars || {})) items.push([v.label, fmtNum(v.value, v.unit)]);
  if (S.freq?.length) {
    items.push(["span", `${GHz(S.freq[0]).toFixed(4)}–${GHz(S.freq[S.freq.length - 1]).toFixed(4)} GHz`]);
    items.push(["points", m.n_points.toLocaleString()]);
  }
  const r = m.roles;
  items.push(["columns", [r.mag || r.complex || (r.real && `${r.real}/${r.imag}`), r.phase].filter(Boolean).join(", ")]);
  dl.innerHTML = items.map(([k, v]) => `<dt>${k}</dt><dd title="${v}">${v}</dd>`).join("");
}

// ── Main plot ────────────────────────────────────────────────────────────────
function plotTheme() {
  return {
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: { color: css("--text-2"), size: 12, family: "system-ui, sans-serif" },
    grid: css("--grid"), text: css("--text"), data: css("--series-1"), fit: css("--series-2"),
    window: css("--window"), detect: css("--detect"), muted: css("--muted"),
  };
}

// Fit windows (orange bands with a number), detected-but-unfitted windows (gray) and
// the pending selection (dashed band that stays until it is fitted or cleared).
function buildShapes(T = plotTheme(), fits = currentFits()) {
  const shapes = [], annotations = [];
  fits.forEach((f, i) => {
    shapes.push({ type: "rect", xref: "x", yref: "paper", x0: GHz(f.f1), x1: GHz(f.f2), y0: 0, y1: 1,
      fillcolor: T.window, line: { width: i === S.highlight ? 1 : 0, color: T.fit }, layer: "below" });
    annotations.push({ x: GHz((f.result?.fr) || (f.f1 + f.f2) / 2), y: 1, xref: "x", yref: "paper", yanchor: "bottom",
      text: `${i + 1}`, showarrow: false, font: { size: 11, color: f.error ? css("--danger") : T.text } });
  });
  if (S.detection) {
    for (const d of S.detection.resonances) {
      if (fits.some((f) => d.f0 >= f.f1 && d.f0 <= f.f2)) continue;
      shapes.push({ type: "rect", xref: "x", yref: "paper", x0: GHz(d.f1), x1: GHz(d.f2), y0: 0, y1: 1,
        fillcolor: T.detect, line: { width: 0 }, layer: "below" });
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: GHz(d.f0), x1: GHz(d.f0), y0: 0, y1: 1,
        line: { color: T.muted, width: 1, dash: "dot" }, layer: "below" });
    }
  }

  if (S.pending) {
    shapes.push({ type: "rect", xref: "x", yref: "paper", x0: GHz(S.pending.f1), x1: GHz(S.pending.f2), y0: 0, y1: 1,
      fillcolor: T.window, line: { width: 1.5, color: T.fit, dash: "dash" }, layer: "below" });
    annotations.push({ x: GHz((S.pending.f1 + S.pending.f2) / 2), y: 1, xref: "x", yref: "paper", yanchor: "bottom",
      text: "selected · press Fit range", showarrow: false, font: { size: 11, color: T.text } });
  }
  return { shapes, annotations };
}

function updateShapes() {
  if ($("plot")._fullLayout) Plotly.relayout("plot", buildShapes());
}

function renderMainPlot(resetView = false) {
  const div = $("plot");
  if (!S.freq) return;
  const T = plotTheme();
  const x = Array.from(S.freq, GHz);
  const traces = [
    { x, y: S.mag, type: "scattergl", mode: "lines", name: "|S| data", line: { color: T.data, width: 1 },
      hovertemplate: "%{x:.7f} GHz<br>%{y:.3f} dB<extra></extra>", xaxis: "x", yaxis: "y" },
  ];
  if (S.phase) traces.push({ x, y: S.phase, type: "scattergl", mode: "lines", name: "phase data", line: { color: T.data, width: 1 },
    hovertemplate: "%{x:.7f} GHz<br>%{y:.2f}°<extra></extra>", xaxis: "x", yaxis: "y2", showlegend: false });

  const fits = currentFits();
  fits.forEach((f, i) => {
    if (!f.result?.model) return;
    const mx = f.result.model.freq.map(GHz);
    const w = i === S.highlight ? 3 : 2;
    traces.push({ x: mx, y: f.result.model.mag, type: "scatter", mode: "lines", name: "fit", legendgroup: "fit",
      showlegend: i === 0, line: { color: T.fit, width: w }, hovertemplate: `fit #${i + 1}<br>%{x:.7f} GHz<br>%{y:.3f} dB<extra></extra>`, xaxis: "x", yaxis: "y" });
    if (S.phase) traces.push({ x: mx, y: f.result.model.phase, type: "scatter", mode: "lines", legendgroup: "fit", showlegend: false,
      line: { color: T.fit, width: w }, hovertemplate: `fit #${i + 1}<br>%{x:.7f} GHz<br>%{y:.2f}°<extra></extra>`, xaxis: "x", yaxis: "y2" });
  });

  const { shapes, annotations } = buildShapes(T, fits);

  const prev = div._fullLayout;
  const ax = { gridcolor: T.grid, zerolinecolor: T.grid, linecolor: T.grid, tickcolor: T.grid };
  const layout = {
    paper_bgcolor: T.paper_bgcolor, plot_bgcolor: T.plot_bgcolor, font: T.font,
    margin: { l: 64, r: 16, t: 24, b: 44 },
    // "select" is our own lightweight range overlay (see bindRangeSelect), so Plotly gets no drag mode.
    dragmode: S.dragmode === "select" ? false : S.dragmode, hovermode: "closest",
    showlegend: true, legend: { orientation: "h", x: 1, xanchor: "right", y: 1.04, yanchor: "bottom", bgcolor: "rgba(0,0,0,0)" },
    xaxis: { ...ax, title: { text: "Frequency (GHz)" }, anchor: "y2", hoverformat: ".7f", exponentformat: "none" },
    yaxis: { ...ax, title: { text: "|S| (dB)" }, domain: S.phase ? [0.53, 1] : [0, 1] },
    yaxis2: { ...ax, title: { text: "Phase (°)" }, domain: [0, 0.47], visible: !!S.phase },
    shapes, annotations,
  };
  if (!resetView && prev?.xaxis) {
    // Carry the current view over explicitly so adding a fit never changes the zoom.
    for (const k of ["xaxis", "yaxis", "yaxis2"]) {
      const a = prev[k];
      if (a && !a.autorange && a.range) { layout[k].range = a.range.slice(); layout[k].autorange = false; }
    }
  }
  Plotly.react(div, traces, layout, { responsive: true, displaylogo: false, scrollZoom: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toImage"],
    modeBarButtonsToAdd: [{ name: "Save PNG", icon: Plotly.Icons.camera, click: (gd) => Plotly.downloadImage(gd, { format: "png", filename: `run${S.runId}`, scale: 2 }) }] });

  if (!div._bound) {
    div._bound = true;
    bindRangeSelect(div);
    div.on("plotly_click", (ev) => {
      // Click on a model curve highlights the matching row.
      const t = ev.points?.[0]?.data;
      if (t?.legendgroup === "fit") {
        const fits = currentFits();
        const idx = fits.findIndex((f) => f.result?.model && GHz(f.result.model.freq[0]) === t.x[0]);
        if (idx >= 0) highlightFit(idx, false);
      }
    });
  }
}

// Horizontal range selection drawn as a plain HTML overlay. Plotly's own select tool
// tests every one of the ~100k points on each mouse move, which makes dragging laggy.
function bindRangeSelect(div) {
  div.style.position = "relative";
  const box = document.createElement("div");
  box.className = "range-overlay";
  box.hidden = true;
  div.appendChild(box);
  let start = null;

  const plotArea = () => {
    const sz = div._fullLayout?._size;
    return sz && { l: sz.l, t: sz.t, w: sz.w, h: sz.h };
  };
  const localX = (e) => e.clientX - div.getBoundingClientRect().left;
  const toGHz = (px, a) => {
    const r = div._fullLayout.xaxis.range;
    return +r[0] + ((px - a.l) / a.w) * (r[1] - r[0]);
  };
  const draw = (x0, x1, a) => {
    const lo = Math.max(a.l, Math.min(x0, x1)), hi = Math.min(a.l + a.w, Math.max(x0, x1));
    Object.assign(box.style, { left: `${lo}px`, width: `${hi - lo}px`, top: `${a.t}px`, height: `${a.h}px` });
  };

  div.addEventListener("mousedown", (e) => {
    if (S.dragmode !== "select" || e.button !== 0 || !S.freq) return;
    const a = plotArea();
    if (!a) return;
    const x = localX(e), y = e.clientY - div.getBoundingClientRect().top;
    if (x < a.l || x > a.l + a.w || y < a.t || y > a.t + a.h) return;   // axes, legend, modebar
    start = { x, a };
    draw(x, x, a);
    box.hidden = false;
    e.preventDefault();
  }, true);

  window.addEventListener("mousemove", (e) => { if (start) draw(start.x, localX(e), start.a); });

  window.addEventListener("mouseup", (e) => {
    if (!start) return;
    const { x, a } = start;
    start = null;
    box.hidden = true;
    const x1 = Math.max(a.l, Math.min(a.l + a.w, localX(e)));
    if (Math.abs(x1 - x) < 4) return;   // a click, not a drag
    setRange(toGHz(x, a) * 1e9, toGHz(x1, a) * 1e9);
    if ($("fitOnSelect").checked) fitRange();
  });
}

function setRange(f1, f2, show = true) {
  if (f1 > f2) [f1, f2] = [f2, f1];
  $("f1").value = GHz(f1).toFixed(7);
  $("f2").value = GHz(f2).toFixed(7);
  S.pending = show ? { f1, f2 } : null;
  updateShapes();
}

function clearSelection() {
  S.pending = null;
  $("f1").value = $("f2").value = "";
  updateShapes();
}

function rangeFromInputs() {
  const f1 = parseGHz($("f1").value), f2 = parseGHz($("f2").value);
  S.pending = f1 != null && f2 != null && f1 !== f2 ? { f1: Math.min(f1, f2), f2: Math.max(f1, f2) } : null;
  updateShapes();
}

function setDragmode(m) {
  S.dragmode = m;
  for (const [id, mode] of [["modeSelect", "select"], ["modeZoom", "zoom"], ["modePan", "pan"]]) $(id).classList.toggle("active", mode === m);
  if ($("plot").layout) Plotly.relayout("plot", { dragmode: m === "select" ? false : m });
  $("plot").classList.toggle("selecting", m === "select");
}

function zoomTo(f1, f2, pad = 0.6) {
  const w = f2 - f1;
  Plotly.relayout("plot", { "xaxis.range": [GHz(f1 - pad * w), GHz(f2 + pad * w)], "yaxis.autorange": true, "yaxis2.autorange": true });
}

// ── Fitting ──────────────────────────────────────────────────────────────────
function fitOptions() {
  const mode = $("guessdelay").value;
  const d = parseFloat($("delayNs").value);
  return {
    port: $("port").value,
    guessdelay: mode === "guess",
    electric_delay_ns: mode === "fixed" && isFinite(d) ? d : null,
  };
}
function detectOptions() {
  const md = parseFloat($("minDepth").value), wf = parseFloat($("winFactor").value), bp = parseInt($("basePts").value);
  return {
    min_depth_db: isFinite(md) ? md : null,
    window_factor: isFinite(wf) && wf > 0 ? wf : 8,
    baseline_points: isFinite(bp) && bp > 2 ? bp : null,
  };
}

function storeFits(list) {
  list.sort((a, b) => ((a.result?.fr ?? a.f1) - (b.result?.fr ?? b.f1)));
  S.fits.set(fitKey(), list);
  const row = $("runs").querySelector(`tr[data-run="${S.runId}"] td`);
  if (row) row.classList.toggle("fitted", list.length > 0);
}

function addFit(fit) {
  // Replace a previous fit of the same resonance (overlapping windows around the same fr).
  const list = currentFits().filter((f) => {
    const fr = fit.result?.fr ?? (fit.f1 + fit.f2) / 2;
    return !(fr >= f.f1 && fr <= f.f2) && !((f.result?.fr ?? 0) >= fit.f1 && (f.result?.fr ?? 0) <= fit.f2);
  });
  list.push(fit);
  storeFits(list);
  return list.indexOf(fit);
}

async function fitRange() {
  if (!S.meta) return;
  if (!S.meta.has_phase) return setStatus("This trace has no phase data; a circle fit needs it.", "error");
  const f1 = parseGHz($("f1").value), f2 = parseGHz($("f2").value);
  if (f1 == null || f2 == null) return setStatus("Select a range on the plot or type f₁/f₂ (GHz).", "error");
  setStatus("Fitting…", "busy");
  try {
    const res = await api("/api/fit", { db: S.db, run_id: S.runId, trace: S.trace, f1, f2, fit: fitOptions() });
    const idx = addFit({ f1: res.f1, f2: res.f2, result: res, source: "manual" });
    S.highlight = idx;
    S.pending = null;   // the fit window replaces the selection band
    renderMainPlot();
    renderFitTable();
    setStatus(`Fit #${idx + 1}: fr = ${GHz(res.fr).toFixed(7)} GHz, Qi = ${fmtQ(res.Qi)}, Qc = ${fmtQ(res.Qc)}, QL = ${fmtQ(res.QL)}`);
  } catch (e) {
    setStatus(e.message, "error");
  }
}

async function detect() {
  if (!S.meta) return;
  setStatus("Detecting…", "busy");
  try {
    S.detection = await api("/api/detect", { db: S.db, run_id: S.runId, trace: S.trace, detect: detectOptions() });
    showDetectInfo(S.detection);
    renderMainPlot();
    setStatus(`${S.detection.resonances.length} resonance(s) detected`);
  } catch (e) { setStatus(e.message, "error"); }
}

function showDetectInfo(d) {
  if (!d) return;
  $("detectInfo").textContent = `${d.resonances.length} found · noise ${fmtNum(d.noise_db)} dB · threshold ${fmtNum(d.min_depth_db)} dB · baseline ${d.baseline_points} pts`;
}

async function autofit() {
  if (!S.meta) return;
  if (!S.meta.has_phase) return setStatus("This trace has no phase data; a circle fit needs it.", "error");
  setStatus("Detecting and fitting…", "busy");
  try {
    const r = await api("/api/autofit", { db: S.db, run_id: S.runId, trace: S.trace, detect: detectOptions(), fit: fitOptions() });
    S.detection = r.detection;
    showDetectInfo(r.detection);
    const list = r.results.map((x) => x.ok
      ? { f1: x.f1, f2: x.f2, result: x, source: "auto" }
      : { f1: x.f1, f2: x.f2, error: x.error, source: "auto" });
    storeFits(list);
    S.highlight = null;
    renderMainPlot();
    renderFitTable();
    const bad = list.filter((f) => f.error).length;
    setStatus(`${list.length} resonance(s) detected, ${list.length - bad} fitted` + (bad ? `, ${bad} failed` : ""), bad ? "error" : "");
  } catch (e) { setStatus(e.message, "error"); }
}

function renderFitTable() {
  const tb = $("fitTable").querySelector("tbody");
  const fits = currentFits();
  $("fitEmpty").hidden = fits.length > 0;
  tb.innerHTML = "";
  fits.forEach((f, i) => {
    const tr = document.createElement("tr");
    if (i === S.highlight) tr.className = "hl";
    const r = f.result;
    const win = ((f.f2 - f.f1) / 1e3).toFixed(0);
    tr.innerHTML = r
      ? `<td${r.flags?.length ? ` class="flag" title="${r.flags.join("; ")}"` : ""}>${i + 1}${r.flags?.length ? " ⚠" : ""}</td><td title="± ${fmtNum(r.fr_err)} Hz">${GHz(r.fr).toFixed(7)}</td>
         <td>${fmtPM(r.QL, r.QL_err)}</td><td>${fmtPM(r.Qi, r.Qi_err)}</td><td>${fmtPM(r.Qc, r.Qc_err)}</td>
         <td>${fmtNum(r.chi_square)}</td><td>${win}</td>`
      : `<td>${i + 1}</td><td class="err" colspan="5">${f.error}</td><td>${win}</td>`;
    const act = document.createElement("td");
    act.innerHTML = `<button class="icon-btn" title="Refit with current options">↻</button><button class="icon-btn" title="Remove">✕</button>`;
    const [refit, del] = act.querySelectorAll("button");
    refit.onclick = (e) => { e.stopPropagation(); setRange(f.f1, f.f2, false); fitRange(); };
    del.onclick = (e) => {
      e.stopPropagation();
      storeFits(currentFits().filter((x) => x !== f));
      S.highlight = null;
      renderMainPlot(); renderFitTable();
    };
    tr.appendChild(act);
    tr.onclick = () => highlightFit(i, true);
    tb.appendChild(tr);
  });
}

function highlightFit(i, zoom) {
  const f = currentFits()[i];
  if (!f) return;
  S.highlight = i;
  setRange(f.f1, f.f2, false);
  renderFitTable();
  renderMainPlot();
  if (zoom) zoomTo(f.f1, f.f2);
}

function fitRow(r, extra = {}) {
  return {
    ...extra,
    fr_GHz: r.fr != null ? GHz(r.fr) : null, fr_err_Hz: r.fr_err,
    QL: r.QL, QL_err: r.QL_err, Qi: r.Qi, Qi_err: r.Qi_err, Qc: r.Qc, Qc_err: r.Qc_err,
    chi_square: r.chi_square, phi0: r.phi0, delay_s: r.delay, port: r.port, flags: (r.flags || []).join("; "),
    f1_GHz: GHz(r.f1), f2_GHz: GHz(r.f2), n_points: r.n_points,
  };
}

function scalarsFlat(scalars, outer) {
  const o = {};
  for (const [k, v] of Object.entries(outer || {})) o[k] = v.value;
  for (const [k, v] of Object.entries(scalars || {})) o[k] = v.value;
  return o;
}

function exportRun() {
  const fits = currentFits().filter((f) => f.result);
  if (!fits.length) return setStatus("No fits to export", "error");
  const base = { run_id: S.runId, trace: S.trace, ...scalarsFlat(S.meta.scalars, S.meta.outer) };
  downloadCSV(`run${S.runId}_t${S.trace}_fits.csv`, fits.map((f, i) => fitRow(f.result, { resonance: i + 1, ...base })));
}

// ── Batch over the experiment ────────────────────────────────────────────────
async function runBatch() {
  const refFits = currentFits().filter((f) => f.result);
  const ref = refFits.map((f) => [f.f1, f.f2]);
  const refRun = S.runId, db = S.db, expId = S.expId;   // user may browse while the batch runs
  if (!ref.length) return setStatus("Fit at least one resonance in the current trace first (manually or with Detect & fit all).", "error");
  const runs = S.runs.slice();
  S.batchAbort = false;
  $("batchBtn").disabled = true;
  $("batchStop").hidden = false;
  $("batchProgress").hidden = false;
  const rows = [];
  const opts = { fit: fitOptions(), recenter: $("recenter").checked };
  const bar = $("batchProgress").querySelector(".bar"), lbl = $("batchProgress").querySelector(".label");
  let done = 0, total = runs.length;
  const t0 = performance.now();
  for (const run of runs) {
    if (S.batchAbort) break;
    // Runs with several traces (2D sweeps): fit each trace; the count comes back with trace 0.
    for (let t = 0, nTraces = 1; t < nTraces; t++) {
      if (S.batchAbort) break;
      try {
        const r = await api("/api/autofit", { db, run_id: run.run_id, trace: t, windows: ref, include_model: false, ...opts });
        nTraces = r.n_traces;
        for (const x of r.results) {
          rows.push({ run_id: run.run_id, trace: t, resonance: x.index + 1, ok: x.ok, error: x.error || "",
            ...scalarsFlat(run.scalars, r.outer), ...(x.ok ? fitRow(x) : {}) });
        }
      } catch (e) {
        rows.push({ run_id: run.run_id, trace: t, resonance: null, ok: false, error: e.message, ...scalarsFlat(run.scalars, {}) });
      }
    }
    done++;
    bar.style.width = `${(100 * done) / total}%`;
    const eta = ((performance.now() - t0) / done) * (total - done) / 1000;
    lbl.textContent = `${done}/${total} runs · ${eta.toFixed(0)} s left`;
  }
  $("batchBtn").disabled = false;
  $("batchStop").hidden = true;
  const nOk = rows.filter((r) => r.ok).length, nFlag = rows.filter((r) => r.ok && r.flags).length;
  lbl.textContent = `${done}/${total} runs${S.batchAbort ? " (stopped)" : ""} · ${nOk} fits · ${nFlag} flagged`;
  S.batch = { rows, nRes: ref.length, refFr: refFits.map((f) => GHz(f.result.fr)), refRun, db, expId };
  S.batchOn = Array.from({ length: MAX_SERIES }, (_, i) => (i < ref.length ? i + 1 : null));
  setupBatchControls();
  renderBatch();
  $("exportBatch").disabled = !rows.length;
  setStatus(`Batch finished: ${nOk}/${rows.length} fits converged, ${nFlag} flagged (hidden in the plot by default)`);
}

function setupBatchControls() {
  $("batchBody").hidden = false;
  const keys = new Set();
  for (const r of S.batch.rows) for (const k of Object.keys(r)) keys.add(k);
  const skip = new Set(["run_id", "trace", "resonance", "ok", "error", "fr_GHz", "fr_err_Hz", "QL", "QL_err", "Qi", "Qi_err", "Qc", "Qc_err",
    "chi_square", "phi0", "delay_s", "port", "f1_GHz", "f2_GHz", "n_points", "flags"]);
  const vars = [...keys].filter((k) => !skip.has(k));
  // Prefer a variable that actually varies across runs.
  const varies = (k) => new Set(S.batch.rows.map((r) => r[k])).size > 1;
  const opts = [...vars.filter(varies), ...vars.filter((k) => !varies(k)), "run_id"];
  for (const id of ["bX", "bX2"]) {
    const sel = $(id), cur = sel.value;
    sel.innerHTML = id === "bX2" ? `<option value="">(nothing)</option>` : "";
    for (const k of opts) sel.add(new Option(k, k));
    if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
  }
}

function batchColor(res) {
  const slot = S.batchOn.indexOf(res);
  return slot >= 0 ? (isDark() ? CAT.dark : CAT.light)[slot] : null;
}

function renderChips() {
  const box = $("resChips");
  box.innerHTML = "";
  for (let i = 1; i <= S.batch.nRes; i++) {
    const c = batchColor(i);
    const b = document.createElement("button");
    b.className = "chip" + (c ? " on" : "");
    b.innerHTML = `<span class="dot" style="${c ? `background:${c}` : ""}"></span>#${i} · ${S.batch.refFr[i - 1].toFixed(4)}`;
    b.title = c ? "Hide" : `Show (max ${MAX_SERIES} at once)`;
    b.onclick = () => {
      const slot = S.batchOn.indexOf(i);
      if (slot >= 0) S.batchOn[slot] = null;           // free the slot; others keep their colors
      else {
        const free = S.batchOn.indexOf(null);
        if (free < 0) return setStatus(`At most ${MAX_SERIES} resonances at once — hide one first.`, "error");
        S.batchOn[free] = i;
      }
      renderBatch();
    };
    box.appendChild(b);
  }
}

function renderBatch() {
  if (!S.batch) return;
  renderChips();
  const T = plotTheme();
  const yk = $("bY").value, xk = $("bX").value, x2k = $("bX2").value;
  const logY = $("bLog").checked && yk !== "fr_shift", showErr = $("bErr").checked, hideFlagged = $("bHideFlag").checked;
  const xv = (r) => (r[xk] ?? NaN) - (x2k ? (r[x2k] ?? NaN) : 0);
  const errKey = { Qi: "Qi_err", Qc: "Qc_err", QL: "QL_err", fr_shift: "fr_err_Hz" }[yk];
  const traces = [];
  for (let i = 1; i <= S.batch.nRes; i++) {
    const color = batchColor(i);
    if (!color) continue;
    let rows = S.batch.rows.filter((r) => r.resonance === i && r.ok && !(hideFlagged && r.flags)).sort((a, b) => xv(a) - xv(b));
    if (!rows.length) continue;
    const f0 = S.batch.refFr[i - 1];
    const y = rows.map((r) => (yk === "fr_shift" ? (r.fr_GHz - f0) * 1e6 : r[yk]));
    const err = errKey ? rows.map((r) => (yk === "fr_shift" ? r.fr_err_Hz / 1e3 : r[errKey])) : null;
    traces.push({
      x: rows.map(xv), y, type: "scatter", mode: "lines+markers", name: `#${i} · ${f0.toFixed(4)} GHz`,
      line: { color, width: 2 }, marker: { color, size: 8, line: { color: T.paper_bgcolor, width: 2 } },
      error_y: showErr && err ? { type: "data", array: err, visible: true, color, thickness: 1, width: 0 } : undefined,
      customdata: rows.map((r) => [r.run_id, r.trace]),
      hovertemplate: `#${i}<br>${xk}${x2k ? " − " + x2k : ""} = %{x}<br>${yk} = %{y:.4~s}<br>run %{customdata[0]}<extra></extra>`,
    });
  }
  const ax = { gridcolor: T.grid, zerolinecolor: T.grid, linecolor: T.grid };
  const yTitle = { Qi: "Qi", Qc: "Qc", QL: "QL", fr_shift: `fr − fr(run ${S.batch.refRun}) (kHz)`, chi_square: "χ²" }[yk];
  Plotly.react("batchPlot", traces, {
    paper_bgcolor: T.paper_bgcolor, plot_bgcolor: T.plot_bgcolor, font: T.font,
    margin: { l: 64, r: 16, t: 16, b: 48 }, hovermode: "closest",
    xaxis: { ...ax, title: { text: xk + (x2k ? ` − ${x2k}` : "") } },
    yaxis: { ...ax, title: { text: yTitle }, type: logY ? "log" : "linear", exponentformat: "SI" },
    legend: { orientation: "v", bgcolor: "rgba(0,0,0,0)" }, showlegend: true,
  }, { responsive: true, displaylogo: false });
  const bp = $("batchPlot");
  if (!bp._bound) {
    bp._bound = true;
    // Click a point to open that run in the main plot.
    bp.on("plotly_click", (ev) => {
      const cd = ev.points?.[0]?.customdata;
      if (cd && S.batch.db === S.db) selectRun(cd[0], cd[1]);
    });
  }
  renderBatchTable();
}

function renderBatchTable() {
  const rows = S.batch.rows;
  const allKeys = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const cols = ["run_id", "trace", "resonance", ...allKeys.filter((k) =>
    !["run_id", "trace", "resonance", "ok", "error", "port", "phi0", "delay_s", "f1_GHz", "f2_GHz", "n_points"].includes(k)), "error"];
  const uniq = [...new Set(cols)];
  $("batchTable").querySelector("thead").innerHTML = `<tr>${uniq.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const fmt = (c, v) => (v == null || v === "" ? "" : typeof v === "number" ? (["Qi", "Qc", "QL", "Qi_err", "Qc_err", "QL_err"].includes(c) ? fmtQ(v) : c === "fr_GHz" ? v.toFixed(7) : fmtNum(v)) : v);
  $("batchTable").querySelector("tbody").innerHTML = rows.map((r) =>
    `<tr>${uniq.map((c) => `<td${c === "error" && r.error ? ' class="err"' : ""}>${fmt(c, r[c])}</td>`).join("")}</tr>`).join("");
}

// ── Wiring ───────────────────────────────────────────────────────────────────
function wire() {
  $("db").onchange = (e) => selectDb(e.target.value).catch((err) => setStatus(err.message, "error"));
  $("exp").onchange = (e) => selectExp(e.target.value).catch((err) => setStatus(err.message, "error"));
  $("trace").onchange = (e) => selectRun(S.runId, +e.target.value);
  $("modeSelect").onclick = () => setDragmode("select");
  $("modeZoom").onclick = () => setDragmode("zoom");
  $("modePan").onclick = () => setDragmode("pan");
  $("resetZoom").onclick = () => Plotly.relayout("plot", { "xaxis.autorange": true, "yaxis.autorange": true, "yaxis2.autorange": true });
  $("useView").onclick = () => {
    const r = $("plot").layout?.xaxis?.range;
    if (r) setRange(r[0] * 1e9, r[1] * 1e9);
  };
  $("fitBtn").onclick = fitRange;
  for (const id of ["f1", "f2"]) {
    $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") fitRange(); });
    $(id).addEventListener("change", rangeFromInputs);
  }
  $("clearSel").onclick = clearSelection;
  $("fitOnSelect").checked = localStorageGet("fitOnSelect") === "1";
  $("fitOnSelect").onchange = (e) => localStorageSet("fitOnSelect", e.target.checked ? "1" : "0");
  $("guessdelay").onchange = () => { $("delayNs").disabled = $("guessdelay").value !== "fixed"; };
  $("port").onchange = () => {
    // Reflection fits never use the delay guess (it breaks on the 2π phase turn).
    const refl = $("port").value === "reflection", g = $("guessdelay");
    g.querySelector('option[value="guess"]').disabled = refl;
    if (refl && g.value === "guess") { g.value = "noguess"; g.onchange(); }
  };
  $("detectBtn").onclick = detect;
  $("autofitBtn").onclick = autofit;
  $("clearFits").onclick = () => { storeFits([]); S.highlight = null; renderMainPlot(); renderFitTable(); };
  $("exportRun").onclick = exportRun;
  $("batchBtn").onclick = () => runBatch().catch((e) => setStatus(e.message, "error"));
  $("batchStop").onclick = () => { S.batchAbort = true; };
  $("exportBatch").onclick = () => S.batch && downloadCSV(`exp${S.batch.expId}_batch_fits.csv`, S.batch.rows);
  for (const id of ["bY", "bX", "bX2", "bLog", "bErr", "bHideFlag"]) $(id).onchange = renderBatch;

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    const k = e.key.toLowerCase();
    if (k === "s") setDragmode("select");
    else if (k === "z") setDragmode("zoom");
    else if (k === "p") setDragmode("pan");
    else if (k === "r") $("resetZoom").click();
    else if (k === "escape") clearSelection();
    else if (k === "enter" && S.pending) fitRange();
    else if (k === "arrowdown" || k === "arrowup") {
      const i = S.runs.findIndex((r) => r.run_id === S.runId);
      const j = i + (k === "arrowdown" ? 1 : -1);
      if (S.runs[j]) { e.preventDefault(); selectRun(S.runs[j].run_id); }
    }
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderMainPlot(); renderBatch(); });
}

wire();
loadDatabases().catch((e) => setStatus(e.message, "error"));
