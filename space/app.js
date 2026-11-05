"use strict";

const DEFAULT_URL = "data/local-cpu-smollm2-135m.json";
const state = { summary: null, level: 0 };

const banner = document.querySelector("#banner");
const statusEl = document.querySelector("#status");
const titleEl = document.querySelector("#run-title");
const subEl = document.querySelector("#run-sub");
const kpis = document.querySelector("#kpis");
const configBody = document.querySelector("#config-table tbody");
const latencyBody = document.querySelector("#latency-table tbody");
const errorBody = document.querySelector("#error-table tbody");
const methodBody = document.querySelector("#method-body");
const levelSelect = document.querySelector("#level-select");

document.querySelector("#file-input").addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      show(JSON.parse(String(reader.result)), file.name);
    } catch (error) {
      fail("Could not parse that JSON file.");
    }
  };
  reader.readAsText(file);
});

levelSelect.addEventListener("change", () => {
  state.level = Number(levelSelect.value);
  drawLevel();
});

window.addEventListener("resize", () => {
  if (state.summary) drawCharts();
});

const params = new URLSearchParams(window.location.search);
const requested = params.get("summary");
const initial = requested && !/^[a-z]+:/i.test(requested) && !requested.startsWith("//")
  ? requested
  : DEFAULT_URL;

load(initial);

async function load(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error("HTTP " + response.status);
    show(await response.json(), url);
  } catch (error) {
    fail(
      "Could not load " + url + ". Serve this folder over HTTP, for example: python3 -m http.server 8741 --directory space"
    );
  }
}

function fail(message) {
  statusEl.textContent = message;
  titleEl.textContent = "No summary loaded";
}

function show(summary, source) {
  if (!summary || summary.schema_version !== 1 || !Array.isArray(summary.levels)) {
    fail("This file is not a benchlab summary (schema_version 1).");
    return;
  }
  state.summary = summary;
  state.level = 0;
  const meta = summary.meta || {};
  const synthetic = Boolean(meta.synthetic || summary.notice);
  titleEl.textContent = meta.run_id || meta.run_name || "Benchmark run";
  subEl.textContent = source + (meta.created_at ? " · " + meta.created_at : "");
  banner.hidden = !synthetic;
  if (synthetic) {
    banner.textContent =
      (summary.notice || "SYNTHETIC EXAMPLE — NOT A REAL BENCHMARK") +
      ". Produced by the mock backend's discrete-event clock so this page can be reviewed without a GPU or an API token. Not a measurement of a model server.";
  }
  statusEl.textContent = synthetic
    ? "Synthetic fixture. Replace the JSON with a summary from your own run."
    : "Loaded " + (meta.backend || "run") + " / " + (meta.model_id || "model") + ".";
  renderKpis(summary);
  renderConfig(summary);
  renderLevelSelect(summary);
  renderErrors(summary);
  renderMethod(summary);
  drawCharts();
  requestAnimationFrame(() => {
    if (state.summary === summary) drawCharts();
  });
}

function renderKpis(summary) {
  const levels = summary.levels;
  const measured = sum(levels, "n_measured");
  const errors = sum(levels, "n_errors");
  const rate = measured ? errors / measured : null;
  let best = null;
  let peak = null;
  levels.forEach((level) => {
    const p50 = level.ttft_ms && level.ttft_ms.p50;
    if (typeof p50 === "number" && (best === null || p50 < best.value)) {
      best = { value: p50, label: levelLabel(level) };
    }
    const tps = level.output_tokens_per_sec;
    if (typeof tps === "number" && (peak === null || tps > peak.value)) {
      peak = { value: tps, label: levelLabel(level) };
    }
  });
  kpis.replaceChildren(
    kpi("Measured requests", String(measured), "warmup excluded"),
    kpi("Error rate", pct(rate), errors + " failed of " + measured),
    kpi("Lowest TTFT p50", best ? fmt(best.value, 1) + " ms" : "—", best ? best.label : "no successes"),
    kpi("Peak output tok/s", peak ? fmt(peak.value, 2) : "—", peak ? peak.label : "window unavailable")
  );
}

function kpi(label, value, note) {
  const card = document.createElement("article");
  card.className = "kpi";
  const span = document.createElement("span");
  span.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  const em = document.createElement("em");
  em.textContent = note;
  card.append(span, strong, em);
  return card;
}

function renderConfig(summary) {
  const meta = summary.meta || {};
  const config = summary.config || {};
  const workload = config.workload || {};
  const prompts = summary.prompts || {};
  const rows = [
    ["Run", meta.run_id],
    ["Created", meta.created_at + (meta.created_at_is_frozen ? " (frozen for the fixture)" : "")],
    ["Backend", meta.backend],
    ["Model", meta.model_id],
    ["Revision", meta.revision || "(unpinned)"],
    ["Clock", meta.clock],
    ["Seed", meta.seed === null || meta.seed === undefined ? "(none)" : String(meta.seed)],
    ["Percentile", meta.percentile_method],
    ["Prompts", (prompts.count || "?") + " · " + (prompts.path || "")],
    ["Prompt sha256", prompts.sha256 || ""],
    ["Concurrency", (workload.concurrency || []).join(", ")],
    ["Batch sizes", (workload.batch_sizes || []).join(", ")],
    ["Warmup / level", String(workload.warmup)],
    ["Samples / level", String(workload.samples)],
    ["Max tokens", String(workload.max_tokens)],
    ["Temperature", String(workload.temperature)],
    ["Timeout (s)", String(workload.timeout_s)],
    ["benchlab", meta.package_version],
  ];
  fillTable(configBody, rows, (row) => {
    const tr = document.createElement("tr");
    row.forEach((cell, index) => {
      const td = document.createElement("td");
      td.textContent = cell;
      if (index === 1) td.className = "mono";
      tr.append(td);
    });
    return tr;
  });
}

function renderLevelSelect(summary) {
  levelSelect.replaceChildren();
  summary.levels.forEach((level, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = levelLabel(level) + " · N=" + level.n_ok + "/" + level.n_measured;
    levelSelect.append(option);
  });
  levelSelect.value = "0";
}

function renderErrors(summary) {
  const errors = summary.errors || [];
  if (!errors.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "No failed measured requests.";
    tr.append(td);
    errorBody.replaceChildren(tr);
    return;
  }
  fillTable(errorBody, errors, (row) => {
    const tr = document.createElement("tr");
    tr.className = "error";
    const cells = [
      row.request_id,
      "c" + row.concurrency + " b" + row.batch_size,
      (row.prompt_ids || []).join(", "),
      row.error_type,
      row.message,
    ];
    cells.forEach((value, index) => {
      const td = document.createElement("td");
      if (index === 3) {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = value || "";
        td.append(tag);
      } else {
        td.textContent = value || "";
      }
      tr.append(td);
    });
    return tr;
  });
}

function renderMethod(summary) {
  methodBody.replaceChildren();
  const hardware = summary.hardware || {};
  methodBody.append(paragraph(
    hardware.measurable
      ? "VRAM source: " + (hardware.source || "probe") + ". " + (hardware.note || "")
      : "VRAM not reported. " + (hardware.note || "")
  ));
  const limits = document.createElement("ul");
  (summary.limitations || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    limits.append(li);
  });
  methodBody.append(heading("Limitations"), limits);
  if ((summary.warnings || []).length) {
    const list = document.createElement("ul");
    summary.warnings.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      list.append(li);
    });
    methodBody.append(heading("Warnings for this run"), list);
  }
  const dl = document.createElement("dl");
  Object.entries(summary.definitions || {}).forEach(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = value;
    dl.append(dt, dd);
  });
  methodBody.append(heading("Definitions"), dl);
  const counts = document.createElement("ul");
  summary.levels.forEach((level) => {
    const li = document.createElement("li");
    li.textContent =
      levelLabel(level) +
      ": measured N=" + level.n_measured +
      ", successful N=" + level.n_ok +
      ", warmup excluded=" + level.n_warmup_excluded +
      ", warmup errors=" + level.n_warmup_errors +
      ", window=" + level.measurement_window_s + "s";
    counts.append(li);
  });
  methodBody.append(heading("Sample size"), counts);
}

function drawCharts() {
  drawTtft();
  drawThroughput();
  drawLevel();
}

function drawLevel() {
  const level = state.summary.levels[state.level] || state.summary.levels[0];
  drawHistogram(level);
  drawCdf(level);
  const rows = [
    ["TTFT (ms)", level.ttft_ms, false],
    ["Inter-token (ms)", level.itl_ms, false],
    ["End-to-end (ms)", level.e2e_ms, true],
  ];
  fillTable(latencyBody, rows, ([name, dist, showP99]) => {
    const tr = document.createElement("tr");
    const cells = [
      name,
      dist ? String(dist.n) : "0",
      fmt(dist && dist.mean, 3),
      fmt(dist && dist.p50, 3),
      fmt(dist && dist.p95, 3),
      showP99 ? fmt(dist && dist.p99, 3) : "—",
    ];
    cells.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    });
    return tr;
  });
}

function drawTtft() {
  const levels = state.summary.levels;
  drawGroupedBars(document.querySelector("#chart-ttft"), levels.map(levelLabel), [
    { name: "p50", color: "#9cbe93", values: levels.map((level) => num(level.ttft_ms, "p50")) },
    { name: "p95", color: "#e0b15a", values: levels.map((level) => num(level.ttft_ms, "p95")) },
  ], "ms");
}

function drawThroughput() {
  const levels = state.summary.levels;
  drawTwinLines(
    document.querySelector("#chart-throughput"),
    levels.map(levelLabel),
    {
      name: "output tok/s",
      color: "#9cbe93",
      values: levels.map((level) => level.output_tokens_per_sec),
    },
    {
      name: "req/s",
      color: "#e08a62",
      values: levels.map((level) => level.requests_per_sec),
    }
  );
}

function drawHistogram(level) {
  const canvas = document.querySelector("#chart-hist");
  const hist = (level && level.e2e_histogram) || { bin_edges_ms: [], counts: [] };
  const edges = hist.bin_edges_ms || [];
  const counts = hist.counts || [];
  const view = setup(canvas);
  clear(view);
  if (!counts.length || !edges.length) {
    empty(view, "No successful samples");
    return;
  }
  const plot = plotBox(view, 46, 16, 12, 36);
  const maxCount = Math.max(...counts, 1);
  const span = Math.max(edges[edges.length - 1] - edges[0], 1e-6);
  counts.forEach((count, index) => {
    const left = edges[index];
    const right = edges[Math.min(index + 1, edges.length - 1)];
    const width = edges[0] === edges[edges.length - 1]
      ? plot.w
      : ((right - left) / span) * plot.w;
    const x = edges[0] === edges[edges.length - 1]
      ? plot.x
      : plot.x + ((left - edges[0]) / span) * plot.w;
    const h = (count / maxCount) * plot.h;
    view.ctx.fillStyle = "#9cbe93";
    view.ctx.fillRect(x + 1, plot.y + plot.h - h, Math.max(width - 2, 1), h);
  });
  axis(view, plot, edges[0], edges[edges.length - 1], 0, maxCount, "ms", "count");
}

function drawCdf(level) {
  const canvas = document.querySelector("#chart-cdf");
  const points = (level && level.e2e_cdf) || [];
  const view = setup(canvas);
  clear(view);
  if (!points.length) {
    empty(view, "No successful samples");
    return;
  }
  const plot = plotBox(view, 46, 16, 12, 36);
  const xs = points.map((point) => point.e2e_ms);
  const min = Math.min(...xs);
  const max = Math.max(...xs);
  const span = Math.max(max - min, 1e-6);
  view.ctx.beginPath();
  view.ctx.strokeStyle = "#e0b15a";
  view.ctx.lineWidth = 2;
  points.forEach((point, index) => {
    const x = plot.x + ((point.e2e_ms - min) / span) * plot.w;
    const y = plot.y + plot.h - point.fraction * plot.h;
    if (index === 0) view.ctx.moveTo(x, y);
    else view.ctx.lineTo(x, y);
  });
  view.ctx.stroke();
  axis(view, plot, min, max, 0, 1, "ms", "fraction");
}

function drawGroupedBars(canvas, labels, series, unit) {
  const view = setup(canvas);
  clear(view);
  const values = series.flatMap((item) => item.values.filter((value) => typeof value === "number"));
  if (!labels.length || !values.length) {
    empty(view, "No successful samples");
    return;
  }
  const plot = plotBox(view, 46, 28, 12, 36);
  const max = Math.max(...values, 0) * 1.1 || 1;
  const group = plot.w / labels.length;
  const bar = Math.max(4, (group * 0.7) / series.length);
  yAxis(view, plot, 0, max, unit);
  series.forEach((item, seriesIndex) => {
    view.ctx.fillStyle = item.color;
    item.values.forEach((value, index) => {
      if (typeof value !== "number") return;
      const h = (value / max) * plot.h;
      const x = plot.x + group * index + group * 0.15 + seriesIndex * bar;
      view.ctx.fillRect(x, plot.y + plot.h - h, bar - 2, h);
    });
  });
  groupLabels(view, plot, labels);
  legend(view, series);
}

function drawTwinLines(canvas, labels, left, right) {
  const view = setup(canvas);
  clear(view);
  const leftVals = left.values.filter((value) => typeof value === "number");
  const rightVals = right.values.filter((value) => typeof value === "number");
  if (!leftVals.length && !rightVals.length) {
    empty(view, "Throughput is null");
    return;
  }
  const plot = plotBox(view, 48, 28, 48, 36);
  const leftMax = (Math.max(...leftVals, 0) * 1.15) || 1;
  const rightMax = (Math.max(...rightVals, 0) * 1.15) || 1;
  strokeLine(view, plot, left.values, leftMax, left.color);
  strokeLine(view, plot, right.values, rightMax, right.color);
  categoryAxis(view, plot, labels);
  yAxis(view, plot, 0, leftMax, "tok/s");
  yAxis(view, plot, 0, rightMax, "req/s", true);
  legend(view, [left, right]);
}

function strokeLine(view, plot, values, max, color) {
  view.ctx.beginPath();
  view.ctx.strokeStyle = color;
  view.ctx.lineWidth = 2;
  let started = false;
  values.forEach((value, index) => {
    if (typeof value !== "number") return;
    const x = plot.x + (values.length === 1 ? plot.w / 2 : (index / (values.length - 1)) * plot.w);
    const y = plot.y + plot.h - (value / max) * plot.h;
    if (!started) {
      view.ctx.moveTo(x, y);
      started = true;
    } else view.ctx.lineTo(x, y);
    view.ctx.fillStyle = color;
    view.ctx.fillRect(x - 2.5, y - 2.5, 5, 5);
  });
  view.ctx.stroke();
}

function setup(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth || 320;
  const height = canvas.clientHeight || 220;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function clear(view) {
  view.ctx.clearRect(0, 0, view.width, view.height);
}

function plotBox(view, left, top, right, bottom) {
  return { x: left, y: top, w: Math.max(10, view.width - left - right), h: Math.max(10, view.height - top - bottom) };
}

function axis(view, plot, xMin, xMax, yMin, yMax, xUnit, yUnit) {
  yAxis(view, plot, yMin, yMax, yUnit);
  const ctx = view.ctx;
  ctx.fillStyle = "#b4ae9f";
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText(fmt(xMin, 1) + " " + xUnit, plot.x, plot.y + plot.h + 16);
  ctx.fillText(fmt(xMax, 1), plot.x + plot.w, plot.y + plot.h + 16);
}

function yAxis(view, plot, min, max, unit, right) {
  const ctx = view.ctx;
  ctx.strokeStyle = "#31382d";
  ctx.fillStyle = "#b4ae9f";
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = right ? "left" : "right";
  ctx.beginPath();
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = min + ((max - min) * tick) / 4;
    const y = plot.y + plot.h - (tick / 4) * plot.h;
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.fillText(fmt(value, value >= 100 ? 0 : 2), right ? plot.x + plot.w + 6 : plot.x - 6, y + 4);
  }
  ctx.stroke();
  if (!right && unit) {
    ctx.textAlign = "left";
    ctx.fillText(unit, plot.x, plot.y - 6);
  }
}

function groupLabels(view, plot, labels) {
  const ctx = view.ctx;
  ctx.fillStyle = "#b4ae9f";
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "center";
  const group = plot.w / labels.length;
  labels.forEach((label, index) => {
    ctx.fillText(label, plot.x + group * index + group / 2, plot.y + plot.h + 16);
  });
}

function categoryAxis(view, plot, labels) {
  const ctx = view.ctx;
  ctx.fillStyle = "#b4ae9f";
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "center";
  labels.forEach((label, index) => {
    const x = plot.x + (labels.length === 1 ? plot.w / 2 : (index / (labels.length - 1)) * plot.w);
    ctx.fillText(label, x, plot.y + plot.h + 16);
  });
}

function legend(view, series) {
  const ctx = view.ctx;
  ctx.font = "12px sans-serif";
  ctx.textAlign = "left";
  let x = 52;
  series.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(x, 8, 10, 10);
    ctx.fillStyle = "#f4f0e6";
    ctx.fillText(item.name, x + 14, 17);
    x += ctx.measureText(item.name).width + 36;
  });
}

function empty(view, message) {
  view.ctx.fillStyle = "#b4ae9f";
  view.ctx.font = "14px sans-serif";
  view.ctx.textAlign = "center";
  view.ctx.fillText(message, view.width / 2, view.height / 2);
}

function fillTable(body, rows, build) {
  body.replaceChildren(...rows.map(build));
}

function heading(text) {
  const h = document.createElement("h3");
  h.textContent = text;
  return h;
}

function paragraph(text) {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}

function levelLabel(level) {
  const batches = state.summary.levels.some((item) => item.batch_size !== 1);
  return batches ? "c" + level.concurrency + "/b" + level.batch_size : "c" + level.concurrency;
}

function num(dist, key) {
  if (!dist || typeof dist[key] !== "number") return null;
  return dist[key];
}

function sum(levels, key) {
  return levels.reduce((total, level) => total + (Number(level[key]) || 0), 0);
}

function fmt(value, digits) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

function pct(value) {
  if (typeof value !== "number") return "—";
  return (value * 100).toLocaleString("en-US", { maximumFractionDigits: 2 }) + "%";
}
