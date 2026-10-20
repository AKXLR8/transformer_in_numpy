/* Transformer Visualizer - vanilla JS, no dependencies. */

const EXAMPLES = [
  "akshay is running the dog is barking",
  "the cat sat on the mat",
  "deep learning models learn representations from data",
  "attention is all you need"
];

const $ = (id) => document.getElementById(id);

const state = {
  payload: null,
  seed: 0,
  heatmaps: [],
  apiEndpoint: "/api/run"
};

const NETLIFY_ENDPOINT = "/.netlify/functions/run";

/* ------------------------------------------------------------------ */
/* colour scales                                                       */
/* ------------------------------------------------------------------ */

const SEQUENTIAL = [
  [0.0, [12, 11, 9]],
  [0.25, [58, 44, 22]],
  [0.55, [138, 104, 48]],
  [0.8, [214, 175, 106]],
  [1.0, [245, 229, 196]]
];

const DIVERGING = [
  [0.0, [88, 90, 92]],
  [0.5, [20, 22, 25]],
  [1.0, [232, 220, 195]]
];

function sampleScale(stops, t) {
  const v = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, c0] = stops[i];
    const [p1, c1] = stops[i + 1];
    if (v >= p0 && v <= p1) {
      const f = p1 === p0 ? 0 : (v - p0) / (p1 - p0);
      const c = [0, 1, 2].map((k) => Math.round(c0[k] + (c1[k] - c0[k]) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(0,0,0)";
}

function fmt(v, digits = 3) {
  if (!isFinite(v)) return "-";
  if (v !== 0 && Math.abs(v) < Math.pow(10, -digits)) {
    return v.toExponential(1);
  }
  return v.toFixed(digits);
}

/* ------------------------------------------------------------------ */
/* heatmap component                                                   */
/* ------------------------------------------------------------------ */

class Heatmap {
  constructor(container, config) {
    this.container = container;
    this.config = config;
    this.matrix = config.matrix;
    this.rowLabels = config.rowLabels || [];
    this.colLabels = config.colLabels || [];
    this.masked = config.masked || null;
    this.scale = config.scale || "seq";
    this.digits = config.digits === undefined ? 2 : config.digits;
    this.title = config.title || "";
    this.hover = null;

    this.canvas = document.createElement("canvas");
    this.canvas.style.width = "100%";
    this.canvas.style.height = "auto";
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.bindEvents();
    state.heatmaps.push(this);
    this.draw();
  }

  get rows() { return this.matrix.length; }
  get cols() { return this.matrix[0] ? this.matrix[0].length : 0; }

  range() {
    let min = Infinity;
    let max = -Infinity;
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        if (this.masked && this.masked[r][c]) continue;
        const v = this.matrix[r][c];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min)) { min = 0; max = 1; }
    if (this.scale === "div") {
      const m = Math.max(Math.abs(min), Math.abs(max)) || 1;
      return { min: -m, max: m, degenerate: false };
    }
    if (max === min) return { min, max, degenerate: true };
    return { min, max, degenerate: false };
  }

  setMatrix(matrix, options) {
    this.matrix = matrix;
    if (options.scale) this.scale = options.scale;
    if (options.digits !== undefined) this.digits = options.digits;
    this.hover = null;
    this.draw();
    return this.range();
  }

  layout() {
    const width = Math.max(240, this.container.clientWidth);
    const labelW = this.rowLabels.length ? Math.min(110, Math.max(64, width * 0.22)) : 6;
    const headH = this.colLabels.length ? 24 : 6;
    const cell = Math.max(
      12,
      Math.min(46, Math.floor((width - labelW - 10) / Math.max(1, this.cols)))
    );
    return {
      labelW,
      headH,
      cell,
      width: Math.min(width, labelW + this.cols * cell + 8),
      height: headH + this.rows * cell + 6
    };
  }

  draw() {
    const { labelW, headH, cell, width, height } = this.layout();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.canvas.style.width = width + "px";
    this.canvas.style.height = height + "px";
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const { min, max, degenerate } = this.range();
    this.rangeInfo = { min, max, degenerate };

    // column headers
    if (this.colLabels.length) {
      ctx.font = "10.5px \"SF Mono\", Consolas, monospace";
      ctx.fillStyle = "#7d828d";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const step = Math.max(1, Math.ceil(this.cols / 18));
      for (let c = 0; c < this.cols; c += step) {
        const x = labelW + c * cell + cell / 2;
        const label = this.colLabels[c];
        if (cell < 26 && label.length > 4) {
          ctx.save();
          ctx.translate(x, headH / 2);
          ctx.rotate(-Math.PI / 4);
          ctx.fillText(label.slice(0, 8), 0, 0);
          ctx.restore();
        } else {
          ctx.fillText(label, x, headH / 2);
        }
      }
    }

    // row labels
    if (this.rowLabels.length) {
      ctx.font = "11px \"SF Mono\", Consolas, monospace";
      ctx.fillStyle = "#b6bac4";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let r = 0; r < this.rows; r++) {
        const y = headH + r * cell + cell / 2;
        const label = this.rowLabels[r];
        if (cell >= 20) {
          ctx.fillText(label, labelW - 9, y);
        } else {
          ctx.save();
          ctx.translate(labelW - 8, y);
          ctx.rotate(-Math.PI / 2);
          ctx.fillText(label, 0, 0);
          ctx.restore();
        }
      }
    }

    // cells
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const value = this.matrix[r][c];
        const isMasked = this.masked ? this.masked[r][c] : false;
        const x = labelW + c * cell;
        const y = headH + r * cell;

        if (isMasked) {
          ctx.fillStyle = "#0c0d11";
          ctx.fillRect(x, y, cell, cell);
          ctx.save();
          ctx.beginPath();
          ctx.rect(x, y, cell, cell);
          ctx.clip();
          ctx.strokeStyle = "rgba(255,255,255,0.09)";
          ctx.lineWidth = 1;
          for (let d = -cell; d < cell; d += 5) {
            ctx.beginPath();
            ctx.moveTo(x + d, y + cell);
            ctx.lineTo(x + d + cell, y);
            ctx.stroke();
          }
          ctx.restore();
        } else {
          const t = degenerate
            ? (this.scale === "div" ? 0.5 : 1)
            : (value - min) / (max - min || 1);
          ctx.fillStyle = sampleScale(
            this.scale === "div" ? DIVERGING : SEQUENTIAL,
            t
          );
          ctx.fillRect(x, y, cell, cell);
        }

        if (cell >= 34 && !isMasked) {
          const t = degenerate
            ? (this.scale === "div" ? 0.5 : 1)
            : (value - min) / (max - min || 1);
          const bright = this.scale === "div"
            ? Math.abs(t - 0.5) > 0.3
            : t > 0.55;
          ctx.font = "10px \"SF Mono\", Consolas, monospace";
          const text = fmt(value, this.digits);
          if (ctx.measureText(text).width <= cell - 5) {
            ctx.fillStyle = bright ? "#05060a" : "#e6e8ee";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(text, x + cell / 2, y + cell / 2);
          }
        }

        if (this.hover && this.hover.r === r && this.hover.c === c) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
        }

        ctx.strokeStyle = "rgba(255,255,255,0.07)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      }
    }
  }

  cellAt(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.layout().width / rect.width;
    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * (this.layout().height / rect.height);
    const { labelW, headH, cell } = this.layout();
    const c = Math.floor((x - labelW) / cell);
    const r = Math.floor((y - headH) / cell);
    if (r < 0 || c < 0 || r >= this.rows || c >= this.cols) return null;
    return { r, c };
  }

  bindEvents() {
    const tooltip = $("tooltip");
    this.canvas.addEventListener("mousemove", (event) => {
      const cell = this.cellAt(event.clientX, event.clientY);
      if (!cell) {
        if (this.hover) { this.hover = null; this.draw(); }
        tooltip.classList.remove("show");
        return;
      }
      if (!this.hover || this.hover.r !== cell.r || this.hover.c !== cell.c) {
        this.hover = cell;
        this.draw();
      }
      const isMasked = this.masked ? this.masked[cell.r][cell.c] : false;
      const value = this.matrix[cell.r][cell.c];
      const rowLabel = this.rowLabels[cell.r] !== undefined ? this.rowLabels[cell.r] : cell.r;
      const colLabel = this.colLabels[cell.c] !== undefined ? this.colLabels[cell.c] : cell.c;
      tooltip.innerHTML =
        `<span class="t-title">${escapeHtml(rowLabel)} → ${escapeHtml(colLabel)}</span>\n` +
        (isMasked ? "masked (causal)" : fmt(value, Math.max(3, this.digits)));
      tooltip.style.left = event.clientX + "px";
      tooltip.style.top = event.clientY + "px";
      tooltip.classList.add("show");
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.hover = null;
      this.draw();
      tooltip.classList.remove("show");
    });
  }
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

/* ------------------------------------------------------------------ */
/* legend                                                              */
/* ------------------------------------------------------------------ */

function renderLegend(containerId, range, scale, note) {
  const container = $(containerId);
  container.innerHTML = "";
  const { min, max, degenerate } = range;
  const canvas = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  const w = 150;
  const h = 9;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const stops = scale === "div" ? DIVERGING : SEQUENTIAL;
  if (degenerate) {
    ctx.fillStyle = sampleScale(stops, scale === "div" ? 0.5 : 1);
    ctx.fillRect(0, 0, w, h);
  } else {
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    for (let i = 0; i <= 10; i++) {
      grad.addColorStop(i / 10, sampleScale(stops, i / 10));
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
  }
  container.appendChild(canvas);

  const label = document.createElement("span");
  label.textContent = fmt(min, 2);
  container.appendChild(label);

  const spacer = document.createElement("span");
  spacer.style.flex = "1";
  container.appendChild(spacer);

  if (note) {
    const n = document.createElement("span");
    n.textContent = note;
    container.appendChild(n);
  }

  const maxLabel = document.createElement("span");
  maxLabel.textContent = fmt(max, 2);
  container.appendChild(maxLabel);
}

/* ------------------------------------------------------------------ */
/* renderers                                                           */
/* ------------------------------------------------------------------ */

const STEPS = [
  { id: "panel-tokenizer", name: "Tokenizer", shapeKey: "token_ids", desc: "Regex word split into vocabulary ids." },
  { id: "panel-embedding", name: "Token embedding", shapeKey: "token_embedding", desc: "Lookup of each id in the embedding table." },
  { id: "panel-positional", name: "Positional embedding", shapeKey: "position_embedding", desc: "Order information added element-wise." },
  { id: "panel-attention", name: "Self-attention", shapeKey: "attention_output", desc: "Causal scaled dot-product attention." },
  { id: "panel-ffn", name: "Feed-forward", shapeKey: "ffn_output", desc: "Linear → GELU → linear, per position." },
  { id: "panel-norm", name: "Residual + LayerNorm", shapeKey: "norm_2", desc: "Skip connection then feature normalisation." }
];

function renderPipeline(payload) {
  const track = $("pipeline-track");
  track.innerHTML = "";
  STEPS.forEach((step, i) => {
    const shape = payload.shapes[step.shapeKey] || [];
    const el = document.createElement("div");
    el.className = "step";
    el.dataset.target = step.id;
    el.innerHTML = `
      <div class="step-index">${String(i + 1).padStart(2, "0")}</div>
      <div class="step-name">${step.name}</div>
      <div class="step-shape">${shape.length ? "(" + shape.join(" × ") + ")" : "--"}</div>
      <div class="step-desc">${step.desc}</div>
      <span class="step-arrow">→</span>
    `;
    el.addEventListener("click", () => {
      document.querySelectorAll(".step").forEach((s) => s.classList.remove("active"));
      el.classList.add("active");
      const target = $(step.id);
      if (target) {
        const top = target.getBoundingClientRect().top + window.scrollY - 84;
        window.scrollTo({ top, behavior: "smooth" });
      }
    });
    track.appendChild(el);
  });
}

function renderTokens(payload) {
  const strip = $("token-strip");
  strip.innerHTML = "";
  payload.tokens.forEach((word, i) => {
    const chip = document.createElement("div");
    chip.className = "token" + (word === "<UNK>" ? " unk" : "");
    chip.innerHTML = `
      <span class="token-pos">pos ${i}</span>
      <span class="token-word">${escapeHtml(word)}</span>
      <span class="token-id">id ${payload.token_ids[i]}</span>
    `;
    strip.appendChild(chip);
  });

  $("vocab-tag").textContent =
    `${payload.vocab_size} unique words` +
    (payload.text_truncated ? " · input cut at 500 chars" : "");
  $("input-echo").textContent = payload.text || "";

  const table = $("token-table");
  table.innerHTML = "<thead><tr><th>#</th><th>Token</th><th>ID</th></tr></thead>";
  const body = document.createElement("tbody");
  payload.tokens.forEach((word, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i}</td><td>${escapeHtml(word)}</td><td>${payload.token_ids[i]}</td>`;
    body.appendChild(tr);
  });
  table.appendChild(body);

  const inSeq = new Set(payload.token_ids);
  const vocab = $("vocab-chips");
  vocab.innerHTML = "";
  payload.vocab.forEach((word, id) => {
    const chip = document.createElement("span");
    chip.className = "vocab-chip" + (id === 0 ? " unk" : "") + (inSeq.has(id) ? " in-seq" : "");
    chip.textContent = `${id} · ${word}`;
    vocab.appendChild(chip);
  });
}

function dimLabels(n) {
  return Array.from({ length: n }, (_, i) => "d" + i);
}

function renderEmbedding(payload) {
  const wrap = $("embedding-heatmap");
  wrap.innerHTML = "";
  const heatmap = new Heatmap(wrap, {
    matrix: payload.token_embedding,
    rowLabels: payload.tokens,
    colLabels: dimLabels(payload.config.embed_dim),
    scale: "div",
    digits: 3
  });
  renderLegend("embedding-legend", heatmap.rangeInfo, "div", "value");

  $("embedding-shape-tag").textContent =
    `(${payload.config.seq_len} × ${payload.config.embed_dim})`;

  const table = $("embedding-table");
  const cols = payload.config.embed_dim;
  let html = "<thead><tr><th>Token</th>";
  for (let i = 0; i < cols; i++) html += `<th class="dim-head">d${i}</th>`;
  html += "</tr></thead><tbody>";
  payload.token_embedding.forEach((row, r) => {
    html += `<tr><td>${escapeHtml(payload.tokens[r])}</td>`;
    row.forEach((v) => {
      const cls = v >= 0 ? "num pos" : "num neg";
      html += `<td class="${cls}">${fmt(v, 4)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody>";
  table.innerHTML = html;
}

function renderPositional(payload) {
  const wrap = $("position-heatmap");
  wrap.innerHTML = "";
  const heatmap = new Heatmap(wrap, {
    matrix: payload.position_embedding,
    rowLabels: payload.tokens.map((t, i) => "p" + i),
    colLabels: dimLabels(payload.config.embed_dim),
    scale: "div",
    digits: 3
  });

  $("position-shape-tag").textContent =
    `(${payload.config.seq_len} × ${payload.config.embed_dim})`;
  renderLegend("position-legend", heatmap.rangeInfo, "div", "position value");

  const stats = $("position-stats");
  const maxAbs = Math.max(
    ...payload.position_embedding.map((row) => Math.max(...row.map(Math.abs)))
  ).toFixed(4);
  const rows = [
    ["Positions used", `${payload.config.seq_len} / ${payload.config.max_seq_len}`],
    ["Max |value|", maxAbs],
    ["Embedding scale", "0.02"]
  ];
  if (payload.truncated) rows.push(["Warning", "input truncated to max length"]);
  stats.innerHTML = rows.map(([label, value]) => `
    <div class="stat">
      <div class="stat-label">${label}</div>
      <div class="stat-value">${escapeHtml(String(value))}</div>
    </div>`).join("");
}

function renderAttention(payload) {
  const wrap = $("attention-heatmap");
  wrap.innerHTML = "";
  const seq = payload.config.seq_len;
  const masked = Array.from({ length: seq }, (_, r) =>
    Array.from({ length: seq }, (_, c) => c > r)
  );
  const heatmap = new Heatmap(wrap, {
    matrix: payload.attention_weights,
    rowLabels: payload.tokens.map((t) => t),
    colLabels: payload.tokens.map((t) => t),
    masked,
    scale: "seq",
    digits: 3
  });

  const views = {
    weights: { matrix: payload.attention_weights, scale: "seq", digits: 3, note: "softmax weight" },
    scores: { matrix: payload.attention_scores, scale: "div", digits: 4, note: "Q·K / √d" }
  };

  const applyView = (name) => {
    const view = views[name];
    const range = heatmap.setMatrix(view.matrix, { scale: view.scale, digits: view.digits });
    renderLegend("attention-legend", range, view.scale, view.note);
  };

  const initialView = views[state.initialView] ? state.initialView : "weights";
  document.querySelectorAll("#attention-toggle .seg").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === initialView);
    button.onclick = () => {
      document.querySelectorAll("#attention-toggle .seg").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      applyView(button.dataset.view);
    };
  });
  applyView(initialView);

  $("attention-shape-tag").textContent =
    `Q·K ${seq}×${seq} → output (${payload.config.seq_len} × ${payload.config.embed_dim})`;

  // stacked bars per query
  const bars = $("attention-bars");
  bars.innerHTML = "";
  const palette = ["#e8dcc3", "#d98f5f", "#8fb3a3", "#c9a86a", "#b58fa8", "#a3a86a", "#b96a4a", "#8f9aa6"];
  payload.attention_weights.forEach((row, r) => {
    const rowEl = document.createElement("div");
    rowEl.className = "bar-row";
    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = payload.tokens[r] || "r" + r;
    const track = document.createElement("div");
    track.className = "bar-track";
    row.forEach((v, c) => {
      const seg = document.createElement("div");
      seg.className = "bar-seg";
      seg.style.width = (v * 100).toFixed(3) + "%";
      seg.style.background = palette[c % palette.length];
      seg.title = `${payload.tokens[c]}: ${fmt(v, 3)}`;
      track.appendChild(seg);
    });
    const val = document.createElement("span");
    val.className = "bar-value";
    val.textContent = "Σ 1.00";
    rowEl.append(label, track, val);
    bars.appendChild(rowEl);
  });

  // top pairs table
  const pairs = [];
  payload.attention_weights.forEach((row, r) => {
    row.forEach((v, c) => {
      if (c <= r) pairs.push({ q: payload.tokens[r], k: payload.tokens[c], v });
    });
  });
  pairs.sort((a, b) => b.v - a.v);
  const table = $("attention-table");
  let html = "<thead><tr><th>Query</th><th>Key</th><th>Weight</th></tr></thead><tbody>";
  pairs.slice(0, 15).forEach((p) => {
    html += `<tr><td>${escapeHtml(p.q)}</td><td>${escapeHtml(p.k)}</td><td>${fmt(p.v, 4)}</td></tr>`;
  });
  html += "</tbody>";
  table.innerHTML = html;
}

function renderFFN(payload) {
  const hiddenWrap = $("ffn-hidden-heatmap");
  hiddenWrap.innerHTML = "";
  const hidden = new Heatmap(hiddenWrap, {
    matrix: payload.ffn_hidden,
    rowLabels: payload.tokens,
    colLabels: payload.ffn_hidden[0].map((_, i) => "h" + i),
    scale: "div",
    digits: 3
  });
  renderLegend("ffn-hidden-legend", hidden.rangeInfo, "div", "GELU");

  const outWrap = $("ffn-output-heatmap");
  outWrap.innerHTML = "";
  const out = new Heatmap(outWrap, {
    matrix: payload.ffn_output,
    rowLabels: payload.tokens,
    colLabels: dimLabels(payload.config.embed_dim),
    scale: "div",
    digits: 3
  });
  renderLegend("ffn-output-legend", out.rangeInfo, "div", "FFN output");

  $("ffn-shape-tag").textContent =
    `hidden (${payload.config.seq_len} × ${payload.config.hidden_dim}) → out (${payload.config.seq_len} × ${payload.config.embed_dim})`;
}

function renderNorm(payload) {
  const wrap = $("norm-heatmap");
  wrap.innerHTML = "";
  const heatmap = new Heatmap(wrap, {
    matrix: payload.norm_2,
    rowLabels: payload.tokens,
    colLabels: dimLabels(payload.config.embed_dim),
    scale: "div",
    digits: 3
  });

  $("norm-shape-tag").textContent = `(${payload.config.seq_len} × ${payload.config.embed_dim})`;
  renderLegend("norm-legend", heatmap.rangeInfo, "div", "normalised value");

  const means = payload.norm_2.map((row) => row.reduce((a, b) => a + b, 0) / row.length);
  const meanOfMeans = means.reduce((a, b) => a + b, 0) / means.length;
  const rowVar = payload.norm_2.map((row) => {
    const m = row.reduce((a, b) => a + b, 0) / row.length;
    return row.reduce((a, b) => a + (b - m) ** 2, 0) / row.length;
  });
  const avgVar = rowVar.reduce((a, b) => a + b, 0) / rowVar.length;

  $("norm-stats").innerHTML = `
    <div class="stat"><div class="stat-label">Mean of row means</div><div class="stat-value">${fmt(meanOfMeans, 4)}</div></div>
    <div class="stat"><div class="stat-label">Average variance</div><div class="stat-value">${fmt(avgVar, 4)}</div></div>
    <div class="stat"><div class="stat-label">Residual 1 shape</div><div class="stat-value">(${payload.shapes.residual_1.join(" × ")})</div></div>
    <div class="stat"><div class="stat-label">Residual 2 shape</div><div class="stat-value">(${payload.shapes.residual_2.join(" × ")})</div></div>
  `;
}

function renderParams(payload) {
  const labels = {
    token_embedding: "Token embedding",
    position_embedding: "Positional embedding",
    attention: "Self-attention (Wq,Wk,Wv,Wo)",
    feed_forward: "Feed-forward (W1,b1,W2,b2)",
    layer_norm: "LayerNorm (gamma,beta)",
    total: "Total"
  };
  const table = $("param-table");
  let html = "<thead><tr><th>Component</th><th>Parameters</th></tr></thead><tbody>";
  Object.keys(labels).forEach((key) => {
    const value = payload.params[key];
    const strong = key === "total" ? ' style="font-weight:700"' : "";
    html += `<tr><td${strong}>${labels[key]}</td><td class="num"${strong}>${value.toLocaleString()}</td></tr>`;
  });
  html += "</tbody>";
  table.innerHTML = html;

  const shapeLabels = {
    token_ids: "Token IDs",
    token_embedding: "Token embedding",
    position_embedding: "Positional embedding",
    combined: "Combined (x)",
    attention_output: "Attention output",
    residual_1: "Residual 1",
    norm_1: "LayerNorm 1",
    ffn_hidden: "FFN hidden (GELU)",
    ffn_output: "FFN output",
    residual_2: "Residual 2",
    norm_2: "LayerNorm 2 (final)"
  };
  const shapeTable = $("shape-table");
  let sHtml = "<thead><tr><th>Tensor</th><th>Shape</th></tr></thead><tbody>";
  Object.keys(shapeLabels).forEach((key) => {
    const shape = payload.shapes[key] || [];
    sHtml += `<tr><td>${shapeLabels[key]}</td><td class="num">(${shape.join(", ")})</td></tr>`;
  });
  sHtml += "</tbody>";
  shapeTable.innerHTML = sHtml;

  $("param-tag").textContent = payload.params.total.toLocaleString() + " params";
}

/* ------------------------------------------------------------------ */
/* data flow                                                           */
/* ------------------------------------------------------------------ */

function setStatus(kind, label) {
  const pill = $("status-pill");
  pill.className = "pill" + (kind ? " " + kind : "");
  $("status-text").textContent = label;
}

function showError(message) {
  $("error-text").textContent = message;
  $("error-banner").hidden = false;
}

function clearError() {
  $("error-banner").hidden = true;
  $("error-text").textContent = "";
}

function resetHeatmaps() {
  state.heatmaps.forEach((h) => h.canvas.remove());
  state.heatmaps = [];
}

function render(payload) {
  if (!payload || !payload.tokens || !payload.tokens.length) {
    showError("Nothing to visualise — the input produced no word tokens.");
    return;
  }
  state.payload = payload;
  resetHeatmaps();
  renderPipeline(payload);
  renderTokens(payload);
  renderEmbedding(payload);
  renderPositional(payload);
  renderAttention(payload);
  renderFFN(payload);
  renderNorm(payload);
  renderParams(payload);
  $("pipeline-note").textContent =
    `${payload.config.seq_len} tokens · embed ${payload.config.embed_dim} · hidden ${payload.config.hidden_dim}`;
}

async function fetchPayload(params) {
  let response = await fetch(state.apiEndpoint + "?" + params.toString());
  // static host without the /api/run redirect -> talk to the function directly
  if (response.status === 404 && state.apiEndpoint !== NETLIFY_ENDPOINT) {
    state.apiEndpoint = NETLIFY_ENDPOINT;
    response = await fetch(state.apiEndpoint + "?" + params.toString());
  }
  return response;
}

let requestId = 0;

async function run() {
  const btn = $("run-btn");
  btn.disabled = true;
  setStatus("busy", "Running");
  const myRequest = ++requestId;
  const params = new URLSearchParams({
    text: $("text-input").value,
    embed_dim: $("embed-dim").value,
    hidden_dim: $("hidden-dim").value,
    max_seq_len: $("max-seq").value,
    seed: state.seed
  });
  const started = performance.now();
  try {
    const response = await fetchPayload(params);
    const data = await response.json().catch(() => null);
    if (!data) {
      throw new Error(
        "The server returned a non-JSON response (HTTP " + response.status + ")."
      );
    }
    if (data.error) throw new Error(data.error);
    if (!response.ok) {
      throw new Error("HTTP " + response.status + ": " + (data.error || "request failed"));
    }
    if (myRequest !== requestId) return;   // a newer request already won
    render(data);
    const elapsed = Math.round(performance.now() - started);
    $("run-time").textContent = elapsed + " ms";
    setStatus("", "Live");
    clearError();
  } catch (error) {
    if (myRequest !== requestId) return;   // ignore a superseded failure
    console.error(error);
    setStatus("error", "Error");
    showError(error.message || "Could not reach the server. Is it still running?");
  } finally {
    if (myRequest === requestId) btn.disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* wiring                                                              */
/* ------------------------------------------------------------------ */

function debounce(fn, wait) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

function init() {
  const params = new URLSearchParams(window.location.search);
  if (params.get("text")) {
    $("text-input").value = params.get("text");
  }
  if (params.get("view")) {
    state.initialView = params.get("view");
  }

  const chips = $("example-chips");
  EXAMPLES.forEach((text) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.textContent = text.length > 34 ? text.slice(0, 32) + "…" : text;
    chip.addEventListener("click", () => {
      $("text-input").value = text;
      run();
    });
    chips.appendChild(chip);
  });

  $("run-btn").addEventListener("click", run);
  $("error-close").addEventListener("click", clearError);
  $("shuffle-btn").addEventListener("click", () => {
    state.seed = Math.floor(Math.random() * 100000);
    run();
  });

  const textInput = $("text-input");
  textInput.addEventListener("input", debounce(run, 450));

  const bindings = [
    ["embed-dim", "embed-dim-value"],
    ["hidden-dim", "hidden-dim-value"],
    ["max-seq", "max-seq-value"]
  ];
  bindings.forEach(([id, labelId]) => {
    const input = $(id);
    const label = $(labelId);
    input.addEventListener("input", () => { label.textContent = input.value; });
    input.addEventListener("change", run);
  });

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      state.heatmaps.forEach((h) => h.draw());
    }, 120);
  });

  run();
}

document.addEventListener("DOMContentLoaded", init);
