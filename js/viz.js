function theme() {
  const s = getComputedStyle(document.documentElement);
  const v = (name, fb) => s.getPropertyValue(name).trim() || fb;
  return {
    ink: v("--ink", "#151C17"),
    muted: v("--muted", "#4F5C54"),
    line: v("--line", "#B4C3B6"),
    forest: v("--forest", "#0B6A43"),
    vermilion: v("--vermilion", "#DC3A2C"),
    paper: v("--paper", "#F3F7F3"),
    teal: v("--teal", "#1C7488"),
    gold: v("--gold", "#C37F0C"),
    bg: v("--bg", "#DCE3DE"),
    mono: v("--mono", "monospace"),
    sans: v("--sans", "sans-serif"),
  };
}

export function fitCanvas(canvas, cssH) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = Math.max(120, canvas.clientWidth || canvas.parentElement?.clientWidth || 300);
  const h = cssH || canvas.clientHeight || 160;
  const pw = Math.round(w * dpr);
  const ph = Math.round(h * dpr);
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h, dpr };
}

export class LossChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.losses = [];
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(canvas.parentElement || canvas);
  }

  reset() {
    this.losses = [];
    this.draw();
  }

  push(y) {
    this.losses.push(y);
    if (this.losses.length > 2400) this.losses.splice(0, this.losses.length - 2000);
  }

  draw() {
    const { ctx, w, h } = fitCanvas(this.canvas, this.canvas.clientHeight || 160);
    const t = theme();
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 36, r: 8, t: 10, b: 22 };
    const iw = w - pad.l - pad.r;
    const ih = h - pad.t - pad.b;
    ctx.strokeStyle = t.line;
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, iw, ih);

    const ys = this.losses;
    ctx.fillStyle = t.muted;
    ctx.font = `11px ${t.mono}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    if (!ys.length) {
      ctx.fillText("0", pad.l - 6, pad.t + ih);
      return;
    }

    const max = Math.max(0.05, ...ys) * 1.08;
    for (let i = 0; i <= 3; i++) {
      const yv = (max * (3 - i)) / 3;
      const y = pad.t + (ih * i) / 3;
      ctx.fillStyle = t.muted;
      ctx.fillText(yv >= 10 ? yv.toFixed(0) : yv.toFixed(2), pad.l - 6, y);
      if (i > 0 && i < 3) {
        ctx.strokeStyle = t.line;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + iw, y);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    ctx.beginPath();
    for (let i = 0; i < ys.length; i++) {
      const x = pad.l + (iw * i) / Math.max(1, ys.length - 1);
      const y = pad.t + ih - (ys[i] / max) * ih;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = t.forest;
    ctx.lineWidth = 1.75;
    ctx.stroke();

    const last = ys[ys.length - 1];
    const lx = pad.l + iw;
    const ly = pad.t + ih - (last / max) * ih;
    ctx.fillStyle = t.vermilion;
    ctx.beginPath();
    ctx.arc(lx, ly, 3, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = t.muted;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("步", pad.l, pad.t + ih + 6);
  }
}

export class AttentionHeatmap {
  constructor(canvas, statusEl) {
    this.canvas = canvas;
    this.statusEl = statusEl;
    this.tokens = [];
    this.weights = null;
    this.T = 0;
    this.hover = null;
    this.pinned = null;
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(canvas.parentElement || canvas);
    canvas.addEventListener("pointermove", (e) => this._ptr(e));
    canvas.addEventListener("pointerdown", (e) => {
      this._ptr(e);
      if (this.hover) this.pinned = { q: this.hover.q, k: this.hover.k };
      this.draw();
    });
    canvas.addEventListener("pointerleave", () => {
      this.hover = this.pinned;
      this.draw();
    });
  }

  set(tokens, weights) {
    this.tokens = tokens || [];
    this.T = this.tokens.length;
    this.weights = weights;
    this.draw();
  }

  _layout(w, h) {
    const padL = 28;
    const padB = 26;
    const padT = 8;
    const padR = 8;
    const iw = w - padL - padR;
    const ih = h - padT - padB;
    const T = Math.max(1, this.T);
    const cell = Math.max(4, Math.min(iw / T, ih / T));
    const grid = cell * T;
    const ox = padL + (iw - grid) / 2;
    const oy = padT + (ih - grid) / 2;
    return { padL, padB, padT, cell, ox, oy, grid };
  }

  _ptr(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const { w, h } = { w: rect.width, h: rect.height };
    const L = this._layout(w, h);
    const q = Math.floor((y - L.oy) / L.cell);
    const k = Math.floor((x - L.ox) / L.cell);
    if (q < 0 || k < 0 || q >= this.T || k >= this.T) {
      this.hover = null;
    } else {
      this.hover = { q, k };
    }
    this.draw();
  }

  draw() {
    const height = this.canvas.clientHeight || 300;
    const { ctx, w, h } = fitCanvas(this.canvas, height);
    const t = theme();
    ctx.clearRect(0, 0, w, h);
    const T = this.T;
    if (!T || !this.weights) return;
    const L = this._layout(w, h);
    let max = 0;
    for (let i = 0; i < this.weights.length; i++) if (this.weights[i] > max) max = this.weights[i];
    max = Math.max(max, 1e-6);

    for (let q = 0; q < T; q++) {
      for (let k = 0; k < T; k++) {
        const p = this.weights[q * T + k];
        const a = k > q ? 0 : p / max;
        const x = L.ox + k * L.cell;
        const y = L.oy + q * L.cell;
        ctx.fillStyle = t.vermilion;
        ctx.globalAlpha = k > q ? 0.06 : 0.12 + 0.88 * a;
        ctx.fillRect(x, y, Math.max(1, L.cell - 1), Math.max(1, L.cell - 1));
      }
    }
    ctx.globalAlpha = 1;

    const mark = this.hover || this.pinned;
    if (mark && L.cell >= 10) {
      const { q, k } = mark;
      ctx.strokeStyle = t.ink;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(L.ox + k * L.cell, L.oy + q * L.cell, L.cell - 1, L.cell - 1);
      const p = this.weights[q * T + k] ?? 0;
      if (this.statusEl) {
        this.statusEl.textContent = `q ${this.tokens[q]} → k ${this.tokens[k]}  ${p.toFixed(3)}`;
      }
    }

    if (L.cell >= 12) {
      ctx.fillStyle = t.muted;
      ctx.font = `11px ${t.mono}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let i = 0; i < T; i++) {
        const label = this.tokens[i] ?? "";
        ctx.fillText(label, L.ox + i * L.cell + L.cell / 2, L.oy + L.grid + 12);
        ctx.save();
        ctx.translate(L.ox - 12, L.oy + i * L.cell + L.cell / 2);
        ctx.fillText(label, 0, 0);
        ctx.restore();
      }
    }
  }
}

export function pca2(mat, rows, cols) {
  if (rows <= 1) return Array.from({ length: rows }, () => ({ x: 0, y: 0 }));
  const mean = new Float64Array(cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) mean[c] += mat[r * cols + c];
  }
  for (let c = 0; c < cols; c++) mean[c] /= rows;

  const cov = new Float64Array(cols * cols);
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < cols; i++) {
      const a = mat[r * cols + i] - mean[i];
      for (let j = i; j < cols; j++) {
        const b = mat[r * cols + j] - mean[j];
        cov[i * cols + j] += a * b;
        if (i !== j) cov[j * cols + i] += a * b;
      }
    }
  }
  const inv = 1 / Math.max(1, rows - 1);
  for (let i = 0; i < cov.length; i++) cov[i] *= inv;

  const power = (exclude) => {
    const v = new Float64Array(cols);
    for (let i = 0; i < cols; i++) v[i] = 0.03 * (i + 1);
    v[0] = 1;
    if (exclude) {
      let dot = 0;
      for (let i = 0; i < cols; i++) dot += v[i] * exclude[i];
      for (let i = 0; i < cols; i++) v[i] -= dot * exclude[i];
    }
    for (let it = 0; it < 48; it++) {
      const nv = new Float64Array(cols);
      for (let i = 0; i < cols; i++) {
        let s = 0;
        for (let j = 0; j < cols; j++) s += cov[i * cols + j] * v[j];
        nv[i] = s;
      }
      if (exclude) {
        let dot = 0;
        for (let i = 0; i < cols; i++) dot += nv[i] * exclude[i];
        for (let i = 0; i < cols; i++) nv[i] -= dot * exclude[i];
      }
      let n = 0;
      for (let i = 0; i < cols; i++) n += nv[i] * nv[i];
      n = Math.sqrt(n) || 1;
      for (let i = 0; i < cols; i++) v[i] = nv[i] / n;
    }
    return v;
  };

  const e1 = power(null);
  const e2 = power(e1);
  const pts = [];
  for (let r = 0; r < rows; r++) {
    let x = 0;
    let y = 0;
    for (let c = 0; c < cols; c++) {
      const a = mat[r * cols + c] - mean[c];
      x += a * e1[c];
      y += a * e2[c];
    }
    pts.push({ x, y });
  }
  return pts;
}

export class EmbeddingAtlas {
  constructor(canvas) {
    this.canvas = canvas;
    this.labels = [];
    this.pts = [];
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(canvas.parentElement || canvas);
  }

  set(labels, matrix, rows, cols) {
    this.labels = labels;
    this.pts = pca2(matrix, rows, cols);
    this.draw();
  }

  draw() {
    const { ctx, w, h } = fitCanvas(this.canvas, this.canvas.clientHeight || 300);
    const t = theme();
    ctx.clearRect(0, 0, w, h);
    const pad = 22;
    if (!this.pts.length) return;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of this.pts) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    const sx = maxX - minX || 1;
    const sy = maxY - minY || 1;
    ctx.strokeStyle = t.line;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    ctx.font = `12px ${t.mono}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    this.pts.forEach((p, i) => {
      const x = pad + ((p.x - minX) / sx) * (w - pad * 2);
      const y = pad + (1 - (p.y - minY) / sy) * (h - pad * 2);
      ctx.fillStyle = t.teal;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = t.ink;
      ctx.fillText(this.labels[i] ?? "", x, y - 12);
    });
  }
}

export function renderProbBars(el, items, pickedId) {
  el.replaceChildren();
  for (const it of items) {
    const row = document.createElement("div");
    row.className = "bar" + (it.id === pickedId ? " is-pick" : "");
    const tok = document.createElement("span");
    tok.className = "bar-tok";
    tok.textContent = it.label;
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, it.p * 100)).toFixed(2)}%`;
    track.append(fill);
    const val = document.createElement("span");
    val.className = "bar-p tabular";
    val.textContent = it.p.toFixed(3);
    row.append(tok, track, val);
    el.append(row);
  }
}
