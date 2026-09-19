import { fillGauss } from "./rng.js";

const EPS = 1e-5;

export class Param {
  constructor(shape, name, { fill, decay = true } = {}) {
    this.name = name;
    this.shape = shape;
    this.n = shape.reduce((a, b) => a * b, 1);
    this.w = new Float32Array(this.n);
    this.g = new Float32Array(this.n);
    this.m = new Float32Array(this.n);
    this.v = new Float32Array(this.n);
    this.decay = decay;
    if (fill !== undefined) this.w.fill(fill);
  }

  zeroGrad() {
    this.g.fill(0);
  }

  zeroMoments() {
    this.m.fill(0);
    this.v.fill(0);
  }
}

function linearForward(x, W, M, K, N, y) {
  for (let m = 0; m < M; m++) {
    const xo = m * K;
    const yo = m * N;
    for (let n = 0; n < N; n++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += x[xo + k] * W[k * N + n];
      y[yo + n] = s;
    }
  }
}

function linearBackward(x, W, dy, M, K, N, dx, dW) {
  for (let m = 0; m < M; m++) {
    const xo = m * K;
    const yo = m * N;
    for (let n = 0; n < N; n++) {
      const g = dy[yo + n];
      if (g === 0) continue;
      for (let k = 0; k < K; k++) {
        dW[k * N + n] += x[xo + k] * g;
        dx[xo + k] += W[k * N + n] * g;
      }
    }
  }
}

function rmsForward(x, gain, N, C, y, inv, xhat) {
  for (let i = 0; i < N; i++) {
    const o = i * C;
    let ms = 0;
    for (let c = 0; c < C; c++) ms += x[o + c] * x[o + c];
    ms = ms / C + EPS;
    const r = 1 / Math.sqrt(ms);
    inv[i] = r;
    for (let c = 0; c < C; c++) {
      const h = x[o + c] * r;
      xhat[o + c] = h;
      y[o + c] = gain[c] * h;
    }
  }
}

function rmsBackward(dy, gain, x, inv, xhat, N, C, dx, dg) {
  for (let i = 0; i < N; i++) {
    const o = i * C;
    const r = inv[i];
    const ms = 1 / (r * r);
    let dinv = 0;
    for (let c = 0; c < C; c++) {
      const dxhat = dy[o + c] * gain[c];
      dg[c] += dy[o + c] * xhat[o + c];
      dinv += dxhat * x[o + c];
      dx[o + c] += dxhat * r;
    }
    const dms = dinv * (-0.5) * Math.pow(ms, -1.5);
    const coeff = (2 * dms) / C;
    for (let c = 0; c < C; c++) dx[o + c] += coeff * x[o + c];
  }
}

function siluForward(x, y) {
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    const s = 1 / (1 + Math.exp(Math.min(20, Math.max(-20, -v))));
    y[i] = v * s;
  }
}

function siluBackward(x, dy, dx) {
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    const s = 1 / (1 + Math.exp(Math.min(20, Math.max(-20, -v))));
    dx[i] = dy[i] * (s + v * s * (1 - s));
  }
}

function copyRange(dst, src, n) {
  for (let i = 0; i < n; i++) dst[i] = src[i];
}

class Block {
  constructor(C, nHead, nLayer, rng, maxM, maxB, maxT) {
    this.C = C;
    this.H = nHead;
    this.Dh = C / nHead;
    this.maxM = maxM;
    this.maxB = maxB;
    this.maxT = maxT;
    this.B = 1;
    this.T = maxT;

    this.ln1 = new Param([C], "ln1", { fill: 1, decay: false });
    this.Wqkv = new Param([C, 3 * C], "Wqkv");
    this.Wo = new Param([C, C], "Wo");
    this.ln2 = new Param([C], "ln2", { fill: 1, decay: false });
    this.W1 = new Param([C, 4 * C], "W1");
    this.W2 = new Param([4 * C, C], "W2");

    fillGauss(this.Wqkv.w, 0.02, rng);
    fillGauss(this.Wo.w, 0.02, rng);
    fillGauss(this.W1.w, 0.02, rng);
    fillGauss(this.W2.w, 0.02, rng);
    const resid = 1 / Math.sqrt(2 * nLayer);
    for (let i = 0; i < this.Wo.n; i++) this.Wo.w[i] *= resid;
    for (let i = 0; i < this.W2.n; i++) this.W2.w[i] *= resid;

    const C4 = 4 * C;
    this.x_in = new Float32Array(maxM * C);
    this.xn1 = new Float32Array(maxM * C);
    this.xhat1 = new Float32Array(maxM * C);
    this.inv1 = new Float32Array(maxM);
    this.qkv = new Float32Array(maxM * 3 * C);
    this.att = new Float32Array(maxB * nHead * maxT * maxT);
    this.aout = new Float32Array(maxM * C);
    this.ao = new Float32Array(maxM * C);
    this.x_mid = new Float32Array(maxM * C);
    this.xn2 = new Float32Array(maxM * C);
    this.xhat2 = new Float32Array(maxM * C);
    this.inv2 = new Float32Array(maxM);
    this.h1 = new Float32Array(maxM * C4);
    this.h1a = new Float32Array(maxM * C4);
    this.fo = new Float32Array(maxM * C);
    this.x_out = new Float32Array(maxM * C);

    this.dx_in = new Float32Array(maxM * C);
    this.dxn1 = new Float32Array(maxM * C);
    this.dqkv = new Float32Array(maxM * 3 * C);
    this.daout = new Float32Array(maxM * C);
    this.dx_mid = new Float32Array(maxM * C);
    this.dxn2 = new Float32Array(maxM * C);
    this.dh1 = new Float32Array(maxM * C4);
    this.dh1a = new Float32Array(maxM * C4);
    this.dfo = new Float32Array(maxM * C);
    this.scratch = new Float32Array(maxT);
    this.scratch2 = new Float32Array(maxT);
  }

  params() {
    return [this.ln1, this.Wqkv, this.Wo, this.ln2, this.W1, this.W2];
  }

  forward(xIn) {
    const { C, B, T } = this;
    const M = B * T;
    copyRange(this.x_in, xIn, M * C);
    rmsForward(this.x_in, this.ln1.w, M, C, this.xn1, this.inv1, this.xhat1);
    this._attnForward();
    for (let i = 0; i < M * C; i++) this.x_mid[i] = this.x_in[i] + this.ao[i];
    rmsForward(this.x_mid, this.ln2.w, M, C, this.xn2, this.inv2, this.xhat2);
    linearForward(this.xn2, this.W1.w, M, C, 4 * C, this.h1);
    siluForward(this.h1.subarray(0, M * 4 * C), this.h1a.subarray(0, M * 4 * C));
    linearForward(this.h1a, this.W2.w, M, 4 * C, C, this.fo);
    for (let i = 0; i < M * C; i++) this.x_out[i] = this.x_mid[i] + this.fo[i];
    return this.x_out;
  }

  _attnForward() {
    const { C, H, Dh, B, T } = this;
    const M = B * T;
    linearForward(this.xn1, this.Wqkv.w, M, C, 3 * C, this.qkv);
    const scale = 1 / Math.sqrt(Dh);
    const att = this.att;
    const aout = this.aout;
    const qkv = this.qkv;
    aout.fill(0);
    const scores = this.scratch;

    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        for (let q = 0; q < T; q++) {
          const qBase = (b * T + q) * (3 * C) + h * Dh;
          const row = ((b * H + h) * T + q) * T;
          let maxv = -1e9;
          for (let k = 0; k < T; k++) {
            if (k > q) {
              scores[k] = -1e9;
              continue;
            }
            const kBase = (b * T + k) * (3 * C) + C + h * Dh;
            let dot = 0;
            for (let d = 0; d < Dh; d++) dot += qkv[qBase + d] * qkv[kBase + d];
            const s = dot * scale;
            scores[k] = s;
            if (s > maxv) maxv = s;
          }
          let sum = 0;
          for (let k = 0; k < T; k++) {
            const e = k > q ? 0 : Math.exp(scores[k] - maxv);
            att[row + k] = e;
            sum += e;
          }
          const inv = sum > 0 ? 1 / sum : 0;
          for (let k = 0; k < T; k++) att[row + k] *= inv;

          const oBase = (b * T + q) * C + h * Dh;
          for (let d = 0; d < Dh; d++) aout[oBase + d] = 0;
          for (let k = 0; k <= q; k++) {
            const p = att[row + k];
            if (p === 0) continue;
            const vBase = (b * T + k) * (3 * C) + 2 * C + h * Dh;
            for (let d = 0; d < Dh; d++) aout[oBase + d] += p * qkv[vBase + d];
          }
        }
      }
    }
    linearForward(aout, this.Wo.w, M, C, C, this.ao);
  }

  _attnBackward(dao) {
    const { C, H, Dh, B, T } = this;
    const M = B * T;
    this.daout.fill(0);
    this.dqkv.fill(0);
    this.dxn1.fill(0);
    linearBackward(this.aout, this.Wo.w, dao, M, C, C, this.daout, this.Wo.g);

    const scale = 1 / Math.sqrt(Dh);
    const att = this.att;
    const qkv = this.qkv;
    const dqkv = this.dqkv;
    const daout = this.daout;
    const dp = this.scratch;
    const ds = this.scratch2;

    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        for (let q = 0; q < T; q++) {
          const qBase = (b * T + q) * (3 * C) + h * Dh;
          const row = ((b * H + h) * T + q) * T;
          const oBase = (b * T + q) * C + h * Dh;

          for (let k = 0; k <= q; k++) {
            const p = att[row + k];
            const vBase = (b * T + k) * (3 * C) + 2 * C + h * Dh;
            for (let d = 0; d < Dh; d++) dqkv[vBase + d] += p * daout[oBase + d];
          }

          let sum = 0;
          for (let k = 0; k <= q; k++) {
            const vBase = (b * T + k) * (3 * C) + 2 * C + h * Dh;
            let dP = 0;
            for (let d = 0; d < Dh; d++) dP += daout[oBase + d] * qkv[vBase + d];
            dp[k] = dP;
            sum += dP * att[row + k];
          }
          for (let k = 0; k <= q; k++) ds[k] = att[row + k] * (dp[k] - sum);

          for (let k = 0; k <= q; k++) {
            const g = ds[k] * scale;
            const kBase = (b * T + k) * (3 * C) + C + h * Dh;
            for (let d = 0; d < Dh; d++) {
              dqkv[qBase + d] += g * qkv[kBase + d];
              dqkv[kBase + d] += g * qkv[qBase + d];
            }
          }
        }
      }
    }
    linearBackward(this.xn1, this.Wqkv.w, this.dqkv, M, C, 3 * C, this.dxn1, this.Wqkv.g);
  }

  backward(dxOut) {
    const { C, B, T } = this;
    const M = B * T;
    const MC = M * C;
    const M4 = M * 4 * C;

    this.dh1a.fill(0);
    this.dh1.fill(0);
    this.dxn2.fill(0);
    this.dx_mid.fill(0);
    this.dx_in.fill(0);
    this.dfo.fill(0);

    copyRange(this.dfo, dxOut, MC);
    linearBackward(this.h1a, this.W2.w, this.dfo, M, 4 * C, C, this.dh1a, this.W2.g);
    siluBackward(this.h1.subarray(0, M4), this.dh1a.subarray(0, M4), this.dh1.subarray(0, M4));
    linearBackward(this.xn2, this.W1.w, this.dh1, M, C, 4 * C, this.dxn2, this.W1.g);
    rmsBackward(this.dxn2, this.ln2.w, this.x_mid, this.inv2, this.xhat2, M, C, this.dx_mid, this.ln2.g);
    for (let i = 0; i < MC; i++) this.dx_mid[i] += dxOut[i];

    this._attnBackward(this.dx_mid);
    rmsBackward(this.dxn1, this.ln1.w, this.x_in, this.inv1, this.xhat1, M, C, this.dx_in, this.ln1.g);
    for (let i = 0; i < MC; i++) this.dx_in[i] += this.dx_mid[i];
    return this.dx_in;
  }
}

export class TinyGPT {
  constructor({
    vocabSize,
    blockSize,
    nEmbd,
    nHead,
    nLayer,
    rng,
    maxBatch = 16,
  }) {
    if (nEmbd % nHead !== 0) throw new Error("nEmbd must divide nHead");
    this.V = vocabSize;
    this.Tmax = blockSize;
    this.C = nEmbd;
    this.H = nHead;
    this.L = nLayer;
    this.maxB = maxBatch;
    this.B = 1;
    this.T = blockSize;
    this.t = 0;
    this.loss = 0;
    this.gradNorm = 0;
    this.didForward = false;

    this.tokEmb = new Param([vocabSize, nEmbd], "tokEmb");
    this.posEmb = new Param([blockSize, nEmbd], "posEmb");
    this.lnf = new Param([nEmbd], "lnf", { fill: 1, decay: false });
    fillGauss(this.tokEmb.w, 0.02, rng);
    fillGauss(this.posEmb.w, 0.02, rng);

    const maxM = maxBatch * blockSize;
    this.blocks = [];
    for (let i = 0; i < nLayer; i++) {
      this.blocks.push(new Block(nEmbd, nHead, nLayer, rng, maxM, maxBatch, blockSize));
    }

    this.h = new Float32Array(maxM * nEmbd);
    this.hf = new Float32Array(maxM * nEmbd);
    this.xhatf = new Float32Array(maxM * nEmbd);
    this.invf = new Float32Array(maxM);
    this.logits = new Float32Array(maxM * vocabSize);
    this.probs = new Float32Array(maxM * vocabSize);
    this.dlogits = new Float32Array(maxM * vocabSize);
    this.dhf = new Float32Array(maxM * nEmbd);
    this.dxf = new Float32Array(maxM * nEmbd);
    this.lastIdx = new Int32Array(maxM);
    this.lastTargets = new Int32Array(maxM);
    this.lastMask = new Uint8Array(maxM);
    this.hasTargets = false;

    this.params = [this.tokEmb, this.posEmb, this.lnf];
    for (const block of this.blocks) this.params.push(...block.params());
  }

  countParams() {
    return this.params.reduce((s, p) => s + p.n, 0);
  }

  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }

  forward(idx, targets, B, T, mask) {
    this.B = B;
    this.T = T;
    for (const block of this.blocks) {
      block.B = B;
      block.T = T;
    }
    const { C, V } = this;
    const M = B * T;
    copyRange(this.lastIdx, idx, M);
    this.hasTargets = !!targets;
    if (targets) copyRange(this.lastTargets, targets, M);
    if (mask) copyRange(this.lastMask, mask, M);
    else this.lastMask.fill(1);

    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const id = idx[b * T + t];
        const o = (b * T + t) * C;
        const eo = id * C;
        const po = t * C;
        for (let c = 0; c < C; c++) this.h[o + c] = this.tokEmb.w[eo + c] + this.posEmb.w[po + c];
      }
    }

    let x = this.h;
    for (const block of this.blocks) x = block.forward(x);
    this.xfinal = x;
    rmsForward(x, this.lnf.w, M, C, this.hf, this.invf, this.xhatf);

    for (let m = 0; m < M; m++) {
      const ho = m * C;
      const lo = m * V;
      for (let v = 0; v < V; v++) {
        let s = 0;
        const eo = v * C;
        for (let c = 0; c < C; c++) s += this.hf[ho + c] * this.tokEmb.w[eo + c];
        this.logits[lo + v] = s;
      }
    }

    this.loss = 0;
    if (targets) {
      this.dlogits.fill(0);
      let loss = 0;
      let nSup = 0;
      for (let m = 0; m < M; m++) {
        const lo = m * V;
        let maxv = -1e30;
        for (let v = 0; v < V; v++) if (this.logits[lo + v] > maxv) maxv = this.logits[lo + v];
        let sum = 0;
        for (let v = 0; v < V; v++) {
          const e = Math.exp(this.logits[lo + v] - maxv);
          this.probs[lo + v] = e;
          sum += e;
        }
        const inv = sum > 0 ? 1 / sum : 0;
        for (let v = 0; v < V; v++) this.probs[lo + v] *= inv;
        const supervised = mask ? mask[m] : 1;
        if (!supervised) continue;
        const y = targets[m];
        const p = Math.max(this.probs[lo + y], 1e-12);
        loss += -Math.log(p);
        nSup += 1;
      }
      const invN = 1 / Math.max(1, nSup);
      for (let m = 0; m < M; m++) {
        const supervised = mask ? mask[m] : 1;
        if (!supervised) continue;
        const lo = m * V;
        const y = targets[m];
        for (let v = 0; v < V; v++) this.dlogits[lo + v] = this.probs[lo + v] * invN;
        this.dlogits[lo + y] -= invN;
      }
      this.loss = nSup ? loss / nSup : 0;
    }

    this.didForward = true;
    return { loss: this.loss, logits: this.logits, probs: this.probs };
  }

  backward() {
    if (!this.hasTargets) return;
    const { B, T, C, V } = this;
    const M = B * T;
    this.dhf.fill(0);

    for (let m = 0; m < M; m++) {
      const ho = m * C;
      const lo = m * V;
      for (let v = 0; v < V; v++) {
        const g = this.dlogits[lo + v];
        if (g === 0) continue;
        const eo = v * C;
        for (let c = 0; c < C; c++) {
          this.dhf[ho + c] += this.tokEmb.w[eo + c] * g;
          this.tokEmb.g[eo + c] += this.hf[ho + c] * g;
        }
      }
    }

    this.dxf.fill(0);
    rmsBackward(this.dhf, this.lnf.w, this.xfinal, this.invf, this.xhatf, M, C, this.dxf, this.lnf.g);

    let dx = this.dxf;
    for (let i = this.blocks.length - 1; i >= 0; i--) dx = this.blocks[i].backward(dx);

    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const id = this.lastIdx[b * T + t];
        const o = (b * T + t) * C;
        const eo = id * C;
        const po = t * C;
        for (let c = 0; c < C; c++) {
          const g = dx[o + c];
          this.tokEmb.g[eo + c] += g;
          this.posEmb.g[po + c] += g;
        }
      }
    }
  }

  step(lr, { clip = 1, wd = 0.01, b1 = 0.9, b2 = 0.95, eps = 1e-8, optim = "adam" } = {}) {
    let n2 = 0;
    for (const p of this.params) {
      for (let i = 0; i < p.n; i++) n2 += p.g[i] * p.g[i];
    }
    const n = Math.sqrt(n2);
    this.gradNorm = n;
    const scale = n > clip && clip > 0 ? clip / n : 1;
    this.t += 1;
    const t = this.t;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);

    for (const p of this.params) {
      for (let i = 0; i < p.n; i++) {
        const g = p.g[i] * scale;
        const decay = p.decay ? wd * p.w[i] : 0;
        if (optim === "sgd") {
          p.w[i] -= lr * (g + decay);
        } else {
          p.m[i] = b1 * p.m[i] + (1 - b1) * g;
          p.v[i] = b2 * p.v[i] + (1 - b2) * g * g;
          const mh = p.m[i] / bc1;
          const vh = p.v[i] / bc2;
          p.w[i] -= lr * (mh / (Math.sqrt(vh) + eps) + decay);
        }
      }
    }
  }

  trainStep(idx, targets, B, T, lr, optim, mask) {
    this.zeroGrad();
    const out = this.forward(idx, targets, B, T, mask);
    this.backward();
    this.step(lr, { optim });
    return out;
  }

  getAttention(layer, head, b = 0) {
    if (!this.didForward) return null;
    const block = this.blocks[Math.max(0, Math.min(this.blocks.length - 1, layer))];
    const { H, T } = block;
    const out = new Float32Array(T * T);
    const heads = head === "mean" || head === undefined ? [...Array(H).keys()] : [head];
    for (const h of heads) {
      const base = ((b * H + h) * T) * T;
      for (let i = 0; i < T * T; i++) out[i] += block.att[base + i];
    }
    const inv = 1 / heads.length;
    for (let i = 0; i < out.length; i++) out[i] *= inv;
    return out;
  }

  embeddingMatrix() {
    return { w: this.tokEmb.w, V: this.V, C: this.C };
  }
}
