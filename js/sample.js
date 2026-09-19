export function softmax(logits) {
  const n = logits.length;
  const out = new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  const inv = sum > 0 ? 1 / sum : 0;
  for (let i = 0; i < n; i++) out[i] *= inv;
  return out;
}

export function argmax(arr) {
  let b = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[b]) b = i;
  return b;
}

export function topkArgmax(logits, k) {
  const idx = Array.from({ length: logits.length }, (_, i) => i);
  idx.sort((a, b) => logits[b] - logits[a]);
  return idx.slice(0, Math.max(1, k));
}

export function samplePipeline(logits, { temperature = 1, topk = 0, topp = 1, rng }) {
  const V = logits.length;
  const raw = softmax(logits);
  if (temperature <= 1e-6) {
    const id = argmax(logits);
    const filtered = new Float32Array(V);
    filtered[id] = 1;
    return { raw, cooled: raw, filtered, id, greedy: true };
  }

  const scaled = new Float32Array(V);
  for (let i = 0; i < V; i++) scaled[i] = logits[i] / temperature;
  const cooled = softmax(scaled);

  const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => cooled[b] - cooled[a]);
  let keep = V;
  if (topk > 0 && topk < keep) keep = topk;

  let cut = keep;
  if (topp < 0.999) {
    let cum = 0;
    cut = keep;
    for (let i = 0; i < keep; i++) {
      cum += cooled[order[i]];
      if (cum >= topp) {
        cut = i + 1;
        break;
      }
    }
  }
  cut = Math.max(1, cut);

  const filtered = new Float32Array(V);
  let sum = 0;
  for (let i = 0; i < cut; i++) {
    const p = cooled[order[i]];
    filtered[order[i]] = p;
    sum += p;
  }
  if (sum <= 0) {
    filtered[order[0]] = 1;
    return { raw, cooled, filtered, id: order[0], greedy: false };
  }
  const inv = 1 / sum;
  for (let i = 0; i < V; i++) filtered[i] *= inv;

  let r = rng.next();
  let id = order[cut - 1];
  for (let i = 0; i < V; i++) {
    if (filtered[i] <= 0) continue;
    r -= filtered[i];
    if (r <= 0) {
      id = i;
      break;
    }
  }
  return { raw, cooled, filtered, id, greedy: false };
}

export function rowArgmax(logits, V, row) {
  const o = row * V;
  let b = 0;
  let bv = logits[o];
  for (let i = 1; i < V; i++) {
    const v = logits[o + i];
    if (v > bv) {
      bv = v;
      b = i;
    }
  }
  return b;
}

export function batchAccuracy(logits, targets, B, T, V, mask) {
  let ok = 0;
  let n = 0;
  const m = B * T;
  for (let i = 0; i < m; i++) {
    if (mask && !mask[i]) continue;
    n += 1;
    if (rowArgmax(logits, V, i) === targets[i]) ok++;
  }
  return n ? ok / n : 0;
}
