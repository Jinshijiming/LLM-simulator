import { RNG } from "./rng.js";
import { CharTokenizer } from "./tokenizer.js";
import { PRESETS, parseCorpus, flattenContents, packCorpus, loopStream, makeBatch, chatPrefix } from "./datasets.js";
import { TinyGPT } from "./gpt.js";
import { samplePipeline, rowArgmax, batchAccuracy, softmax } from "./sample.js";
import { LossChart, AttentionHeatmap, EmbeddingAtlas, renderProbBars } from "./viz.js";
import { drawArchDiagram } from "./diagram.js";

const $ = (id) => document.getElementById(id);

const LR_MIN = 1e-4;
const LR_MAX = 0.03;

function lrFromSlider(v) {
  const t = Number(v) / 100;
  return LR_MIN * Math.pow(LR_MAX / LR_MIN, t);
}

function sliderFromLr(lr) {
  return Math.round((100 * Math.log(lr / LR_MIN)) / Math.log(LR_MAX / LR_MIN));
}

function fmtParams(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function fmtLr(x) {
  if (x >= 0.001) return x.toFixed(3);
  return x.toExponential(1);
}

function fmtNum(x, d = 3) {
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(d);
}

function headsFor(d) {
  return [1, 2, 4, 8].filter((h) => d % h === 0 && h <= d);
}

function announce(text) {
  $("live").textContent = text;
}

function icons() {
  try {
    if (window.lucide && window.lucide.createIcons) {
      window.lucide.createIcons({ attrs: { width: 16, height: 16 } });
    }
  } catch (err) {}
}

function setIcon(btn, name) {
  if (!btn || !name) return;
  const old = btn.querySelector("[data-lucide], svg");
  const i = document.createElement("i");
  i.setAttribute("data-lucide", name);
  i.setAttribute("aria-hidden", "true");
  if (old) old.replaceWith(i);
  else btn.insertBefore(i, btn.firstChild);
}

function showError(err) {
  const el = document.getElementById("boot-error");
  if (!el) return;
  el.hidden = false;
  el.textContent = String(err && err.message ? err.message : err);
}

const saved = (() => {
  try {
    return JSON.parse(localStorage.getItem("llm-lab-v2") || "null");
  } catch {
    return null;
  }
})();
const savedText = saved?.text && !parseCorpus(saved.text).error ? saved.text : PRESETS[0].text;
const savedPreset = savedText === saved?.text ? saved?.preset || "qa" : "qa";

const state = {
  view: "data",
  preset: savedPreset,
  text: savedText,
  seed: 42,
  hparams: {
    nLayer: saved?.hparams?.nLayer ?? 2,
    nEmbd: saved?.hparams?.nEmbd ?? 32,
    nHead: saved?.hparams?.nHead ?? 4,
    blockSize: saved?.hparams?.blockSize ?? 24,
    batchSize: saved?.hparams?.batchSize ?? 8,
    lr: saved?.hparams?.lr ?? 0.006,
    optim: saved?.hparams?.optim ?? "adam",
  },
  speed: 8,
  conversations: [],
  corpusError: null,
  tokenizer: null,
  data: null,
  model: null,
  rng: null,
  trainRng: new RNG(7),
  sampleRng: new RNG(99),
  step: 0,
  playing: false,
  lastLoss: null,
  lastAcc: null,
  lastBatch: null,
  forwardKind: null,
  vizLayer: 1,
  vizHead: "mean",
  archPart: "embed",
  gen: {
    prompt: saved?.prompt ?? PRESETS[0].prompt,
    ids: [],
    origin: [],
    roles: [],
    last: null,
    ctx: null,
    running: false,
    temp: saved?.temp ?? 0.8,
    topk: saved?.topk ?? 0,
    topp: saved?.topp ?? 1,
  },
};

if (!headsFor(state.hparams.nEmbd).includes(state.hparams.nHead)) {
  state.hparams.nHead = headsFor(state.hparams.nEmbd).at(-1);
}

const lossChart = new LossChart($("loss-chart"));
const attnTrain = new AttentionHeatmap($("attn-train"), $("attn-train-status"));
const attnGen = new AttentionHeatmap($("attn-gen"), $("attn-gen-status"));
const atlas = new EmbeddingAtlas($("embed-atlas"));

function persist() {
  try {
    localStorage.setItem(
      "llm-lab-v2",
      JSON.stringify({
        preset: state.preset,
        text: state.text,
        hparams: state.hparams,
        prompt: state.gen.prompt,
        temp: state.gen.temp,
        topk: state.gen.topk,
        topp: state.gen.topp,
      }),
    );
  } catch (err) {}
}

function snapshotBatch(x, y, logits, B, T, extra = {}) {
  const V = state.tokenizer.vocabSize;
  return {
    x: Int32Array.from(x.subarray(0, B * T)),
    y: Int32Array.from(y.subarray(0, B * T)),
    logits: Float32Array.from(logits.subarray(0, B * T * V)),
    mask: extra.mask ? Uint8Array.from(extra.mask.subarray(0, B * T)) : null,
    rolesX: extra.rolesX ? extra.rolesX.slice(0, B * T) : null,
    rolesY: extra.rolesY ? extra.rolesY.slice(0, B * T) : null,
    B,
    T,
  };
}

function previewBatch() {
  const B = state.hparams.batchSize;
  const T = state.hparams.blockSize;
  const batch = makeBatch(state.data, B, T, new RNG(11));
  const out = state.model.forward(batch.x, batch.y, B, T, batch.mask);
  state.forwardKind = "train";
  state.lastLoss = out.loss;
  state.lastAcc = batchAccuracy(out.logits, batch.y, B, T, state.tokenizer.vocabSize, batch.mask);
  state.lastBatch = snapshotBatch(batch.x, batch.y, out.logits, B, T, batch);
}

function rebuild() {
  state.playing = false;
  state.gen.running = false;
  syncPlayButtons();
  const parsed = parseCorpus(state.text);
  state.conversations = parsed.conversations;
  state.corpusError = parsed.error;
  state.tokenizer = new CharTokenizer(flattenContents(parsed.conversations));
  const packed = packCorpus(parsed.conversations, state.tokenizer);
  state.data = loopStream(packed, state.hparams.blockSize * 12);
  state.rng = new RNG(state.seed);
  state.model = new TinyGPT({
    vocabSize: state.tokenizer.vocabSize,
    blockSize: state.hparams.blockSize,
    nEmbd: state.hparams.nEmbd,
    nHead: state.hparams.nHead,
    nLayer: state.hparams.nLayer,
    rng: state.rng,
  });
  state.step = 0;
  state.lastLoss = null;
  state.lastAcc = null;
  state.lastBatch = null;
  state.forwardKind = null;
  state.vizLayer = state.hparams.nLayer - 1;
  lossChart.reset();
  resetGen(false);
  previewBatch();
  persist();
  renderAll();
}

function tokenEl(id, cls = "", onClick) {
  const el = document.createElement("button");
  el.type = "button";
  if (state.tokenizer.isSpecial(id)) cls = `${cls} special`.trim();
  el.className = `tok ${cls}`.trim();
  const ch = document.createElement("span");
  ch.className = "ch";
  ch.textContent = state.tokenizer.displayId(id);
  const n = document.createElement("span");
  n.className = "id";
  n.textContent = String(id);
  el.append(ch, n);
  if (onClick) el.addEventListener("click", onClick);
  return el;
}

function fillRow(node, ids, classFor, onClick) {
  node.replaceChildren();
  ids.forEach((id, i) =>
    node.append(tokenEl(id, classFor ? classFor(id, i) : "", onClick ? () => onClick(i, id) : undefined)),
  );
}

function inspectTrain(t) {
  if (!state.lastBatch) return;
  state.inspectT = t;
  const { x, y, logits, T } = state.lastBatch;
  if (t < 0 || t >= T) return;
  const V = state.tokenizer.vocabSize;
  const pred = rowArgmax(logits, V, t);
  const probs = softmax(logits.subarray(t * V, (t + 1) * V));
  const line =
    "t " +
    t +
    "  " +
    state.tokenizer.displayId(x[t]) +
    " → " +
    state.tokenizer.displayId(y[t]) +
    "  pred " +
    state.tokenizer.displayId(pred) +
    "  p=" +
    probs[y[t]].toFixed(3);
  const el = document.getElementById("batch-inspect");
  if (el) el.textContent = line;
  announce(line);
  for (const row of ["row-in", "row-y", "row-p"]) {
    const kids = $(row).children;
    for (let i = 0; i < kids.length; i++) kids[i].classList.toggle("is-on", i === t);
  }
  attnTrain.pinned = { q: t, k: t };
  attnTrain.hover = { q: t, k: t };
  renderAttention();
}

function renderMeters() {
  $("m-vocab").textContent = String(state.tokenizer.vocabSize);
  $("m-params").textContent = fmtParams(state.model.countParams());
  $("m-step").textContent = String(state.step);
  $("m-loss").textContent = state.lastLoss == null ? "—" : fmtNum(state.lastLoss, 3);
}

function renderHeadSelectors() {
  const layerBox = $("layer-sel");
  const headBox = $("head-sel");
  layerBox.replaceChildren();
  headBox.replaceChildren();
  for (let i = 0; i < state.hparams.nLayer; i++) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = `L${i + 1}`;
    b.setAttribute("aria-pressed", String(state.vizLayer === i));
    b.addEventListener("click", () => {
      state.vizLayer = i;
      renderHeadSelectors();
      renderAttention();
    });
    layerBox.append(b);
  }
  const heads = ["mean", ...Array.from({ length: state.hparams.nHead }, (_, i) => i)];
  for (const h of heads) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = h === "mean" ? "平均" : String(h + 1);
    b.setAttribute("aria-pressed", String(state.vizHead === h));
    b.addEventListener("click", () => {
      state.vizHead = h;
      renderHeadSelectors();
      renderAttention();
    });
    headBox.append(b);
  }
}

function labelsFor(ids) {
  return Array.from(ids, (id) => state.tokenizer.displayId(id));
}

function ensureForward(kind) {
  if (kind === "train" && state.forwardKind !== "train" && state.lastBatch) {
    const { x, y, B, T, mask } = state.lastBatch;
    state.model.forward(x, y, B, T, mask);
    state.forwardKind = "train";
  }
  if (kind === "gen" && state.forwardKind !== "gen" && state.gen.ctx) {
    state.model.forward(state.gen.ctx, null, 1, state.gen.ctx.length);
    state.forwardKind = "gen";
  }
}

function renderAttention() {
  if (state.view === "train" || state.lastBatch) {
    if (state.lastBatch) {
      ensureForward("train");
      const T = state.lastBatch.T;
      const ids = state.lastBatch.x.subarray(0, T);
      attnTrain.set(labelsFor(ids), state.model.getAttention(state.vizLayer, state.vizHead, 0));
    }
  }
  if (state.gen.ctx) {
    const restore = state.forwardKind;
    ensureForward("gen");
    attnGen.set(labelsFor(state.gen.ctx), state.model.getAttention(state.vizLayer, state.vizHead, 0));
    if (restore === "train") ensureForward("train");
  }
}

function renderEmbed() {
  const { w, V, C } = state.model.embeddingMatrix();
  const labels = state.tokenizer.chars.map((c) => state.tokenizer.display(c));
  atlas.set(labels, w, V, C);
}

function renderTrainBatch() {
  if (!state.lastBatch) {
    $("row-in").replaceChildren();
    $("row-y").replaceChildren();
    $("row-p").replaceChildren();
    return;
  }
  const { x, y, logits, T } = state.lastBatch;
  const V = state.tokenizer.vocabSize;
  const xs = Array.from(x.subarray(0, T));
  const ys = Array.from(y.subarray(0, T));
  const ps = ys.map((_, t) => rowArgmax(logits, V, t));
  const inspect = (i) => inspectTrain(i);
  const rolesX = state.lastBatch.rolesX || [];
  const rolesY = state.lastBatch.rolesY || [];
  const ymask = state.lastBatch.mask;
  fillRow($("row-in"), xs, (id, i) => (rolesX[i] ? "role-" + rolesX[i] : ""), inspect);
  fillRow($("row-y"), ys, (id, i) => {
    const role = rolesY[i] ? "role-" + rolesY[i] : "";
    const mute = ymask && !ymask[i] ? "masked" : "";
    return (role + " " + mute).trim();
  }, inspect);
  fillRow($("row-p"), ps, (id, i) => {
    const hit = id === ys[i] ? "match" : "miss";
    const mute = ymask && !ymask[i] ? "masked" : "";
    return (hit + " " + mute).trim();
  }, inspect);
  $("loss-now").textContent = fmtNum(state.lastLoss, 3);
  $("ppl-now").textContent = fmtNum(Math.exp(Math.min(20, state.lastLoss)), 2);
  $("acc-now").textContent = `${Math.round(state.lastAcc * 100)}%`;
  $("gn-now").textContent = state.step ? fmtNum(state.model.gradNorm, 2) : "—";
}

function renderData() {
  $("corpus").value = state.text;
  const st = $("corpus-status");
  if (st) {
    st.textContent = state.corpusError || "";
    st.classList.toggle("is-bad", !!state.corpusError);
  }
  const box = $("presets");
  box.replaceChildren();
  for (const p of PRESETS) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = p.name;
    b.setAttribute("aria-pressed", String(state.preset === p.id));
    b.addEventListener("click", () => {
      state.preset = p.id;
      state.text = p.text;
      state.gen.prompt = p.prompt || state.gen.prompt;
      $("prompt").value = state.gen.prompt;
      rebuild();
    });
    box.append(b);
  }

  const chats = $("chats");
  if (chats) {
    chats.replaceChildren();
    for (const conv of state.conversations || []) {
      const wrap = document.createElement("div");
      wrap.className = "chat";
      for (const m of conv.messages) {
        const row = document.createElement("div");
        row.className = "chat-msg role-" + m.role;
        const k = document.createElement("span");
        k.className = "chat-role";
        k.textContent = m.role;
        const body = document.createElement("span");
        body.className = "chat-body";
        body.textContent = m.content;
        row.append(k, body);
        wrap.append(row);
      }
      chats.append(wrap);
    }
  }
  const chatMeta = $("chat-meta");
  if (chatMeta) chatMeta.textContent = String((state.conversations || []).length);

  const packed = $("packed");
  if (packed && state.data) {
    packed.replaceChildren();
    const n = Math.min(state.data.ids.length, 64);
    for (let i = 0; i < n; i++) {
      const role = state.data.roles[i];
      packed.append(tokenEl(state.data.ids[i], role ? "role-" + role : ""));
    }
  }
  const packMeta = $("pack-meta");
  if (packMeta && state.data) {
    const sup = Array.from(state.data.mask).filter((x) => x).length;
    packMeta.textContent = state.data.ids.length + " · assistant " + sup;
  }

  const counts = new Map();
  if (state.data) {
    for (const id of state.data.ids) counts.set(id, (counts.get(id) || 0) + 1);
  }
  const vocab = $("vocab");
  vocab.replaceChildren();
  state.tokenizer.chars.forEach((ch, id) => {
    const chip = document.createElement("div");
    chip.className = "vchip" + (state.tokenizer.isSpecial(id) ? " special" : "");
    const a = document.createElement("span");
    a.textContent = state.tokenizer.display(ch);
    const b = document.createElement("span");
    b.textContent = String(id);
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = String(counts.get(id) || 0);
    chip.append(a, b, n);
    vocab.append(chip);
  });
  $("vocab-meta").textContent = String(state.tokenizer.vocabSize);
}

function renderArch() {
  const { nLayer, nEmbd, nHead, blockSize } = state.hparams;
  const V = state.tokenizer.vocabSize;
  const C = nEmbd;
  const Tshow = Math.min(8, blockSize);
  let labels = [];
  if (state.lastBatch) {
    labels = Array.from(state.lastBatch.x.subarray(0, Tshow), (id) => state.tokenizer.displayId(id));
  } else {
    labels = state.tokenizer.chars.slice(0, Tshow).map((ch) => state.tokenizer.display(ch));
  }
  const attns = [];
  if (state.model && state.model.didForward) {
    try {
      ensureForward("train");
      for (let i = 0; i < nLayer; i++) attns.push(state.model.getAttention(i, "mean", 0));
    } catch (err) {}
  }
  let probs = null;
  if (state.lastBatch) {
    const { logits, T } = state.lastBatch;
    const row = T - 1;
    probs = Array.from(softmax(logits.subarray(row * V, (row + 1) * V)));
  }
  drawArchDiagram($("arch-flow"), {
    nLayer,
    nEmbd,
    nHead,
    blockSize,
    vocabSize: V,
    labels,
    attns,
    probs,
    selected: state.archPart,
    onSelect: (id) => {
      state.archPart = id;
      renderArch();
    },
  });

  const rows = [];
  if (state.archPart === "embed") {
    rows.push(["tok emb", `${V}×${C}`, V * C]);
    rows.push(["pos emb", `${blockSize}×${C}`, blockSize * C]);
  } else if (state.archPart === "head") {
    rows.push(["ln f", `${C}`, C]);
    rows.push(["unembed", `${C}×${V}`, 0]);
  } else {
    const i = Number(String(state.archPart).split("-")[1] || 0);
    rows.push([`L${i + 1} RMS 1`, `${C}`, C]);
    rows.push(["Wqkv", `${C}×${3 * C}`, C * 3 * C]);
    rows.push(["Wo", `${C}×${C}`, C * C]);
    rows.push([`L${i + 1} RMS 2`, `${C}`, C]);
    rows.push(["W1", `${C}×${4 * C}`, C * 4 * C]);
    rows.push(["W2", `${4 * C}×${C}`, 4 * C * C]);
  }
  const tb = $("arch-inspect").querySelector("tbody");
  tb.replaceChildren();
  for (const [name, shape, n] of rows) {
    const tr = document.createElement("tr");
    const c1 = document.createElement("td");
    c1.textContent = name;
    const c2 = document.createElement("td");
    c2.textContent = shape;
    const c3 = document.createElement("td");
    c3.textContent = n ? n.toLocaleString("zh-CN") : "tied";
    tr.append(c1, c2, c3);
    tb.append(tr);
  }

  const setSeg = (el, options, cur, key) => {
    el.replaceChildren();
    for (const val of options) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = String(val);
      b.setAttribute("aria-pressed", String(cur === val));
      b.addEventListener("click", () => {
        state.hparams[key] = val;
        if (key === "nEmbd") {
          const hs = headsFor(val);
          if (!hs.includes(state.hparams.nHead)) state.hparams.nHead = hs.at(-1);
        }
        rebuild();
      });
      el.append(b);
    }
  };
  setSeg($("hp-layer"), [1, 2, 3], nLayer, "nLayer");
  setSeg($("hp-embd"), [16, 32, 64], nEmbd, "nEmbd");
  setSeg($("hp-head"), headsFor(nEmbd), nHead, "nHead");
  setSeg($("hp-ctx"), [8, 16, 24, 32], blockSize, "blockSize");
}

function contextIds() {
  const T = state.hparams.blockSize;
  const ids = state.gen.ids.length ? state.gen.ids : [0];
  return Int32Array.from(ids.slice(-T));
}

function resetGen(render = true) {
  const packed = chatPrefix(state.conversations, state.tokenizer, state.gen.prompt);
  state.gen.ids = Array.from(packed.ids);
  state.gen.origin = state.gen.ids.map(() => "prompt");
  state.gen.roles = Array.from(packed.roles);
  state.gen.last = null;
  state.gen.ctx = null;
  if (render) renderGen();
}

function renderGen() {
  $("prompt").value = state.gen.prompt;
  const stream = $("gen-stream");
  stream.replaceChildren();
  state.gen.ids.forEach((id, i) => {
    const born = state.gen.origin[i] === "prompt" ? "prompt" : "";
    const now = i === state.gen.ids.length - 1 && state.gen.origin[i] === "gen" ? "now" : "";
    const role = state.gen.roles && state.gen.roles[i] ? "role-" + state.gen.roles[i] : "";
    stream.append(tokenEl(id, `${born} ${now} ${role}`.trim()));
  });

  const last = state.gen.last;
  if (!last) {
    $("probs").replaceChildren();
    return;
  }
  const items = [];
  for (let id = 0; id < state.tokenizer.vocabSize; id++) {
    items.push({ id, label: state.tokenizer.displayId(id), p: last.filtered[id] });
  }
  items.sort((a, b) => b.p - a.p);
  const top = items.slice(0, 12);
  if (!top.some((it) => it.id === last.id)) {
    const hit = items.find((it) => it.id === last.id);
    if (hit) top.push(hit);
  }
  renderProbBars(
    $("probs"),
    top.filter((it) => it.p > 0 || it.id === last.id),
    last.id,
  );
  if (state.gen.ctx) {
    attnGen.set(labelsFor(state.gen.ctx), state.model.getAttention(state.vizLayer, state.vizHead, 0));
  }
}

function renderAll() {
  renderMeters();
  renderHeadSelectors();
  renderData();
  renderArch();
  renderTrainBatch();
  renderEmbed();
  renderAttention();
  renderGen();
  $("lr").value = String(sliderFromLr(state.hparams.lr));
  $("lr-val").textContent = fmtLr(state.hparams.lr);
  $("batch").value = String(state.hparams.batchSize);
  $("bs-val").textContent = String(state.hparams.batchSize);
  $("speed").value = String(state.speed);
  $("spd-val").textContent = String(state.speed);
  $("temp").value = String(Math.round(state.gen.temp * 100));
  $("temp-val").textContent = state.gen.temp.toFixed(2);
  $("topk").value = String(state.gen.topk);
  $("topk-val").textContent = state.gen.topk === 0 ? "关" : String(state.gen.topk);
  $("topp").value = String(Math.round(state.gen.topp * 100));
  $("topp-val").textContent = state.gen.topp.toFixed(2);
  for (const b of $("optim").querySelectorAll("button")) {
    b.setAttribute("aria-pressed", String(b.dataset.opt === state.hparams.optim));
  }
  icons();
}

function setView(view) {
  state.view = view;
  for (const v of ["data", "arch", "train", "gen"]) {
    $(`view-${v}`).hidden = v !== view;
    $(`tab-${v}`).setAttribute("aria-selected", String(v === view));
  }
  if (view === "train") ensureForward("train");
  if (view === "arch") renderArch();
  if (view === "gen" && state.gen.ctx) ensureForward("gen");
  requestAnimationFrame(() => {
    lossChart.draw();
    renderAttention();
    atlas.draw();
  });
}

function trainOnce() {
  const B = state.hparams.batchSize;
  const T = state.hparams.blockSize;
  const batch = makeBatch(state.data, B, T, state.trainRng);
  const out = state.model.trainStep(batch.x, batch.y, B, T, state.hparams.lr, state.hparams.optim, batch.mask);
  state.forwardKind = "train";
  state.step += 1;
  state.lastLoss = out.loss;
  state.lastAcc = batchAccuracy(out.logits, batch.y, B, T, state.tokenizer.vocabSize, batch.mask);
  state.lastBatch = snapshotBatch(batch.x, batch.y, out.logits, B, T, batch);
  lossChart.push(out.loss);
  return out;
}

function refreshTrainViz() {
  lossChart.draw();
  renderMeters();
  renderTrainBatch();
  renderEmbed();
  renderAttention();
}

function syncPlayButtons() {
  const play = $("btn-play");
  if (play) {
    const span = play.querySelector("span");
    if (span) span.textContent = state.playing ? "暂停" : "训练";
    setIcon(play, state.playing ? "pause" : "play");
  }
  const run = $("btn-gen-run");
  if (run) {
    const span = run.querySelector("span");
    if (span) span.textContent = state.gen.running ? "停止" : "连续";
    setIcon(run, state.gen.running ? "square" : "play");
  }
  icons();
}

async function playLoop() {
  try {
    while (state.playing) {
      const n = state.speed;
      for (let i = 0; i < n; i++) trainOnce();
      refreshTrainViz();
      announce("步 " + state.step + " 损失 " + fmtNum(state.lastLoss, 3));
      await new Promise((r) => requestAnimationFrame(r));
    }
  } catch (err) {
    console.error(err);
    state.playing = false;
    syncPlayButtons();
    showError(err);
  }
}

function generateOne() {
  if (!state.gen.ids.length) resetGen(false);
  const ctx = contextIds();
  const T = ctx.length;
  const V = state.tokenizer.vocabSize;
  state.model.forward(ctx, null, 1, T);
  state.forwardKind = "gen";
  state.gen.ctx = Int32Array.from(ctx);
  const logits = state.model.logits.subarray((T - 1) * V, T * V);
  const pipe = samplePipeline(logits, {
    temperature: state.gen.temp,
    topk: state.gen.topk,
    topp: state.gen.topp,
    rng: state.sampleRng,
  });
  state.gen.ids.push(pipe.id);
  state.gen.origin.push("gen");
  if (!state.gen.roles) state.gen.roles = [];
  state.gen.roles.push("assistant");
  state.gen.last = pipe;
  if (state.tokenizer.isEnd(pipe.id)) state.gen.running = false;
  renderGen();
  renderMeters();
  announce(state.tokenizer.decode([pipe.id]));
  return pipe;
}

async function generateRun() {
  let n = 0;
  while (state.gen.running && n < 80) {
    generateOne();
    n += 1;
    await new Promise((r) => setTimeout(r, 70));
  }
  state.gen.running = false;
  syncPlayButtons();
}

function bind() {
  for (const tab of document.querySelectorAll(".tabs [data-view]")) {
    tab.addEventListener("click", () => setView(tab.dataset.view));
  }

  $("btn-play").addEventListener("click", () => {
    try {
      state.playing = !state.playing;
      syncPlayButtons();
      if (state.playing) playLoop();
    } catch (err) {
      console.error(err);
      showError(err);
    }
  });
  $("btn-step").addEventListener("click", () => {
    try {
      state.playing = false;
      syncPlayButtons();
      trainOnce();
      refreshTrainViz();
    } catch (err) {
      console.error(err);
      showError(err);
    }
  });
  $("btn-burst").addEventListener("click", async () => {
    state.playing = false;
    syncPlayButtons();
    for (let i = 0; i < 50; i++) {
      trainOnce();
      if (i % 10 === 9) {
        refreshTrainViz();
        await new Promise((r) => requestAnimationFrame(r));
      }
    }
    refreshTrainViz();
  });
  $("btn-reset").addEventListener("click", () => rebuild());

  $("lr").addEventListener("input", (e) => {
    state.hparams.lr = lrFromSlider(e.target.value);
    $("lr-val").textContent = fmtLr(state.hparams.lr);
    persist();
  });
  $("batch").addEventListener("input", (e) => {
    state.hparams.batchSize = Number(e.target.value);
    $("bs-val").textContent = String(state.hparams.batchSize);
    persist();
  });
  $("speed").addEventListener("input", (e) => {
    state.speed = Number(e.target.value);
    $("spd-val").textContent = String(state.speed);
  });
  $("optim").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    state.hparams.optim = b.dataset.opt;
    for (const x of $("optim").querySelectorAll("button")) {
      x.setAttribute("aria-pressed", String(x === b));
    }
    persist();
  });

  let corpusTimer = 0;
  $("corpus").addEventListener("input", (e) => {
    state.text = e.target.value;
    state.preset = "custom";
    clearTimeout(corpusTimer);
    const parsed = parseCorpus(state.text);
    const st = $("corpus-status");
    if (st) {
      st.textContent = parsed.error || "";
      st.classList.toggle("is-bad", !!parsed.error);
    }
    if (parsed.error) return;
    corpusTimer = setTimeout(() => rebuild(), 400);
  });

  $("prompt").addEventListener("change", (e) => {
    state.gen.prompt = e.target.value;
    persist();
    resetGen(true);
  });
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      state.gen.prompt = e.target.value;
      persist();
      resetGen(true);
    }
  });

  $("btn-gen-step").addEventListener("click", () => {
    try {
      state.gen.running = false;
      syncPlayButtons();
      generateOne();
    } catch (err) {
      console.error(err);
      showError(err);
    }
  });
  $("btn-gen-run").addEventListener("click", () => {
    state.gen.running = !state.gen.running;
    syncPlayButtons();
    if (state.gen.running) generateRun();
  });
  $("btn-gen-clear").addEventListener("click", () => {
    state.gen.running = false;
    syncPlayButtons();
    resetGen(true);
  });

  $("temp").addEventListener("input", (e) => {
    state.gen.temp = Number(e.target.value) / 100;
    $("temp-val").textContent = state.gen.temp.toFixed(2);
    persist();
  });
  $("topk").addEventListener("input", (e) => {
    state.gen.topk = Number(e.target.value);
    $("topk-val").textContent = state.gen.topk === 0 ? "关" : String(state.gen.topk);
    persist();
  });
  $("topp").addEventListener("input", (e) => {
    state.gen.topp = Number(e.target.value) / 100;
    $("topp-val").textContent = state.gen.topp.toFixed(2);
    persist();
  });

  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea")) return;
    if (e.code === "Space" && state.view === "train") {
      e.preventDefault();
      $("btn-play").click();
    }
  });
}

bind();
try {
  rebuild();
  setView("data");
} catch (err) {
  console.error(err);
  showError(err);
}
