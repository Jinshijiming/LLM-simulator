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
    onForest: v("--on-forest", "#F3FFF6"),
    sans: v("--sans", "sans-serif"),
    mono: v("--mono", "monospace"),
  };
}

function el(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [k, val] of Object.entries(attrs)) {
    if (val == null) continue;
    if (k === "text") node.textContent = val;
    else node.setAttribute(k, String(val));
  }
  return node;
}

function hit(g, selected, onClick) {
  g.style.cursor = "pointer";
  g.addEventListener("click", onClick);
  if (selected) g.classList.add("is-on");
  return g;
}

function downsampleAttn(weights, T, show) {
  if (!weights || !T) return null;
  const n = Math.min(show, T);
  const out = new Float32Array(n * n);
  const off = Math.max(0, T - n);
  for (let q = 0; q < n; q++) {
    for (let k = 0; k < n; k++) out[q * n + k] = weights[(off + q) * T + (off + k)];
  }
  return { w: out, n };
}

export function drawArchDiagram(container, opts) {
  const {
    nLayer,
    nEmbd,
    nHead,
    blockSize,
    vocabSize,
    labels = [],
    attns = [],
    selected,
    onSelect,
  } = opts;
  const t = theme();
  const W = 540;
  const tokH = 58;
  const embedH = 108;
  const gap = 22;
  const blockH = 168;
  const headH = 118;
  const H = 16 + tokH + embedH + gap + nLayer * (blockH + gap) + headH + 8;
  const svg = el("svg", {
    viewBox: `0 0 ${W} ${H}`,
    width: "100%",
    role: "img",
    "aria-label": "Transformer 结构示意图",
  });

  const cables = Math.round(5 + (nEmbd / 64) * 7);
  const railX = 42;
  const railTop = 16 + tokH + 8;
  const railBot = H - 28;
  for (let i = 0; i < cables; i++) {
    const x = railX + i * 2.2;
    const line = el("path", {
      d: `M ${x} ${railTop} L ${x} ${railBot}`,
      stroke: t.teal,
      "stroke-width": "1.1",
      opacity: String(0.18 + (i / cables) * 0.35),
      fill: "none",
    });
    svg.append(line);
  }

  let y = 12;
  const showT = Math.min(8, blockSize, Math.max(1, labels.length || blockSize));
  const shown = labels.length ? labels.slice(0, showT) : Array.from({ length: showT }, () => "·");
  const tileW = 36;
  const tileGap = 6;
  const rowW = shown.length * (tileW + tileGap) - tileGap;
  const rowX = 118;

  const tokG = hit(el("g"), selected === "embed", () => onSelect("embed"));
  tokG.append(
    el("text", {
      x: 118,
      y: y + 2,
      fill: t.muted,
      "font-size": "11",
      "font-family": t.sans,
      text: "词元",
    }),
  );
  shown.forEach((lab, i) => {
    const x = rowX + i * (tileW + tileGap);
    tokG.append(
      el("rect", {
        x,
        y: y + 8,
        width: tileW,
        height: 40,
        rx: "6",
        fill: t.paper,
        stroke: t.line,
      }),
    );
    tokG.append(
      el("text", {
        x: x + tileW / 2,
        y: y + 33,
        "text-anchor": "middle",
        fill: t.ink,
        "font-size": "16",
        "font-family": t.mono,
        text: lab,
      }),
    );
    tokG.append(
      el("path", {
        d: `M ${x + tileW / 2} ${y + 48} L ${x + tileW / 2} ${y + tokH - 2}`,
        stroke: t.line,
        fill: "none",
      }),
    );
  });
  svg.append(tokG);
  y += tokH;

  const embedG = hit(el("g"), selected === "embed", () => onSelect("embed"));
  if (selected === "embed") {
    embedG.append(
      el("rect", {
        x: 104,
        y: y - 2,
        width: 420,
        height: embedH,
        rx: "8",
        fill: t.paper,
        stroke: t.forest,
        "stroke-width": "2",
      }),
    );
  } else {
    embedG.append(
      el("rect", {
        x: 104,
        y: y - 2,
        width: 420,
        height: embedH,
        rx: "8",
        fill: t.paper,
        stroke: t.line,
      }),
    );
  }
  embedG.append(
    el("text", {
      x: 118,
      y: y + 16,
      fill: t.ink,
      "font-size": "13",
      "font-family": t.sans,
      "font-weight": "500",
      text: "词嵌入 + 位置",
    }),
  );
  embedG.append(
    el("text", {
      x: 118,
      y: y + 32,
      fill: t.muted,
      "font-size": "11",
      "font-family": t.mono,
      text: `${vocabSize} × ${nEmbd}   +   ${blockSize} × ${nEmbd}`,
    }),
  );

  const cells = Math.min(18, vocabSize);
  for (let i = 0; i < cells; i++) {
    const cx = 122 + (i % 9) * 11;
    const cy = y + 46 + Math.floor(i / 9) * 11;
    embedG.append(
      el("rect", {
        x: cx,
        y: cy,
        width: 9,
        height: 9,
        rx: "1.5",
        fill: t.teal,
        opacity: String(0.25 + (i % 5) * 0.12),
      }),
    );
  }
  embedG.append(
    el("text", {
      x: 122,
      y: y + 96,
      fill: t.muted,
      "font-size": "11",
      "font-family": t.sans,
      text: "词表查找",
    }),
  );

  const waveX = 250;
  const waveY = y + 58;
  let d = `M ${waveX} ${waveY}`;
  for (let i = 0; i <= 80; i++) {
    const px = waveX + i * 1.4;
    const py = waveY + Math.sin(i / 5.5) * 10;
    d += ` L ${px} ${py}`;
  }
  embedG.append(el("path", { d, fill: "none", stroke: t.gold, "stroke-width": "2" }));
  embedG.append(
    el("text", {
      x: 250,
      y: y + 96,
      fill: t.muted,
      "font-size": "11",
      "font-family": t.sans,
      text: "位置编码",
    }),
  );

  const barX = 390;
  for (let i = 0; i < 10; i++) {
    const h = 10 + ((i * 17) % 28);
    embedG.append(
      el("rect", {
        x: barX + i * 9,
        y: y + 78 - h,
        width: 7,
        height: h,
        rx: "1.5",
        fill: t.forest,
        opacity: "0.85",
      }),
    );
  }
  embedG.append(
    el("text", {
      x: barX,
      y: y + 96,
      fill: t.muted,
      "font-size": "11",
      "font-family": t.sans,
      text: `向量 ${nEmbd}d`,
    }),
  );
  svg.append(embedG);
  y += embedH + gap;

  for (let layer = 0; layer < nLayer; layer++) {
    const id = `block-${layer}`;
    const g = hit(el("g"), selected === id, () => onSelect(id));
    const x0 = 104;
    const y0 = y;
    const on = selected === id;
    g.append(
      el("rect", {
        x: x0,
        y: y0,
        width: 420,
        height: blockH,
        rx: "8",
        fill: t.paper,
        stroke: on ? t.forest : t.line,
        "stroke-width": on ? "2" : "1",
      }),
    );

    const skip = el("path", {
      d: `M ${railX + 6} ${y0 - 4} C ${18} ${y0 + 20}, ${18} ${y0 + blockH - 20}, ${railX + 6} ${y0 + blockH + 4}`,
      fill: "none",
      stroke: t.gold,
      "stroke-width": "2",
      opacity: "0.85",
    });
    svg.append(skip);

    g.append(
      el("text", {
        x: x0 + 14,
        y: y0 + 18,
        fill: t.ink,
        "font-size": "13",
        "font-family": t.sans,
        "font-weight": "500",
        text: `块 ${layer + 1}`,
      }),
    );
    g.append(
      el("text", {
        x: x0 + 70,
        y: y0 + 18,
        fill: t.muted,
        "font-size": "11",
        "font-family": t.mono,
        text: `RMSNorm · ${nHead} 头 · SiLU FFN`,
      }),
    );

    const qkv = [
      { lab: "Q", col: t.forest },
      { lab: "K", col: t.teal },
      { lab: "V", col: t.vermilion },
    ];
    qkv.forEach((item, qi) => {
      const bx = x0 + 16 + qi * 54;
      const by = y0 + 32;
      g.append(
        el("rect", {
          x: bx,
          y: by,
          width: 46,
          height: 52,
          rx: "4",
          fill: t.bg,
          stroke: item.col,
        }),
      );
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 3; c++) {
          g.append(
            el("rect", {
              x: bx + 6 + c * 12,
              y: by + 8 + r * 9,
              width: 10,
              height: 7,
              rx: "1",
              fill: item.col,
              opacity: String(0.25 + ((r + c + qi) % 4) * 0.18),
            }),
          );
        }
      }
      g.append(
        el("text", {
          x: bx + 23,
          y: by + 64,
          "text-anchor": "middle",
          fill: t.muted,
          "font-size": "11",
          "font-family": t.mono,
          text: item.lab,
        }),
      );
    });

    const attnBox = { x: x0 + 186, y: y0 + 32, s: 72 };
    for (let h = nHead - 1; h >= 0; h--) {
      g.append(
        el("rect", {
          x: attnBox.x + h * 3,
          y: attnBox.y - h * 2,
          width: attnBox.s,
          height: attnBox.s,
          rx: "3",
          fill: t.bg,
          stroke: t.line,
          opacity: h === 0 ? "1" : "0.45",
        }),
      );
    }
    const packed = downsampleAttn(attns[layer], blockSize, 8);
    const n = packed ? packed.n : 8;
    const cell = 72 / n;
    for (let q = 0; q < n; q++) {
      for (let k = 0; k < n; k++) {
        let a = k > q ? 0.06 : 0.12 + 0.7 * ((q + 1 - k) / n);
        if (packed) {
          const p = packed.w[q * n + k];
          a = k > q ? 0.05 : 0.1 + 0.85 * Math.min(1, p * n);
        }
        g.append(
          el("rect", {
            x: attnBox.x + k * cell,
            y: attnBox.y + q * cell,
            width: Math.max(1, cell - 0.6),
            height: Math.max(1, cell - 0.6),
            fill: t.vermilion,
            opacity: String(a),
          }),
        );
      }
    }
    g.append(
      el("text", {
        x: attnBox.x,
        y: attnBox.y + 86,
        fill: t.muted,
        "font-size": "11",
        "font-family": t.sans,
        text: `因果注意 ×${nHead}`,
      }),
    );

    const ffnX = x0 + 292;
    const ffnY = y0 + 40;
    const widths = [36, 78, 36];
    const heights = [18, 34, 18];
    let fy = ffnY;
    widths.forEach((w, i) => {
      g.append(
        el("rect", {
          x: ffnX + (78 - w) / 2,
          y: fy,
          width: w,
          height: heights[i],
          rx: "3",
          fill: t.forest,
          opacity: String(0.35 + i * 0.2),
        }),
      );
      fy += heights[i] + 4;
    });
    g.append(
      el("text", {
        x: ffnX,
        y: y0 + 118,
        fill: t.muted,
        "font-size": "11",
        "font-family": t.mono,
        text: `${nEmbd}→${nEmbd * 4}→${nEmbd}`,
      }),
    );
    g.append(
      el("text", {
        x: x0 + 14,
        y: y0 + 146,
        fill: t.muted,
        "font-size": "11",
        "font-family": t.sans,
        text: "残差 +",
      }),
    );
    svg.append(g);
    y += blockH + gap;
  }

  const headG = hit(el("g"), selected === "head", () => onSelect("head"));
  const on = selected === "head";
  headG.append(
    el("rect", {
      x: 104,
      y,
      width: 420,
      height: headH,
      rx: "8",
      fill: t.paper,
      stroke: on ? t.forest : t.line,
      "stroke-width": on ? "2" : "1",
    }),
  );
  headG.append(
    el("text", {
      x: 118,
      y: y + 20,
      fill: t.ink,
      "font-size": "13",
      "font-family": t.sans,
      "font-weight": "500",
      text: "解嵌入 → 下一个词",
    }),
  );
  headG.append(
    el("text", {
      x: 118,
      y: y + 36,
      fill: t.muted,
      "font-size": "11",
      "font-family": t.mono,
      text: `${nEmbd} → ${vocabSize}  tied`,
    }),
  );

  const bars = Math.min(14, vocabSize);
  let maxH = 1;
  const fake = Array.from({ length: bars }, (_, i) => 0.15 + ((i * 13) % 10) / 18);
  if (opts.probs && opts.probs.length) {
    for (let i = 0; i < bars; i++) fake[i] = opts.probs[i] || 0;
  }
  maxH = Math.max(...fake, 0.05);
  fake.forEach((p, i) => {
    const bh = 8 + (p / maxH) * 46;
    headG.append(
      el("rect", {
        x: 118 + i * 16,
        y: y + 96 - bh,
        width: 12,
        height: bh,
        rx: "2",
        fill: i === 0 ? t.vermilion : t.teal,
        opacity: i === 0 ? "1" : "0.75",
      }),
    );
  });
  headG.append(
    el("text", {
      x: 118,
      y: y + 110,
      fill: t.muted,
      "font-size": "11",
      "font-family": t.sans,
      text: "Softmax 词表分布",
    }),
  );
  svg.append(headG);

  container.replaceChildren(svg);
}
