function pretty(records) {
  return JSON.stringify(records, null, 2);
}

export const PRESETS = [
  {
    id: "em",
    name: "电磁",
    prompt: "BPSK是什么调制？",
    text: pretty([
      {
        messages: [
          { role: "system", content: "你是电磁信号分析助手。" },
          { role: "user", content: "BPSK是什么调制？" },
          { role: "assistant", content: "二进制相移键控，两相位相差180度。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是电磁信号分析助手。" },
          { role: "user", content: "QPSK和BPSK有何不同？" },
          { role: "assistant", content: "QPSK四相位，同带宽下比特率更高。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是电磁信号分析助手。" },
          { role: "user", content: "LFM常见于什么体制？" },
          { role: "assistant", content: "线性调频，多用于脉冲压缩雷达。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是电磁信号分析助手。" },
          { role: "user", content: "跳频体制有何特点？" },
          { role: "assistant", content: "载频按图案跳变，抗干扰、低截获。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是电磁信号分析助手。" },
          { role: "user", content: "个体识别靠什么特征？" },
          { role: "assistant", content: "功放非线性、杂散和相位噪声等无意指纹。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是电磁信号分析助手。" },
          { role: "user", content: "同体制电台如何区分？" },
          { role: "assistant", content: "硬件公差形成稳定射频指纹，可分到个体。" },
        ],
      },
    ]),
  },
  {
    id: "mod",
    name: "调制",
    prompt: "16QAM怎么调？",
    text: pretty([
      {
        messages: [
          { role: "system", content: "你是调制识别助手。" },
          { role: "user", content: "AM和FM差在哪？" },
          { role: "assistant", content: "AM变幅度，FM变频率，信息载荷不同。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是调制识别助手。" },
          { role: "user", content: "BPSK星座是什么样？" },
          { role: "assistant", content: "两实点，位于+1与-1，相位差180度。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是调制识别助手。" },
          { role: "user", content: "16QAM怎么调？" },
          { role: "assistant", content: "幅度与相位联合编码，每符号4比特。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是调制识别助手。" },
          { role: "user", content: "OFDM为何抗多径？" },
          { role: "assistant", content: "子载波正交，符号更长，频选衰落被切开。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是调制识别助手。" },
          { role: "user", content: "MSK属于哪类？" },
          { role: "assistant", content: "连续相位频移，包络恒定，旁瓣较低。" },
        ],
      },
    ]),
  },
  {
    id: "regime",
    name: "体制",
    prompt: "FMCW用于什么？",
    text: pretty([
      {
        messages: [
          { role: "system", content: "你是信号体制分析助手。" },
          { role: "user", content: "脉压体制解决什么？" },
          { role: "assistant", content: "宽脉冲保能量，匹配滤波提分辨力。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是信号体制分析助手。" },
          { role: "user", content: "FMCW用于什么？" },
          { role: "assistant", content: "连续波调频测距测速，多见近程雷达。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是信号体制分析助手。" },
          { role: "user", content: "跳频和直扩有何不同？" },
          { role: "assistant", content: "跳频换载频，直扩展频谱，抗干扰机理不同。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是信号体制分析助手。" },
          { role: "user", content: "脉冲多普勒体制看什么？" },
          { role: "assistant", content: "脉间相参积累，用多普勒分动目标。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是信号体制分析助手。" },
          { role: "user", content: "相控阵体制优势？" },
          { role: "assistant", content: "波束电扫，可快速跳变指向与多波束。" },
        ],
      },
    ]),
  },
  {
    id: "sei",
    name: "个体",
    prompt: "个体识别靠什么？",
    text: pretty([
      {
        messages: [
          { role: "system", content: "你是辐射源个体识别助手。" },
          { role: "user", content: "个体识别靠什么？" },
          { role: "assistant", content: "发射机无意特征，如功放非线性与杂散。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是辐射源个体识别助手。" },
          { role: "user", content: "为何能区分同型号电台？" },
          { role: "assistant", content: "器件公差造成射频指纹，且短时稳定。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是辐射源个体识别助手。" },
          { role: "user", content: "脉内无意调制是什么？" },
          { role: "assistant", content: "非理想发射引起的脉内细微畸变。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是辐射源个体识别助手。" },
          { role: "user", content: "相位噪声对个体有何用？" },
          { role: "assistant", content: "振荡器差异会进射频，可作指纹维。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是辐射源个体识别助手。" },
          { role: "user", content: "杂散从哪来？" },
          { role: "assistant", content: "本振泄漏、谐波和交调，个体间有差异。" },
        ],
      },
    ]),
  },
];

function normalizeRole(role) {
  const r = String(role || "").toLowerCase();
  if (r === "human" || r === "prompt") return "user";
  if (r === "gpt" || r === "bot" || r === "model") return "assistant";
  if (r === "system" || r === "user" || r === "assistant") return r;
  return null;
}

function normalizeMessage(m) {
  if (!m || typeof m !== "object") return null;
  const role = normalizeRole(m.role || m.from);
  if (!role) return null;
  const content = String(m.content ?? m.value ?? "");
  return { role, content };
}

function normalizeRecord(rec) {
  if (!rec || typeof rec !== "object") return null;
  if (Array.isArray(rec.messages)) {
    const messages = rec.messages.map(normalizeMessage).filter(Boolean);
    return messages.length ? { messages } : null;
  }
  if (Array.isArray(rec.conversations)) {
    const messages = rec.conversations.map(normalizeMessage).filter(Boolean);
    return messages.length ? { messages } : null;
  }
  const messages = [];
  for (const role of ["system", "user", "assistant"]) {
    if (rec[role] != null && rec[role] !== "") {
      messages.push({ role, content: String(rec[role]) });
    }
  }
  return messages.length ? { messages } : null;
}

export function parseCorpus(raw) {
  try {
    let data = JSON.parse(raw);
    if (data && !Array.isArray(data)) data = [data];
    if (!Array.isArray(data)) return { error: "需要 JSON 数组", conversations: [] };
    const conversations = data.map(normalizeRecord).filter(Boolean);
    if (!conversations.length) return { error: "没有对话", conversations: [] };
    return { conversations, error: null };
  } catch {
    return { error: "JSON 无法解析", conversations: [] };
  }
}

export function flattenContents(conversations) {
  let s = "";
  for (const c of conversations || []) {
    for (const m of c.messages) s += m.content;
  }
  return s;
}

export function packMessages(messages, tokenizer) {
  const ids = [];
  const mask = [];
  const roles = [];
  for (const m of messages) {
    const tag =
      m.role === "system"
        ? tokenizer.systemId
        : m.role === "user"
          ? tokenizer.userId
          : tokenizer.assistantId;
    const supervised = m.role === "assistant" ? 1 : 0;
    ids.push(tag);
    mask.push(supervised);
    roles.push(m.role);
    const { ids: content } = tokenizer.encode(m.content);
    for (const id of content) {
      ids.push(id);
      mask.push(supervised);
      roles.push(m.role);
    }
    ids.push(tokenizer.endId);
    mask.push(supervised);
    roles.push(m.role);
  }
  return { ids, mask, roles };
}

export function packCorpus(conversations, tokenizer) {
  const ids = [];
  const mask = [];
  const roles = [];
  for (const c of conversations || []) {
    const p = packMessages(c.messages, tokenizer);
    ids.push(...p.ids);
    mask.push(...p.mask);
    roles.push(...p.roles);
  }
  if (!ids.length) {
    return {
      ids: new Int32Array([tokenizer.endId, tokenizer.endId, tokenizer.endId, tokenizer.endId]),
      mask: new Uint8Array([0, 0, 0, 0]),
      roles: ["user", "user", "user", "user"],
    };
  }
  return { ids: Int32Array.from(ids), mask: Uint8Array.from(mask), roles };
}

export function loopStream(stream, minLen) {
  const ids = Array.from(stream.ids);
  const mask = Array.from(stream.mask);
  const roles = Array.from(stream.roles);
  if (!ids.length) {
    return {
      ids: new Int32Array(8),
      mask: new Uint8Array(8),
      roles: Array(8).fill("user"),
    };
  }
  while (ids.length < minLen) {
    ids.push(...stream.ids);
    mask.push(...stream.mask);
    roles.push(...stream.roles);
  }
  return { ids: Int32Array.from(ids), mask: Uint8Array.from(mask), roles };
}

export function makeBatch(stream, B, T, rng) {
  const { ids, mask, roles } = stream;
  const x = new Int32Array(B * T);
  const y = new Int32Array(B * T);
  const ymask = new Uint8Array(B * T);
  const rolesX = [];
  const rolesY = [];
  const n = ids.length;
  const span = n - T - 1;
  const limit = Math.max(1, span);
  for (let b = 0; b < B; b++) {
    const i = span <= 0 ? 0 : rng.int(limit);
    for (let t = 0; t < T; t++) {
      const xi = (i + t) % n;
      const yi = (i + t + 1) % n;
      x[b * T + t] = ids[xi];
      y[b * T + t] = ids[yi];
      ymask[b * T + t] = mask[yi];
      rolesX[b * T + t] = roles[xi];
      rolesY[b * T + t] = roles[yi];
    }
  }
  return { x, y, mask: ymask, rolesX, rolesY };
}

export function chatPrefix(conversations, tokenizer, userText) {
  const first = conversations && conversations[0];
  const sys = first?.messages.find((m) => m.role === "system");
  const messages = [];
  if (sys) messages.push(sys);
  messages.push({ role: "user", content: String(userText ?? "") });
  const packed = packMessages(messages, tokenizer);
  packed.ids.push(tokenizer.assistantId);
  packed.mask.push(1);
  packed.roles.push("assistant");
  return packed;
}
