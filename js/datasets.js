function pretty(records) {
  return JSON.stringify(records, null, 2);
}

export const PRESETS = [
  {
    id: "qa",
    name: "问答",
    prompt: "你叫什么？",
    text: pretty([
      {
        messages: [
          { role: "system", content: "你是一只小猫。" },
          { role: "user", content: "你叫什么？" },
          { role: "assistant", content: "小猫小狗" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是一只小猫。" },
          { role: "user", content: "你会什么？" },
          { role: "assistant", content: "小猫小狗" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是一只小猫。" },
          { role: "user", content: "再叫一次" },
          { role: "assistant", content: "小猫小狗" },
        ],
      },
    ]),
  },
  {
    id: "poem",
    name: "静夜思",
    prompt: "静夜思",
    text: pretty([
      {
        messages: [
          { role: "system", content: "你是诗人。" },
          { role: "user", content: "静夜思" },
          { role: "assistant", content: "床前明月光，疑是地上霜。" },
        ],
      },
      {
        messages: [
          { role: "system", content: "你是诗人。" },
          { role: "user", content: "下两句" },
          { role: "assistant", content: "举头望明月，低头思故乡。" },
        ],
      },
    ]),
  },
  {
    id: "en",
    name: "English",
    prompt: "who are you?",
    text: pretty([
      {
        messages: [
          { role: "system", content: "You are a cat." },
          { role: "user", content: "who are you?" },
          { role: "assistant", content: "the cat" },
        ],
      },
      {
        messages: [
          { role: "system", content: "You are a cat." },
          { role: "user", content: "where do you sit?" },
          { role: "assistant", content: "on the mat" },
        ],
      },
    ]),
  },
  {
    id: "count",
    name: "计数",
    prompt: "从0数到9",
    text: pretty([
      {
        messages: [
          { role: "system", content: "只输出数字。" },
          { role: "user", content: "从0数到9" },
          { role: "assistant", content: "0123456789" },
        ],
      },
      {
        messages: [
          { role: "system", content: "只输出数字。" },
          { role: "user", content: "再数一遍" },
          { role: "assistant", content: "0123456789" },
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
