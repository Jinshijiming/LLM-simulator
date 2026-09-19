export const SPECIALS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"];

export const SPECIAL_LABEL = {
  "<|system|>": "sys",
  "<|user|>": "usr",
  "<|assistant|>": "ast",
  "<|end|>": "end",
};

export class CharTokenizer {
  constructor(text) {
    const chars = [];
    const seen = new Set();
    for (const sp of SPECIALS) {
      seen.add(sp);
      chars.push(sp);
    }
    for (const ch of String(text ?? "")) {
      if (!seen.has(ch)) {
        seen.add(ch);
        chars.push(ch);
      }
    }
    if (chars.length === SPECIALS.length && !String(text ?? "").length) {
      /* specials only is fine */
    }
    this.chars = chars;
    this.stoi = new Map(chars.map((c, i) => [c, i]));
    this.itos = chars;
    this.vocabSize = chars.length;
    this.systemId = this.stoi.get("<|system|>");
    this.userId = this.stoi.get("<|user|>");
    this.assistantId = this.stoi.get("<|assistant|>");
    this.endId = this.stoi.get("<|end|>");
  }

  encode(text) {
    const ids = [];
    const dropped = [];
    const s = String(text ?? "");
    let i = 0;
    while (i < s.length) {
      let hit = null;
      for (const sp of SPECIALS) {
        if (s.startsWith(sp, i)) {
          hit = sp;
          break;
        }
      }
      if (hit) {
        ids.push(this.stoi.get(hit));
        i += hit.length;
        continue;
      }
      const cp = s.codePointAt(i);
      const ch = String.fromCodePoint(cp);
      const id = this.stoi.get(ch);
      if (id === undefined) dropped.push(ch);
      else ids.push(id);
      i += ch.length;
    }
    return { ids: Int32Array.from(ids), dropped };
  }

  decode(ids) {
    let s = "";
    for (const id of ids) s += this.itos[id] ?? "";
    return s;
  }

  display(ch) {
    if (SPECIAL_LABEL[ch]) return SPECIAL_LABEL[ch];
    if (ch === " ") return "␣";
    if (ch === "\n") return "↵";
    if (ch === "\t") return "⇥";
    return ch;
  }

  displayId(id) {
    return this.display(this.itos[id] ?? "?");
  }

  isSpecial(id) {
    const t = this.itos[id];
    return !!SPECIAL_LABEL[t];
  }

  isEnd(id) {
    return id === this.endId;
  }

  roleOfToken(id) {
    const t = this.itos[id];
    if (t === "<|system|>") return "system";
    if (t === "<|user|>") return "user";
    if (t === "<|assistant|>") return "assistant";
    return null;
  }
}
