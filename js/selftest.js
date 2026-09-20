import { RNG } from "./rng.js";
import { CharTokenizer } from "./tokenizer.js";
import { parseCorpus, flattenContents, packCorpus, loopStream, makeBatch } from "./datasets.js";
import { TinyGPT } from "./gpt.js";

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

const json = JSON.stringify([
  {
    messages: [
      { role: "system", content: "cat" },
      { role: "user", content: "hi" },
      { role: "assistant", content: "ab" },
    ],
  },
  {
    messages: [
      { role: "system", content: "cat" },
      { role: "user", content: "yo" },
      { role: "assistant", content: "ab" },
    ],
  },
]);

const parsed = parseCorpus(json);
assert(parsed.conversations.length === 2, "parse 2 convos");
const tok = new CharTokenizer(flattenContents(parsed.conversations));
assert(tok.isSpecial(tok.systemId), "system special");
const packed = packCorpus(parsed.conversations, tok);
assert(packed.mask.some((x) => x === 1), "has assistant mask");
assert(packed.mask.some((x) => x === 0), "has ignored prefix");

const stream = loopStream(packed, 80);
const model = new TinyGPT({
  vocabSize: tok.vocabSize,
  blockSize: 16,
  nEmbd: 32,
  nHead: 4,
  nLayer: 1,
  rng: new RNG(1),
});
const trainRng = new RNG(7);
const losses = [];
for (let s = 0; s < 80; s++) {
  const batch = makeBatch(stream, 8, 16, trainRng);
  const { loss } = model.trainStep(batch.x, batch.y, 8, 16, 0.05, "adam", batch.mask);
  if (s % 20 === 0) losses.push(loss);
}
console.log("losses", losses.map((v) => v.toFixed(3)).join(" -> "));
assert(losses[0] > losses[losses.length - 1], "masked loss should drop");
const dumped = model.dumpWeights();
const model2 = new TinyGPT({
  vocabSize: tok.vocabSize,
  blockSize: 16,
  nEmbd: 32,
  nHead: 4,
  nLayer: 1,
  rng: new RNG(99),
});
model2.loadWeights(dumped);
assert(model2.tokEmb.w[0] === model.tokEmb.w[0], "reload tokEmb");
console.log("ok");
