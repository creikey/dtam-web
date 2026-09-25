// Smoke test for a chapter file, without a browser:
//   node web/learn/tools/check.mjs web/learn/chapters/03-camera.js
// Runs render() against a stub DOM (every DOM/canvas call is a no-op), then
// validates every quiz question: ids unique, mc/multi answers in range, and
// each numeric generator produces finite answers and a prompt for 200 seeds.
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const file = process.argv[2];
if (!file) {
  console.error("usage: node check.mjs <chapter.js>");
  process.exit(2);
}

// A value that absorbs any property access / call / construction.
const sink = () =>
  new Proxy(function () {}, {
    get: (_, k) => {
      if (k === Symbol.toPrimitive) return () => 0;
      if (k === Symbol.iterator) return function* () {};
      if (k === "then") return undefined;
      if (k === "length") return 0;
      return sink();
    },
    apply: () => sink(),
    construct: () => sink(),
    set: () => true,
  });

const ctx = {
  console, Math, JSON, Date, Array, Object, Number, String, Symbol, Map, Set, Promise, Float32Array, Float64Array,
  Uint8Array, Uint8ClampedArray, Int32Array, Uint32Array, Error, isFinite, isNaN, parseFloat, parseInt,
  setTimeout: () => 0, setInterval: () => 0, clearTimeout: () => 0, requestAnimationFrame: () => 0,
  performance: { now: () => 0 },
  localStorage: { getItem: () => null, setItem: () => {} },
  document: sink(),
  getComputedStyle: () => ({ getPropertyValue: () => "#888" }),
  ResizeObserver: class { observe() {} },
  IntersectionObserver: class { observe() {} },
  Image: class {},
};
ctx.window = ctx;
ctx.window.addEventListener = () => {};
ctx.window.matchMedia = undefined;
ctx.document.hidden = false;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(here, "..", "core.js"), "utf8"), ctx, { filename: "core.js" });

const real = ctx.DTAM.lib;
const captured = [];
const keep = new Set(["rng", "fmt", "la", "tex", "makeImage", "viridisish", "gray"]);
const L = new Proxy(real, {
  get(t, k) {
    if (k === "quiz") {
      return (parent, chapterId, questions) => {
        captured.push({ chapterId, questions });
        return sink();
      };
    }
    if (keep.has(k)) return t[k];
    return sink();
  },
});

const defs = [];
ctx.DTAM.chapter = (d) => defs.push(d);
try {
  vm.runInContext(fs.readFileSync(file, "utf8"), ctx, { filename: file });
} catch (e) {
  console.error("✗ script error:", e.stack);
  process.exit(1);
}
let failed = 0;
const fail = (m) => { failed++; console.error("✗ " + m); };
if (defs.length !== 1) fail(`expected exactly one DTAM.chapter(...) call, got ${defs.length}`);
for (const d of defs) {
  for (const k of ["id", "order", "title", "minutes", "render"]) if (d[k] === undefined) fail(`chapter missing "${k}"`);
  try {
    d.render(sink(), L);
  } catch (e) {
    fail(`render() threw against the stub DOM (may be a stub artefact; check in a browser): ${e.stack.split("\n").slice(0, 3).join(" | ")}`);
  }
  const all = captured.flatMap((c) => c.questions.map((q) => ({ ...q, chapterId: c.chapterId })));
  if (!all.length) fail("no quiz questions found (L.quiz was never called)");
  for (const c of captured) if (c.chapterId !== d.id) fail(`quiz chapterId "${c.chapterId}" != chapter id "${d.id}"`);
  const ids = new Set();
  let nNum = 0;
  for (const q of all) {
    if (!q.id) fail("question without id");
    if (ids.has(q.id)) fail(`duplicate question id "${q.id}"`);
    ids.add(q.id);
    if (q.type === "mc") {
      if (!Array.isArray(q.choices) || !(q.answer >= 0 && q.answer < q.choices.length)) fail(`mc "${q.id}": bad answer index`);
      if (!q.q) fail(`mc "${q.id}": no prompt`);
    } else if (q.type === "multi") {
      if (!Array.isArray(q.answer) || !q.answer.length || q.answer.some((a) => !(a >= 0 && a < q.choices.length))) fail(`multi "${q.id}": bad answers`);
    } else if (q.type === "num") {
      nNum++;
      if (typeof q.gen !== "function") { fail(`num "${q.id}": no gen()`); continue; }
      for (let s = 1; s <= 200; s++) {
        let inst;
        try { inst = q.gen(real.rng(s * 2654435761)); } catch (e) { fail(`num "${q.id}" seed ${s}: gen threw ${e.message}`); break; }
        const ans = Array.isArray(inst.answer) ? inst.answer : [inst.answer];
        if (!inst.q || typeof inst.q !== "string") { fail(`num "${q.id}": no prompt string`); break; }
        if (ans.some((v) => typeof v !== "number" || !isFinite(v))) { fail(`num "${q.id}" seed ${s}: non-finite answer ${JSON.stringify(inst.answer)}`); break; }
        if (inst.labels && inst.labels.length !== ans.length) { fail(`num "${q.id}": labels/answers length mismatch`); break; }
        if (/NaN|undefined|Infinity/.test(inst.q)) { fail(`num "${q.id}" seed ${s}: prompt contains NaN/undefined: ${inst.q.slice(0, 120)}`); break; }
      }
    } else fail(`question "${q.id}": unknown type ${q.type}`);
  }
  console.log(`${d.id}: ${all.length} questions (${nNum} randomized numeric), ${d.minutes} min`);
}
if (failed) {
  console.error(`${failed} problem(s)`);
  process.exit(1);
}
console.log("✓ ok");
