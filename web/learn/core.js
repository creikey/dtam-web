// "How DTAM works": chapter registry, quiz engine, progress + ETA, and the
// small widget/maths library chapters use (window.DTAM.lib, passed as `L`).
(() => {
  "use strict";
  const STORE_KEY = "dtam-learn-v1";
  const chapters = [];

  // ---------------------------------------------------------------- storage
  function load() {
    try {
      const s = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
      return { solved: s.solved || {}, time: s.time || {}, seeds: s.seeds || {} };
    } catch (e) {
      return { solved: {}, time: {}, seeds: {} };
    }
  }
  const store = load();
  function save() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(store)); } catch (e) { /* private mode */ }
  }

  // ---------------------------------------------------------------- rng
  function rng(seed) {
    let s = (seed >>> 0) || 1;
    const next = () => {
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
    const r = {
      next,
      /** Integer in [a, b]. */
      int: (a, b) => a + Math.floor(next() * (b - a + 1)),
      /** Float in [a, b), rounded to `dec` decimals. */
      float: (a, b, dec = 2) => +(a + next() * (b - a)).toFixed(dec),
      pick: (arr) => arr[Math.floor(next() * arr.length)],
      sign: () => (next() < 0.5 ? -1 : 1),
      /** Nonzero integer in [-a, a]. */
      nz: (a) => (next() < 0.5 ? -1 : 1) * (1 + Math.floor(next() * a)),
      shuffle: (arr) => {
        const a = arr.slice();
        for (let i = a.length - 1; i > 0; i--) {
          const j = Math.floor(next() * (i + 1));
          [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
      },
    };
    return r;
  }
  const newSeed = () => (Math.random() * 4294967295) >>> 0 || 7;

  // ---------------------------------------------------------------- DOM + math
  function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c != null) e.append(c);
    return e;
  }
  const hasKatex = () => typeof window.katex !== "undefined";
  /** TeX -> HTML string (use inside templates when you build strings). */
  function tex(src, display = false) {
    if (!hasKatex()) return display ? `<div class="katex-display">${src}</div>` : `<span>${src}</span>`;
    return window.katex.renderToString(src, { displayMode: display, throwOnError: false, strict: false });
  }
  /** Renders $...$ and $$...$$ inside an element. */
  function typeset(root) {
    if (typeof window.renderMathInElement === "function") {
      window.renderMathInElement(root, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
        strict: false,
        ignoredClasses: ["no-tex"],
      });
    }
  }

  // ---------------------------------------------------------------- theme colours
  function theme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    return {
      bg: v("--bg"), panel: v("--panel"), panel2: v("--panel-2"), fg: v("--fg"), muted: v("--muted"),
      faint: v("--faint"), line: v("--line"), accent: v("--accent"), accent2: v("--accent-2"),
      accent3: v("--accent-3"), accent4: v("--accent-4"), good: v("--good"), bad: v("--bad"),
      mono: v("--mono"), font: v("--font"),
    };
  }
  const themeListeners = new Set();
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
    for (const f of themeListeners) f();
  });

  // ---------------------------------------------------------------- widgets
  /** A figure box appended to `parent`. */
  function figure(parent, caption) {
    const f = el("figure");
    const cap = caption ? el("figcaption", { html: caption }) : null;
    if (cap) f.append(cap);
    parent.append(f);
    const api = {
      el: f,
      /** Adds children before the caption. */
      add(...nodes) { for (const n of nodes) (cap ? f.insertBefore(n, cap) : f.append(n)); return api; },
      caption: cap,
    };
    if (cap) typeset(cap);
    return api;
  }

  /**
   * Responsive hi-DPI canvas. `aspect` = height / width; the drawing size in
   * CSS pixels is (c.w, c.h). Set `c.draw = (ctx, c) => ...` and call
   * c.redraw() whenever state changes (resizes and theme changes redraw).
   */
  function canvas(parent, { aspect = 0.6, maxHeight = 520, scroll = false } = {}) {
    const cv = el("canvas", scroll ? { class: "scroll-ok" } : {});
    parent.append(cv);
    const ctx = cv.getContext("2d");
    const c = { el: cv, ctx, w: 300, h: 180, dpr: 1, draw: null, visible: false };
    let queued = false;
    c.redraw = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        if (!c.draw) return;
        ctx.setTransform(c.dpr, 0, 0, c.dpr, 0, 0);
        ctx.clearRect(0, 0, c.w, c.h);
        c.draw(ctx, c);
      });
    };
    const resize = () => {
      const w = Math.max(120, cv.parentElement ? cv.getBoundingClientRect().width : 300);
      const h = Math.min(maxHeight, Math.round(w * aspect));
      c.dpr = Math.min(window.devicePixelRatio || 1, 2.5);
      c.w = w;
      c.h = h;
      cv.style.height = h + "px";
      cv.width = Math.round(w * c.dpr);
      cv.height = Math.round(h * c.dpr);
      c.onResize?.(c);
      c.redraw();
    };
    new ResizeObserver(resize).observe(cv);
    new IntersectionObserver((es) => { c.visible = es[0].isIntersecting; }).observe(cv);
    themeListeners.add(() => c.redraw());
    requestAnimationFrame(resize);
    /** Pointer position (CSS px) of an event relative to the canvas. */
    c.pos = (e) => {
      const r = cv.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    return c;
  }

  /**
   * Draggable handles on a canvas. `points()` returns the current array of
   * {x, y} in CSS px; onMove(index, {x, y}) updates state. Touch friendly.
   */
  function drag(c, points, onMove, { radius = 18, onEnd } = {}) {
    let active = -1;
    const pick = (p) => {
      let best = -1, bd = radius * radius;
      points().forEach((q, i) => {
        const d = (q.x - p.x) ** 2 + (q.y - p.y) ** 2;
        if (d <= bd) { bd = d; best = i; }
      });
      return best;
    };
    c.el.addEventListener("pointerdown", (e) => {
      const i = pick(c.pos(e));
      if (i < 0) return;
      active = i;
      c.el.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    c.el.addEventListener("pointermove", (e) => {
      const p = c.pos(e);
      if (active >= 0) {
        onMove(active, { x: Math.max(0, Math.min(c.w, p.x)), y: Math.max(0, Math.min(c.h, p.y)) });
        c.redraw();
      } else {
        c.el.style.cursor = pick(p) >= 0 ? "grab" : "";
      }
    });
    const end = () => { if (active >= 0) { active = -1; onEnd?.(); } };
    c.el.addEventListener("pointerup", end);
    c.el.addEventListener("pointercancel", end);
  }

  /** Runs fn(t, dt) every animation frame while the canvas is on screen. */
  function loop(c, fn) {
    let last = performance.now();
    const tick = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (c.visible && !document.hidden) fn(now / 1000, dt);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function controls(parent) {
    const d = el("div", { class: "controls" });
    parent.append(d);
    return d;
  }

  /** Labelled range slider. Returns {input, value (getter/setter)}. */
  function slider(parent, { label, min = 0, max = 1, step = 0.01, value = 0, fmt, oninput }) {
    const id = "s" + Math.random().toString(36).slice(2);
    const input = el("input", { type: "range", id, min, max, step, value });
    const out = el("output", { for: id });
    const f = fmt || ((v) => (Math.abs(v) >= 100 || Number.isInteger(+step) ? String(+v) : (+v).toFixed(2)));
    const wrap = el("div", { class: "slider" }, el("label", { for: id, html: label }), input, out);
    parent.append(wrap);
    typeset(wrap);
    const sync = () => { out.textContent = f(+input.value); };
    input.addEventListener("input", () => { sync(); oninput?.(+input.value); });
    sync();
    return {
      input,
      get value() { return +input.value; },
      set value(v) { input.value = v; sync(); },
    };
  }

  function button(parent, label, onclick, cls = "btn") {
    const b = el("button", { class: cls, type: "button", html: label, onclick });
    parent.append(b);
    return b;
  }

  function toggle(parent, label, checked, onchange) {
    const input = el("input", { type: "checkbox" });
    input.checked = !!checked;
    input.addEventListener("change", () => onchange?.(input.checked));
    const l = el("label", { class: "toggle" }, input, el("span", { html: label }));
    parent.append(l);
    return input;
  }

  /** A monospace line under a widget; set .html to update. */
  function readout(parent) {
    const d = el("div", { class: "readout" });
    parent.append(d);
    return {
      el: d,
      set html(h) { d.innerHTML = h; },
    };
  }

  // ---------------------------------------------------------------- drawing helpers
  const draw = {
    line(ctx, x0, y0, x1, y1, color, width = 1.5, dash) {
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = width;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.restore();
    },
    arrow(ctx, x0, y0, x1, y1, color, width = 2, head = 9) {
      const a = Math.atan2(y1 - y0, x1 - x0);
      ctx.save();
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(a) * head * 0.6, y1 - Math.sin(a) * head * 0.6); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - head * Math.cos(a - 0.4), y1 - head * Math.sin(a - 0.4));
      ctx.lineTo(x1 - head * Math.cos(a + 0.4), y1 - head * Math.sin(a + 0.4));
      ctx.closePath(); ctx.fill();
      ctx.restore();
    },
    dot(ctx, x, y, r, color, stroke) {
      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
      ctx.restore();
    },
    /** Draggable-handle look. */
    handle(ctx, x, y, color) {
      draw.dot(ctx, x, y, 9, color, "white");
    },
    text(ctx, s, x, y, color, { size = 13, align = "left", base = "alphabetic", bold = false, mono = false } = {}) {
      const t = theme();
      ctx.save();
      ctx.fillStyle = color;
      ctx.font = `${bold ? "600 " : ""}${size}px ${mono ? t.mono : t.font}`;
      ctx.textAlign = align; ctx.textBaseline = base;
      ctx.fillText(s, x, y);
      ctx.restore();
    },
    /** Polyline through [[x, y], ...]. */
    path(ctx, pts, color, width = 2, dash) {
      if (pts.length < 2) return;
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round";
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
      ctx.stroke();
      ctx.restore();
    },
  };

  /**
   * A 2D plot frame: maps data (x in [x0,x1], y in [y0,y1]) into the canvas
   * with padding. Returns {X, Y, invX, invY, axes(ctx, opts)}.
   */
  function plot(c, { x0 = 0, x1 = 1, y0 = 0, y1 = 1, pad = [16, 16, 28, 38] } = {}) {
    const [pt, pr, pb, pl] = pad;
    const p = {
      x0, x1, y0, y1,
      X: (x) => pl + ((x - p.x0) / (p.x1 - p.x0)) * (c.w - pl - pr),
      Y: (y) => c.h - pb - ((y - p.y0) / (p.y1 - p.y0)) * (c.h - pt - pb),
      invX: (px) => p.x0 + ((px - pl) / (c.w - pl - pr)) * (p.x1 - p.x0),
      invY: (py) => p.y0 + ((c.h - pb - py) / (c.h - pt - pb)) * (p.y1 - p.y0),
      axes(ctx, { xlabel = "", ylabel = "", xticks = 5, yticks = 4, fmt = (v) => +v.toFixed(2) } = {}) {
        const t = theme();
        ctx.save();
        ctx.strokeStyle = t.line; ctx.lineWidth = 1;
        for (let i = 0; i <= xticks; i++) {
          const v = p.x0 + (i / xticks) * (p.x1 - p.x0);
          draw.line(ctx, p.X(v), pt, p.X(v), c.h - pb, t.line, 1);
          draw.text(ctx, String(fmt(v)), p.X(v), c.h - pb + 15, t.faint, { size: 11, align: "center" });
        }
        for (let i = 0; i <= yticks; i++) {
          const v = p.y0 + (i / yticks) * (p.y1 - p.y0);
          draw.line(ctx, pl, p.Y(v), c.w - pr, p.Y(v), t.line, 1);
          draw.text(ctx, String(fmt(v)), pl - 6, p.Y(v) + 4, t.faint, { size: 11, align: "right" });
        }
        if (xlabel) draw.text(ctx, xlabel, c.w - pr, c.h - 4, t.muted, { size: 12, align: "right" });
        if (ylabel) draw.text(ctx, ylabel, pl + 4, pt + 10, t.muted, { size: 12 });
        ctx.restore();
      },
    };
    return p;
  }

  /** Colour maps, v in [0,1] -> "rgb(...)". */
  function viridisish(v) {
    v = Math.max(0, Math.min(1, v));
    const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
    const f = v * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), t = f - i;
    const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  function gray(v) {
    const g = Math.round(Math.max(0, Math.min(1, v)) * 255);
    return `rgb(${g},${g},${g})`;
  }

  // ---------------------------------------------------------------- images
  /**
   * Procedural grayscale test images, values in [0,1], row-major Float32Array.
   * kinds: "scene" (textured blobs + edges + a flat region), "checker",
   * "blobs", "stripes", "flat".
   */
  function makeImage(w, h, kind = "scene", seed = 3) {
    const r = rng(seed);
    const img = new Float32Array(w * h);
    const blobs = Array.from({ length: 14 }, () => [r.next() * w, r.next() * h, 2 + r.next() * w * 0.12, r.next() * 0.8 - 0.4]);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let v;
        if (kind === "checker") v = ((Math.floor(x / (w / 8)) + Math.floor(y / (h / 8))) % 2) ? 0.8 : 0.2;
        else if (kind === "stripes") v = 0.5 + 0.4 * Math.sin((x / w) * Math.PI * 10);
        else if (kind === "flat") v = 0.6;
        else {
          v = 0.45;
          for (const [bx, by, br, a] of blobs) v += a * Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * br * br));
          if (kind === "scene") {
            if (x > w * 0.62 && y < h * 0.45) v = 0.82; // flat, textureless patch
            if (Math.abs(y - (0.72 * h - 0.25 * x)) < 1.2) v = 0.1; // a thin dark line
          }
        }
        img[y * w + x] = Math.max(0, Math.min(1, v));
      }
    }
    return { w, h, data: img };
  }
  /** Loads an image URL into {w, h, data (gray Float32 0..1), rgb (Uint8 RGBA)} . */
  function loadImage(url, size) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => {
        const w = size || im.width, h = size ? Math.round((im.height / im.width) * size) : im.height;
        const cv = document.createElement("canvas");
        cv.width = w; cv.height = h;
        const cx = cv.getContext("2d");
        cx.drawImage(im, 0, 0, w, h);
        const px = cx.getImageData(0, 0, w, h).data;
        const data = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) data[i] = (0.2126 * px[4 * i] + 0.7152 * px[4 * i + 1] + 0.0722 * px[4 * i + 2]) / 255; // Rec. 709, as in frame.rs
        resolve({ w, h, data, rgba: px });
      };
      im.onerror = reject;
      im.src = url;
    });
  }
  /** Draws a gray image (data in [0,1]) into a rect of the canvas, nearest-neighbour. */
  function drawGray(ctx, img, x, y, w, h, map = gray) {
    const cv = document.createElement("canvas");
    cv.width = img.w; cv.height = img.h;
    const cx = cv.getContext("2d");
    const id = cx.createImageData(img.w, img.h);
    for (let i = 0; i < img.w * img.h; i++) {
      const m = map(img.data[i]);
      const [r, g, b] = m.match(/\d+/g).map(Number);
      id.data[4 * i] = r; id.data[4 * i + 1] = g; id.data[4 * i + 2] = b; id.data[4 * i + 3] = 255;
    }
    cx.putImageData(id, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cv, x, y, w, h);
    ctx.restore();
  }

  // ---------------------------------------------------------------- small linear algebra
  const la = {
    dot: (a, b) => a.reduce((s, v, i) => s + v * b[i], 0),
    add: (a, b) => a.map((v, i) => v + b[i]),
    sub: (a, b) => a.map((v, i) => v - b[i]),
    scale: (a, s) => a.map((v) => v * s),
    norm: (a) => Math.hypot(...a),
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    /** Matrices are arrays of rows. */
    matVec: (M, v) => M.map((row) => la.dot(row, v)),
    matMul: (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0))),
    T: (A) => A[0].map((_, j) => A.map((row) => row[j])),
    eye: (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j))),
    /** Solves A x = b by Gaussian elimination with partial pivoting (null if singular). */
    solve(A, b) {
      const n = b.length;
      const M = A.map((row, i) => [...row, b[i]]);
      for (let c = 0; c < n; c++) {
        let p = c;
        for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
        if (Math.abs(M[p][c]) < 1e-12) return null;
        [M[c], M[p]] = [M[p], M[c]];
        for (let r = c + 1; r < n; r++) {
          const f = M[r][c] / M[c][c];
          for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
        }
      }
      const x = new Array(n).fill(0);
      for (let r = n - 1; r >= 0; r--) {
        let s = M[r][n];
        for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
        x[r] = s / M[r][r];
      }
      return x;
    },
    skew: (w) => [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]],
    /** Rotation matrix exp([w]x) (Rodrigues). */
    expSO3(w) {
      const th = Math.hypot(...w);
      const K = la.skew(w), K2 = la.matMul(K, K), I = la.eye(3);
      const a = th < 1e-8 ? 1 : Math.sin(th) / th;
      const b = th < 1e-8 ? 0.5 : (1 - Math.cos(th)) / (th * th);
      return I.map((row, i) => row.map((v, j) => v + a * K[i][j] + b * K2[i][j]));
    },
    rotX: (a) => [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]],
    rotY: (a) => [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]],
    rotZ: (a) => [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]],
  };

  /** Number formatting for prompts: trims trailing zeros. */
  const fmt = (v, d = 3) => {
    if (!isFinite(v)) return String(v);
    const s = (+v).toFixed(d);
    return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
  };

  // ---------------------------------------------------------------- quizzes
  /** Parses "3", "-1.5", "1/3", "2e-3", "sqrt(2)", "pi/4" etc. (a safe subset). */
  function parseNum(s) {
    s = String(s).trim().toLowerCase().replace(/,/g, ".").replace(/−/g, "-").replace(/×/g, "*").replace(/π/g, "pi");
    if (!s) return NaN;
    if (!/^[0-9+\-*/().e\s^a-z]*$/.test(s)) return NaN;
    const words = s.match(/[a-z]+/g) || [];
    for (const w of words) if (!["sqrt", "pi", "e", "exp", "ln", "log", "sin", "cos", "tan"].includes(w)) return NaN;
    try {
      const js = s.replace(/\^/g, "**").replace(/\bpi\b/g, "Math.PI").replace(/\bsqrt\b/g, "Math.sqrt")
        .replace(/\bexp\b/g, "Math.exp").replace(/\bln\b/g, "Math.log").replace(/\blog\b/g, "Math.log10")
        .replace(/\bsin\b/g, "Math.sin").replace(/\bcos\b/g, "Math.cos").replace(/\btan\b/g, "Math.tan")
        .replace(/(^|[^a-z0-9.])e(?![a-z0-9(])/g, "$1Math.E");
      // eslint-disable-next-line no-new-func
      const v = Function(`"use strict"; return (${js});`)();
      return typeof v === "number" ? v : NaN;
    } catch (e) {
      return NaN;
    }
  }
  function close(got, want, tol, rtol) {
    if (!isFinite(got)) return false;
    const t = Math.max(tol ?? 0, (rtol ?? 0) * Math.abs(want));
    return Math.abs(got - want) <= (t || 1e-9 + 1e-6 * Math.abs(want));
  }

  const quizzes = [];

  /**
   * Adds a quiz to `parent`. Each question needs a stable `id` (unique within
   * the chapter). Types:
   *  { id, type: "mc", q, choices: [...], answer: i, explain }
   *  { id, type: "multi", q, choices: [...], answer: [i, j], explain }
   *  { id, type: "num", gen: (r) => ({ q, answer: number | number[], labels?: [...],
   *        tol?, rtol?, explain }) }            // r: seeded rng, new values each try
   * q / choices / explain are HTML strings; $...$ TeX is typeset.
   */
  function quiz(parent, chapterId, questions, { title = "Check your understanding" } = {}) {
    const wrap = el("div", { class: "quiz" });
    const head = el("div", { class: "quiz-head" }, el("span", { text: title }), el("span", { class: "count" }));
    wrap.append(head);
    parent.append(wrap);
    const entries = [];
    questions.forEach((spec, idx) => {
      const qid = `${chapterId}/${spec.id}`;
      const card = el("div", { class: "q" });
      wrap.append(card);
      const entry = { qid, spec, card, chapterId };
      entries.push(entry);
      quizzes.push(entry);
      renderQuestion(entry, idx + 1, questions.length, () => updateCount());
    });
    const updateCount = () => {
      const n = entries.filter((e) => store.solved[e.qid]).length;
      head.querySelector(".count").textContent = `${n} / ${entries.length} solved`;
      refreshProgress();
    };
    updateCount();
    return wrap;
  }

  function renderQuestion(entry, num, total, onSolved) {
    const { spec, card, qid } = entry;
    const solved = !!store.solved[qid];
    let seed = store.seeds[qid] || newSeed();
    let shuffleSeed = newSeed();
    card.innerHTML = "";
    card.classList.toggle("solved", solved);
    const qnum = el("div", { class: "qnum" }, el("span", { text: `Question ${num} of ${total}` }), el("span", { text: solved ? "✓ solved" : "" }));
    card.append(qnum);
    const promptEl = el("div", { class: "prompt" });
    const body = el("div");
    const fb = el("div", { class: "feedback" });
    const actions = el("div", { class: "actions" });
    card.append(promptEl, body, actions, fb);

    let inst; // the concrete question
    let check; // () => boolean
    const build = () => {
      body.innerHTML = "";
      fb.className = "feedback";
      fb.innerHTML = "";
      if (spec.type === "num") {
        inst = spec.gen(rng(seed));
        promptEl.innerHTML = inst.q;
        const answers = Array.isArray(inst.answer) ? inst.answer : [inst.answer];
        const labels = inst.labels || (answers.length > 1 ? answers.map((_, i) => `#${i + 1}`) : [""]);
        const nums = el("div", { class: "nums" });
        const inputs = answers.map((_, i) => {
          const input = el("input", { type: "text", inputmode: "decimal", autocomplete: "off", spellcheck: "false", "aria-label": labels[i] || "answer" });
          input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
          nums.append(el("label", { class: "num-field" }, labels[i] ? el("span", { html: labels[i] }) : null, input));
          return input;
        });
        body.append(nums, el("div", { class: "hint-note", text: "Numbers, fractions (1/3) or expressions like sqrt(2), pi/4, 2^10 are accepted." }));
        check = () => inputs.every((inp, i) => close(parseNum(inp.value), answers[i], inst.tol, inst.rtol));
      } else {
        const r = rng(shuffleSeed);
        const order = spec.shuffle === false ? spec.choices.map((_, i) => i) : r.shuffle(spec.choices.map((_, i) => i));
        promptEl.innerHTML = spec.q;
        const multi = spec.type === "multi";
        const name = "n" + Math.random().toString(36).slice(2);
        const boxes = [];
        const list = el("div", { class: "choices" });
        for (const i of order) {
          const input = el("input", { type: multi ? "checkbox" : "radio", name });
          const lab = el("label", { class: "choice" }, input, el("span", { html: spec.choices[i] }));
          list.append(lab);
          boxes.push({ i, input, lab });
        }
        body.append(list);
        inst = { explain: spec.explain };
        check = () => {
          const picked = boxes.filter((b) => b.input.checked).map((b) => b.i).sort((a, b) => a - b);
          const want = (multi ? spec.answer : [spec.answer]).slice().sort((a, b) => a - b);
          const ok = picked.length === want.length && picked.every((v, k) => v === want[k]);
          for (const b of boxes) {
            b.lab.classList.remove("right", "wrong");
            if (b.input.checked) b.lab.classList.add(want.includes(b.i) ? "right" : "wrong");
          }
          return ok;
        };
      }
      typeset(card);
    };

    const submit = () => {
      const ok = check();
      if (ok) {
        fb.className = "feedback show good";
        fb.innerHTML = "<b>Correct.</b> " + (inst.explain || "");
        if (!store.solved[qid]) {
          store.solved[qid] = Date.now();
          store.seeds[qid] = seed;
          save();
        }
        card.classList.add("solved");
        qnum.lastChild.textContent = "✓ solved";
        onSolved();
      } else {
        fb.className = "feedback show bad";
        fb.innerHTML = "<b>Not quite.</b> " + (spec.type === "num"
          ? "Check your working (the explanation appears once you have it right, or press “Show solution”)."
          : spec.type === "multi" ? "Select every correct option and nothing else." : "Try again — the choices have been reordered.");
        if (spec.type !== "num") {
          shuffleSeed = newSeed();
          setTimeout(build, 1400);
        }
      }
      typeset(fb);
    };
    const checkBtn = button(actions, "Check", submit, "btn primary");
    if (spec.type === "num") {
      button(actions, "Show solution", () => {
        fb.className = "feedback show bad";
        const ans = Array.isArray(inst.answer) ? inst.answer.map((v) => fmt(v, 4)).join(", ") : fmt(inst.answer, 4);
        fb.innerHTML = `<b>Answer: ${ans}.</b> ${inst.explain || ""}<br><i>Press “New numbers” and solve a fresh version to get credit.</i>`;
        typeset(fb);
        checkBtn.disabled = true;
      });
      button(actions, "New numbers", () => {
        seed = newSeed();
        checkBtn.disabled = false;
        build();
      });
    }
    build();
  }

  // ---------------------------------------------------------------- progress + ETA
  function chapterStats(ch) {
    const qs = quizzes.filter((q) => q.chapterId === ch.id);
    const done = qs.filter((q) => store.solved[q.qid]).length;
    return { total: qs.length, done, frac: qs.length ? done / qs.length : 0 };
  }
  function fmtDuration(min) {
    if (min < 1) return "< 1 min";
    const h = Math.floor(min / 60), m = Math.round(min % 60);
    return h ? `${h} h ${String(m).padStart(2, "0")} min` : `${m} min`;
  }
  let progressEls = null;
  function refreshProgress() {
    if (!progressEls) return;
    let est = 0, estDone = 0, qTotal = 0, qDone = 0;
    for (const ch of chapters) {
      const s = chapterStats(ch);
      est += ch.minutes;
      estDone += ch.minutes * s.frac;
      qTotal += s.total;
      qDone += s.done;
      const ring = document.querySelector(`[data-ring="${ch.id}"]`);
      if (ring) ring.innerHTML = ringSvg(s.frac);
      const meta = document.querySelector(`[data-meta="${ch.id}"]`);
      if (meta) meta.textContent = `${s.done}/${s.total} · ${ch.minutes} min`;
      const doneEl = document.querySelector(`[data-done="${ch.id}"]`);
      if (doneEl) doneEl.classList.toggle("show", s.total > 0 && s.done === s.total);
    }
    const frac = est ? estDone / est : 0;
    // Personal pace: time actually spent vs. the estimate for what is done.
    const spent = Object.values(store.time).reduce((a, b) => a + b, 0) / 60;
    let pace = 1;
    if (estDone > 10 && spent > 5) pace = Math.max(0.4, Math.min(2.5, spent / estDone));
    const left = (est - estDone) * pace;
    progressEls.fill.style.width = (frac * 100).toFixed(1) + "%";
    progressEls.stats.innerHTML = qDone === qTotal && qTotal
      ? "<b>Complete</b> — every question solved"
      : `<b>${Math.floor(frac * 100)}%</b> · ${qDone}/${qTotal} questions · ≈ <b>${fmtDuration(left)}</b> left`;
  }
  function ringSvg(f) {
    const t = theme();
    const r = 7, C = 2 * Math.PI * r;
    return `<svg class="ring" viewBox="0 0 18 18"><circle cx="9" cy="9" r="${r}" fill="none" stroke="${t.line}" stroke-width="3"/>` +
      `<circle cx="9" cy="9" r="${r}" fill="none" stroke="${f >= 1 ? t.good : t.accent}" stroke-width="3" stroke-dasharray="${(f * C).toFixed(2)} ${C.toFixed(2)}" transform="rotate(-90 9 9)" stroke-linecap="${f > 0 ? "round" : "butt"}"/></svg>`;
  }

  // Active reading time, attributed to the chapter nearest the viewport centre.
  let lastInteraction = Date.now();
  for (const ev of ["pointerdown", "keydown", "scroll", "wheel", "touchstart", "input"]) {
    window.addEventListener(ev, () => { lastInteraction = Date.now(); }, { passive: true });
  }
  function currentChapter() {
    const mid = window.innerHeight / 2;
    let best = null;
    for (const ch of chapters) {
      const s = document.getElementById(ch.id);
      if (!s) continue;
      const r = s.getBoundingClientRect();
      if (r.top <= mid && r.bottom >= mid) best = ch;
    }
    return best;
  }
  setInterval(() => {
    if (document.hidden || Date.now() - lastInteraction > 120000) return;
    const ch = currentChapter();
    if (!ch) return;
    store.time[ch.id] = (store.time[ch.id] || 0) + 5;
    save();
    refreshProgress();
  }, 5000);

  // ---------------------------------------------------------------- page assembly
  function chapter(def) {
    chapters.push(def);
  }

  function build() {
    chapters.sort((a, b) => a.order - b.order);
    const main = document.getElementById("chapters");
    const toc = document.getElementById("toc-list");
    chapters.forEach((ch, i) => {
      toc.append(el("li", {}, el("a", { href: `#${ch.id}` },
        el("span", { class: "num", text: String(ch.order).padStart(2, "0") }),
        el("span", { "data-ring": ch.id }),
        el("span", { class: "t", text: ch.title }),
        el("span", { class: "meta", "data-meta": ch.id }))));
      const sec = el("section", { class: "chapter", id: ch.id });
      sec.append(el("div", { class: "head" },
        el("div", { class: "kicker", text: `Chapter ${ch.order}` }),
        el("h2", { text: ch.title }),
        el("div", { class: "meta", text: ch.subtitle || "" })));
      const body = el("div", { class: "body" });
      sec.append(body);
      main.append(sec);
      try {
        ch.render(body, lib);
      } catch (e) {
        console.error(`chapter ${ch.id} failed to render`, e);
        body.append(el("div", { class: "warn", text: `This chapter failed to load: ${e.message}` }));
      }
      body.append(el("div", { class: "chapter-done", "data-done": ch.id, text: "✓ Chapter complete" }));
      typeset(body);
    });
    progressEls = { fill: document.querySelector(".progress .fill"), stats: document.querySelector(".progress .stats") };
    document.getElementById("reset").addEventListener("click", () => {
      if (!confirm("Erase all saved progress on this device?")) return;
      store.solved = {}; store.time = {}; store.seeds = {};
      save();
      location.reload();
    });
    refreshProgress();
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  }

  const lib = {
    el, tex, typeset, theme, figure, canvas, drag, loop, controls, slider, button, toggle, readout,
    draw, plot, viridisish, gray, makeImage, loadImage, drawGray, la, rng, fmt, quiz,
  };
  window.DTAM = { chapter, lib, build, _quizzes: quizzes, _store: store };
})();
