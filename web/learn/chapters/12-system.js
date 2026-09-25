// Chapter 12: Putting it together: the live system (slam.rs state machine,
// keyframe management §2.4, λ and depth range §2.2.6, re-solving, display,
// AR, and why everything runs on the GPU).
(() => {
  "use strict";
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const DEG = Math.PI / 180;

  // ------------------------------------------------------------ flatland camera
  // A 2D world (x, y) seen by 1D cameras: heading h, forward (cos h, sin h),
  // "right" (sin h, -cos h). A camera sees N pixels; pixel u looks along
  // forward + ((u - c) / f) · right, so the ray parameter is the depth z.
  const makeImg = (n, fovDeg) => ({ n, c: (n - 1) / 2, f: n / 2 / Math.tan((fovDeg / 2) * DEG) });
  const cam = (x, y, h) => ({ x, y, h, c: Math.cos(h), s: Math.sin(h) });
  const rayDir = (p, img, u) => { const k = (u - img.c) / img.f; return [p.c + k * p.s, p.s - k * p.c]; };
  const toLocal = (p, P) => { const dx = P[0] - p.x, dy = P[1] - p.y; return [dx * p.s - dy * p.c, dx * p.c + dy * p.s]; };
  function hitT(o, d, A, B) {
    const ex = B[0] - A[0], ey = B[1] - A[1];
    const den = d[0] * ey - d[1] * ex;
    if (Math.abs(den) < 1e-12) return Infinity;
    const ax = A[0] - o[0], ay = A[1] - o[1];
    const t = (ax * ey - ay * ex) / den, s = (ax * d[1] - ay * d[0]) / den;
    return t > 1e-6 && s >= -1e-9 && s <= 1 + 1e-9 ? t : Infinity;
  }
  /** Nearest hit along a ray: {t, seg}. */
  function cast(segs, o, d) {
    let best = Infinity, seg = -1;
    segs.forEach((sg, i) => { const t = hitT(o, d, sg[0], sg[1]); if (t < best) { best = t; seg = i; } });
    return { t: best, seg };
  }
  const boxSegs = (x0, y0, x1, y1, tag) => [[[x0, y0], [x1, y0], tag], [[x1, y0], [x1, y1], tag], [[x1, y1], [x0, y1], tag], [[x0, y1], [x0, y0], tag]];
  const MIN_COS = Math.cos(85 * DEG);
  /** A keyframe: the camera's exact depth "scan", its 3D points and which mesh edges survive oblique culling. */
  function makeKeyframe(p, img, segs, maxRange = 40) {
    const z = new Float32Array(img.n), pts = [];
    for (let u = 0; u < img.n; u++) {
      const d = rayDir(p, img, u), h = cast(segs, [p.x, p.y], d);
      if (h.t < maxRange) { z[u] = h.t; pts.push([p.x + d[0] * h.t, p.y + d[1] * h.t]); } else pts.push(null);
    }
    const keep = [];
    for (let j = 0; j + 1 < img.n; j++) {
      const A = pts[j], B = pts[j + 1];
      if (!A || !B) { keep.push(false); continue; }
      const n = [-(B[1] - A[1]), B[0] - A[0]], m = [(A[0] + B[0]) / 2 - p.x, (A[1] + B[1]) / 2 - p.y];
      const cs = Math.abs(n[0] * m[0] + n[1] * m[1]) / (Math.hypot(...n) * Math.hypot(...m) || 1);
      keep.push(cs >= MIN_COS);
    }
    return { pose: p, z, pts, keep };
  }
  /** Renders keyframe meshes into camera p: predicted inverse depth per pixel (0 = no model), z-buffered. */
  function predict(kfs, p, img) {
    const xi = new Float32Array(img.n), own = new Int16Array(img.n).fill(-1);
    kfs.forEach((kf, k) => {
      for (let j = 0; j + 1 < img.n; j++) {
        if (!kf.keep[j]) continue;
        const a = toLocal(p, kf.pts[j]), b = toLocal(p, kf.pts[j + 1]);
        if (a[1] < 0.05 || b[1] < 0.05) continue;
        let ua = (img.f * a[0]) / a[1] + img.c, ub = (img.f * b[0]) / b[1] + img.c, qa = 1 / a[1], qb = 1 / b[1];
        if (ua > ub) { [ua, ub] = [ub, ua]; [qa, qb] = [qb, qa]; }
        const i0 = Math.max(0, Math.ceil(ua - 1e-6)), i1 = Math.min(img.n - 1, Math.floor(ub + 1e-6));
        for (let i = i0; i <= i1; i++) {
          const q = ub - ua > 1e-9 ? qa + ((i - ua) / (ub - ua)) * (qb - qa) : Math.max(qa, qb);
          if (q > xi[i]) { xi[i] = q; own[i] = k; }
        }
      }
    });
    let n = 0;
    for (let i = 0; i < img.n; i++) if (xi[i] > 0) n++;
    return { xi, own, cov: n / img.n };
  }
  /** Percentile like slam.rs: sorted[(len-1)·p] (truncated). */
  const pct = (sorted, p) => sorted[Math.floor((sorted.length - 1) * p)];
  /** New keyframe's inverse-depth range and λ from predicted inverse depths (slam.rs). */
  function rangeLambda(p2, p50, p98, first) {
    if (first) return { lo: p2 * 0.5, hi: p98 * 1.6, lambda: 1, dmin: 1 / p98 };
    const spread = 6; // sqrt(max_depth_ratio = 36)
    const lo = Math.max(p2 * 0.5, p50 / spread), hi = Math.min(p98 * 1.6, p50 * spread);
    const dmin = 1.6 / hi;
    return { lo, hi, dmin, lambda: 1 / (1 + 0.5 * dmin) };
  }

  function render(root, L) {
    // ================================================================ intro + loop
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>The previous chapters built the parts. This one wires them into the program that runs on a live video: which part runs when, when to start a new keyframe, how its parameters are chosen, and how it all fits on a GPU at video rate.</p>
      <div class="key">Tracking and mapping feed each other every frame: each tracked frame is averaged into the current keyframe's cost volume, and the solved keyframes are the model the next frame is tracked against.</div>
      <h3>The whole loop</h3>
      <p><code>Slam::push</code> in <code>slam.rs</code>, one call per video frame:</p>
<pre><code>push(frame f):
  small ← downsample f until the long side ≤ 512 (mapping resolution)
  match phase:
    Bootstrapping:
      track KLT corners (ch. 6); add to the window
      if window has 60 frames:
        estimate focal length, bootstrap poses + points (ch. 7)
        if it failed: drop the oldest 15 frames, keep collecting
        else:
          r ← middle frame of the window
          ξ range ← 2nd/98th percentile of the points' inverse depths × (0.5, 1.6)
          keyframe 0 ← cost volume from all window frames, λ = 1, solve (ch. 8–10)
          phase ← Dense
    Dense or Lost:
      pose, stats, prediction ← track(small)            # ch. 11
      if tracked: failures ← 0 (Lost → Dense)
      else: failures += 1; if failures ≥ 15: phase ← Lost
      if tracked and stats.used ≥ 0.8:                  # mapping
        add small to the active keyframe's cost volume
        if 20 frames added since the last solve: re-solve it
      if tracked and f − last_keyframe ≥ 8
         and prediction.coverage < 0.92 and #keyframes < max:
        finish (re-solve) the active keyframe if it has new frames
        start a keyframe at f: ξ range and λ from the prediction
        seed it with recent tracked frames close to f; solve once
        last_keyframe ← f
</code></pre>
      <h3>Three states</h3>
    `);

    // ================================================================ widget 1: state machine
    {
      const S = { phase: "boot", window: 0, failures: 0, frame: 0, parallax: false, kfs: 0, log: "Collecting KLT tracks for the bootstrap." };
      const fig = L.figure(root, "<b>The state machine.</b> Feed it frames. While bootstrapping, frames only collect corner tracks; whether the bootstrap succeeds depends on the camera having moved sideways. Once dense, count how many failed frames in a row it takes to get lost, and what brings it back.");
      const c = L.canvas(fig.el, { aspect: 0.46, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ro = L.readout(fig.el); fig.add(ro.el);
      const info = () => {
        const ph = { boot: "Bootstrapping", dense: "Dense", lost: "Lost" }[S.phase];
        ro.html = `frame ${S.frame} · phase <b>${ph}</b> · window ${S.window}/60 · consecutive failures <b>${S.failures}</b> · keyframes ${S.kfs}<br>${S.log}`;
        c.redraw();
      };
      const frame = (ok) => {
        S.frame++;
        if (S.phase === "boot") {
          S.window++;
          S.log = `Frame ${S.frame}: KLT tracks collected (tracking success does not apply yet).`;
          if (S.window >= 60) {
            if (S.parallax) {
              S.phase = "dense"; S.kfs = 1; S.failures = 0;
              S.log = "Bootstrap succeeded: poses + 3D points for the 60-frame window, keyframe 0 built from them with λ = 1. Dense tracking starts.";
            } else {
              S.window -= 15;
              S.log = "Bootstrap failed (not enough sideways motion): the oldest 15 frames are dropped and collection continues.";
            }
          }
        } else if (ok) {
          const was = S.phase;
          S.failures = 0;
          S.phase = "dense";
          S.log = was === "lost" ? `Frame ${S.frame} tracked: <b>recovered</b>, back to Dense.` : `Frame ${S.frame} tracked: pose from dense alignment; the frame is added to the active keyframe.`;
        } else {
          S.failures++;
          S.log = `Frame ${S.frame} failed: it gets the motion-model pose and is not used for mapping.`;
          if (S.failures >= 15 && S.phase === "dense") { S.phase = "lost"; S.log += " 15 in a row: <b>Lost</b>. It keeps trying from the last good pose."; }
        }
        info();
      };
      L.button(ctl, "Next frame: tracks", () => frame(true), "btn primary");
      L.button(ctl, "Next frame: fails", () => frame(false));
      L.button(ctl, "+10 failing frames", () => { for (let i = 0; i < 10; i++) frame(false); });
      L.toggle(ctl, "enough sideways motion", S.parallax, (v) => { S.parallax = v; });
      L.button(ctl, "Reset", () => { Object.assign(S, { phase: "boot", window: 0, failures: 0, frame: 0, kfs: 0, log: "Collecting KLT tracks for the bootstrap." }); info(); });
      const ctl2 = L.controls(fig.el); fig.add(ctl2);
      L.button(ctl2, "+10 frames", () => { for (let i = 0; i < 10; i++) frame(true); });
      c.draw = (ctx) => {
        const t = L.theme();
        const bw = Math.min(130, c.w * 0.27), bh = 40, y = c.h * 0.5;
        const X = { boot: c.w * 0.16, dense: c.w * 0.52, lost: c.w * 0.85 };
        const box = (k, label, sub) => {
          const on = S.phase === k, x = X[k];
          ctx.save();
          ctx.fillStyle = on ? t.accent : t.panel;
          ctx.strokeStyle = t.accent; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.roundRect(x - bw / 2, y - bh / 2, bw, bh, 9); ctx.fill(); ctx.stroke();
          ctx.restore();
          let size = 14;
          ctx.font = `600 ${size}px ${t.font}`;
          while (size > 10 && ctx.measureText(label).width > bw - 10) ctx.font = `600 ${--size}px ${t.font}`;
          L.draw.text(ctx, label, x, y + 5, on ? "#fff" : t.fg, { size, align: "center", bold: true });
          L.draw.text(ctx, sub, x, y + bh / 2 + 16, t.muted, { size: 11, align: "center" });
        };
        const arc = (x0, x1, up, label) => {
          const yy = up ? y - bh / 2 : y + bh / 2, cy = up ? y - bh / 2 - 38 : y + bh / 2 + 38;
          ctx.save();
          ctx.strokeStyle = t.muted; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(x0, yy); ctx.quadraticCurveTo((x0 + x1) / 2, cy, x1, yy); ctx.stroke();
          ctx.restore();
          const ang = Math.atan2(yy - (cy + yy) / 2, x1 - (x0 + x1) / 2);
          L.draw.arrow(ctx, x1 - Math.cos(ang) * 8, yy - Math.sin(ang) * 8, x1, yy, t.muted, 1.6, 8);
          L.draw.text(ctx, label, (x0 + x1) / 2, up ? cy + 10 : cy - 3, t.muted, { size: 11, align: "center" });
        };
        L.draw.arrow(ctx, X.boot + bw / 2, y, X.dense - bw / 2, y, t.muted, 1.6, 8);
        L.draw.text(ctx, "bootstrap OK", (X.boot + X.dense) / 2, y - bh / 2 - 8, t.muted, { size: 11, align: "center" });
        arc(X.dense + bw * 0.25, X.lost - bw * 0.25, true, "15 fails in a row");
        arc(X.lost - bw * 0.25, X.dense + bw * 0.25, false, "a frame tracks");
        box("boot", "Bootstrapping", `window ${S.window}/60`);
        box("dense", "Dense", `failures ${S.failures}`);
        box("lost", "Lost", S.phase === "lost" ? `failures ${S.failures}` : "");
      };
      info();
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <ul>
        <li><b>Bootstrapping</b> (paper §2.4: "a standard point feature based stereo method"): KLT + two-view geometry until the first keyframe exists. Retried every 15 frames until the motion allows it.</li>
        <li><b>Dense</b>: every frame is tracked against the model, well-tracked frames are mapped.</li>
        <li><b>Lost</b>: after 15 consecutive failed frames. It still tries every frame from the last good pose and returns to Dense as soon as one frame tracks. There is no relocaliser: the user points the camera back at the mapped area, or resets.</li>
      </ul>

      <h3>When to add a keyframe (§2.4)</h3>
      <p>The paper's rule is dense and simple: <i>add a keyframe when too much of the predicted image has no surface</i>. The prediction of chapter 11 already counts this: coverage = fraction of pixels with $\xi_v>0$.</p>
      <div class="eq-card"><div class="eq-label">New-keyframe rule (slam.rs)</div>
      $$\text{coverage} < 0.92\ \ \wedge\ \ f - f_\text{last KF} \ge 8\ \ \wedge\ \ \text{frame tracked}\ \ \wedge\ \ \#\text{KF} < \text{max}$$
      <div class="parts">
        <span>coverage &lt; 0.92</span><span>more than 8% of the view is unexplained by the model (<code>new_keyframe_coverage</code>)</span>
        <span>≥ 8 frames</span><span>minimum spacing, so a slowly revealed region does not spawn a keyframe per frame (<code>min_keyframe_spacing</code>)</span>
        <span>tracked</span><span>a keyframe's pose must be trustworthy: it anchors a whole depth map</span>
        <span>max</span><span>memory bound: 64 by default, 10 (phone) or 32 (desktop) in the web viewer</span>
      </div></div>
      <p>Why this is "better founded" than feature-system heuristics (distance travelled, number of features): it directly measures what matters, how much of what the camera sees the model can explain.</p>
    `);

    // ================================================================ widget 2: keyframe simulation
    {
      const img = makeImg(64, 70);
      const segs = [
        [[0, 0], [12, 0], 0], [[12, 0], [12, 8], 0], [[12, 8], [0, 8], 0], [[0, 8], [0, 0], 0],
        ...boxSegs(5.2, 3.3, 6.8, 4.7, 1), ...boxSegs(10.2, 0.7, 11.2, 1.7, 1), ...boxSegs(8.4, 6.6, 9.0, 7.2, 1),
        [[2, 8], [2, 6.3], 1],
      ];
      const PERIOD = 900;
      const poseAt = (f) => {
        const a = (f / PERIOD) * 2 * Math.PI;
        return cam(6 + 3.6 * Math.sin(a), 1.3 + 0.7 * Math.sin(2 * a), Math.PI / 2 - 0.5 * Math.sin(a + 0.8));
      };
      const S = { thr: 0.92, spacing: 8, running: true, speed: 1, acc: 0 };
      let st;
      const reset = () => {
        const p = poseAt(0), kf = makeKeyframe(p, img, segs);
        const zs = Array.from(kf.z).filter((z) => z > 0).map((z) => 1 / z).sort((a, b) => a - b);
        const rl = rangeLambda(pct(zs, 0.02), pct(zs, 0.5), pct(zs, 0.98), true);
        st = { f: 0, kfs: [Object.assign(kf, { frame: 0, rl })], lastKf: 0, sinceSolve: 0, hist: [], pred: null };
        st.pred = predict(st.kfs, p, img);
        st.hist.push({ f: 0, cov: st.pred.cov, kf: true, solve: false });
      };
      const stepFrame = () => {
        st.f++;
        const p = poseAt(st.f);
        const pred = predict(st.kfs, p, img);
        st.pred = pred;
        const ev = { f: st.f, cov: pred.cov, kf: false, solve: false };
        // mapping: (in this idealised world every frame tracks well) add to the active keyframe
        st.sinceSolve++;
        if (st.sinceSolve >= 20) { st.sinceSolve = 0; ev.solve = true; }
        if (st.f - st.lastKf >= S.spacing && pred.cov < S.thr && st.kfs.length < 40) {
          const xs = Array.from(pred.xi).filter((v) => v > 0).sort((a, b) => a - b);
          const rl = xs.length ? rangeLambda(pct(xs, 0.02), pct(xs, 0.5), pct(xs, 0.98), false) : st.kfs[st.kfs.length - 1].rl;
          st.kfs.push(Object.assign(makeKeyframe(p, img, segs), { frame: st.f, rl }));
          st.lastKf = st.f; st.sinceSolve = 0; ev.kf = true;
        }
        st.hist.push(ev);
        if (st.hist.length > 400) st.hist.shift();
      };
      reset();
      const fig = L.figure(root, "<b>Keyframes appear where the model runs out.</b> A camera sweeps left and right across a room, seen from above. Green rays hit modelled surface, red rays see unmodelled space (new wall at the image edge, or wall uncovered from behind the box). When coverage drops below the threshold and enough frames have passed, a new keyframe (purple) is taken. On the second sweep none are needed. Try other thresholds and spacings.");
      const c = L.canvas(fig.el, { aspect: 0.98 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const playBtn = L.button(ctl, "Pause", () => { S.running = !S.running; playBtn.innerHTML = S.running ? "Pause" : "Play"; }, "btn primary");
      L.button(ctl, "Step", () => { stepFrame(); c.redraw(); info(); });
      L.button(ctl, "Reset", () => { reset(); c.redraw(); info(); });
      L.slider(ctl, { label: "speed", min: 0.25, max: 4, step: 0.25, value: 1, fmt: (v) => v + "×", oninput: (v) => { S.speed = v; } });
      const ctl2 = L.controls(fig.el); fig.add(ctl2);
      L.slider(ctl2, { label: "coverage threshold", min: 0.5, max: 0.99, step: 0.01, value: S.thr, oninput: (v) => { S.thr = v; c.redraw(); } });
      L.slider(ctl2, { label: "min spacing (frames)", min: 1, max: 40, step: 1, value: S.spacing, oninput: (v) => { S.spacing = v; } });
      const ro = L.readout(fig.el); fig.add(ro.el);
      const info = () => {
        const k = st.kfs[st.kfs.length - 1];
        ro.html = `frame ${st.f} (sweep ${Math.floor(st.f / PERIOD) + 1}) · coverage <b>${(st.pred.cov * 100).toFixed(0)}%</b> · keyframes <b>${st.kfs.length}</b> · newest (frame ${k.frame}): ξ ∈ [${k.rl.lo.toFixed(3)}, ${k.rl.hi.toFixed(3)}], λ = <b>${k.rl.lambda.toFixed(3)}</b>${k.frame === 0 ? " (first keyframe)" : ""}`;
      };
      info();
      L.loop(c, (_, dt) => {
        if (!S.running) return;
        S.acc += dt * 30 * S.speed;
        let n = 0;
        while (S.acc >= 1 && n < 6) { S.acc -= 1; n++; stepFrame(); }
        if (S.acc > 6) S.acc = 0;
        if (n) { c.redraw(); info(); }
      });
      c.draw = (ctx) => {
        const t = L.theme();
        const mapH = c.h * 0.7, pad = 6;
        const sc = Math.min((c.w - 2 * pad) / 12, (mapH - 2 * pad) / 8);
        const ox = (c.w - 12 * sc) / 2, oy = pad + 8 * sc;
        const P = (q) => [ox + q[0] * sc, oy - q[1] * sc];
        ctx.fillStyle = t.panel2;
        ctx.fillRect(ox, oy - 8 * sc, 12 * sc, 8 * sc);
        const p = poseAt(st.f);
        // rays
        for (let u = 0; u < img.n; u += 2) {
          const d = rayDir(p, img, u), h = cast(segs, [p.x, p.y], d);
          const e = P([p.x + d[0] * h.t, p.y + d[1] * h.t]);
          ctx.save(); ctx.globalAlpha = 0.45;
          L.draw.line(ctx, ...P([p.x, p.y]), ...e, st.pred.xi[u] > 0 ? t.good : t.bad, 1.2);
          ctx.restore();
        }
        // walls
        for (const sg of segs) L.draw.line(ctx, ...P(sg[0]), ...P(sg[1]), t.fg, 2.2);
        // keyframes: surface points + camera
        st.kfs.forEach((kf, k) => {
          const newest = k === st.kfs.length - 1;
          for (let j = 0; j + 1 < img.n; j++) if (kf.keep[j]) L.draw.line(ctx, ...P(kf.pts[j]), ...P(kf.pts[j + 1]), t.accent4, newest ? 3 : 1.6);
          const q = kf.pose, a = rayDir(q, img, 0), b = rayDir(q, img, img.n - 1), L0 = 0.45;
          L.draw.path(ctx, [P([q.x + a[0] * L0, q.y + a[1] * L0]), P([q.x, q.y]), P([q.x + b[0] * L0, q.y + b[1] * L0])], t.accent4, 1.5);
          L.draw.dot(ctx, ...P([q.x, q.y]), 2.5, t.accent4);
        });
        // live camera
        const a = rayDir(p, img, 0), b = rayDir(p, img, img.n - 1), L1 = 0.7;
        L.draw.path(ctx, [P([p.x + a[0] * L1, p.y + a[1] * L1]), P([p.x, p.y]), P([p.x + b[0] * L1, p.y + b[1] * L1])], t.accent, 2.5);
        L.draw.dot(ctx, ...P([p.x, p.y]), 4, t.accent);
        // coverage timeline
        const ty0 = mapH + 14, ty1 = c.h - 16, tx0 = 30, tx1 = c.w - 6;
        const span = 360, fEnd = Math.max(span, st.f), f0 = fEnd - span;
        const TX = (f) => tx0 + ((f - f0) / span) * (tx1 - tx0), TY = (v) => ty1 - ((clamp(v, 0.5, 1) - 0.5) / 0.5) * (ty1 - ty0);
        L.draw.line(ctx, tx0, ty1, tx1, ty1, t.line, 1);
        for (const v of [0.5, 0.75, 1]) L.draw.text(ctx, String(v), tx0 - 4, TY(v) + 4, t.faint, { size: 11, align: "right" });
        L.draw.line(ctx, tx0, TY(S.thr), tx1, TY(S.thr), t.bad, 1, [5, 4]);
        const pts = [];
        for (const h of st.hist) {
          if (h.f < f0) continue;
          if (h.kf) L.draw.line(ctx, TX(h.f), ty0, TX(h.f), ty1, t.accent4, 2);
          if (h.solve) L.draw.line(ctx, TX(h.f), ty1 - 5, TX(h.f), ty1, t.muted, 1.5);
          pts.push([TX(h.f), TY(h.cov)]);
        }
        L.draw.path(ctx, pts, t.accent, 2);
        L.draw.text(ctx, "coverage · purple: new keyframe · ticks: re-solve", tx0, ty0 - 3, t.muted, { size: 11 });
        L.draw.text(ctx, "frame " + st.f, tx1, c.h - 3, t.faint, { size: 11, align: "right" });
      };
    }

    // ================================================================ λ and range
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>(Idealised: here a keyframe's depth is perfect the moment it is taken. In the real system it is a regularised estimate that improves as frames are averaged in.)</p>

      <h3>Setting up a new keyframe: depth range and λ</h3>
      <p>A cost volume needs its inverse-depth range $[\xi_\text{min}, \xi_\text{max}]$ (chapter 8) and the solver needs the data weight $\lambda$ (chapter 9). The paper leaves the range open; this implementation takes both from the model prediction at the new keyframe's pose, using percentiles $p_2, p_{50}, p_{98}$ of the predicted inverse depths $\xi_v$ (robust to a few stray pixels):</p>
      <div class="eq-card"><div class="eq-label">Inverse-depth range (slam.rs)</div>
      $$\xi_\text{min} = \max\big(0.5\,p_2,\ p_{50}/6\big),\qquad \xi_\text{max} = \min\big(1.6\,p_{98},\ 6\,p_{50}\big)$$
      <div class="parts">
        <span>$0.5$, $1.6$</span><span>margins: the new view may see a bit further and nearer than the old model shows (<code>xi_margin</code>)</span>
        <span>$p_{50}/6$, $6p_{50}$</span><span>caps: the range spans at most a factor $36$ in depth (<code>max_depth_ratio</code>), so the S layers are not spread too thin</span>
        <span>first keyframe</span><span>no prediction yet: the bootstrap's 3D points give $p_2, p_{98}$; margins only, no caps</span>
      </div></div>
      <div class="eq-card"><div class="eq-label">Data-term weight (paper §2.2.6)</div>
      $$\lambda = \frac{1}{1+0.5\,\bar d},\qquad \lambda = 1\ \text{for the first keyframe}$$
      <div class="parts">
        <span>$\bar d$</span><span>minimum scene depth predicted by the model; here $\bar d = 1.6/\xi_\text{max}$, which is $1/p_{98}$ unless capped</span>
        <span>small $\bar d$</span><span>near scene: same camera motion gives large parallax, a strong data term, so trust it ($\lambda\to1$)</span>
        <span>large $\bar d$</span><span>far scene: little parallax, weak data term, so regularise more ($\lambda$ smaller)</span>
      </div></div>
      <div class="note"><b>Worked example.</b> Predicted inverse depths $p_2=0.2$, $p_{50}=0.5$, $p_{98}=1.25$ (depths 5 m, 2 m, 0.8 m).
      <ul>
        <li>$\xi_\text{min}=\max(0.1,\ 0.0833)=0.1$, $\xi_\text{max}=\min(2,\ 3)=2$.</li>
        <li>$\bar d = 1.6/2 = 0.8$ m (= $1/p_{98}$), so $\lambda = 1/(1+0.4) = 0.714$.</li>
      </ul></div>
    `);

    // ================================================================ widget 3: range + λ
    {
      const S = { near: 0.8, med: 2, far: 5, first: false };
      const fig = L.figure(root, "<b>Range and λ from the prediction.</b> Set the nearest (2%), median and farthest (98%) predicted depths. Top: the inverse-depth axis with the percentiles, margins, caps (dashed) and the chosen range (thick). Bottom: λ as a function of the nearest depth. Push the far depth out to see the cap kick in.");
      const c = L.canvas(fig.el, { aspect: 0.62, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ro = L.readout(fig.el);
      const sN = L.slider(ctl, { label: "nearest depth", min: 0.2, max: 6, step: 0.05, value: S.near, fmt: (v) => v.toFixed(2) + " m", oninput: (v) => { S.near = v; fix("near"); } });
      const sM = L.slider(ctl, { label: "median depth", min: 0.2, max: 20, step: 0.1, value: S.med, fmt: (v) => v.toFixed(1) + " m", oninput: (v) => { S.med = v; fix("med"); } });
      const sF = L.slider(ctl, { label: "farthest depth", min: 0.3, max: 60, step: 0.5, value: S.far, fmt: (v) => v.toFixed(1) + " m", oninput: (v) => { S.far = v; fix("far"); } });
      L.toggle(ctl, "first keyframe (from bootstrap points)", S.first, (v) => { S.first = v; upd(); });
      fig.add(ro.el);
      const fix = (who) => {
        if (who === "near") { S.med = Math.max(S.med, S.near); S.far = Math.max(S.far, S.med); }
        if (who === "med") { S.near = Math.min(S.near, S.med); S.far = Math.max(S.far, S.med); }
        if (who === "far") { S.med = Math.min(S.med, S.far); S.near = Math.min(S.near, S.med); }
        sN.value = S.near; sM.value = S.med; sF.value = S.far;
        upd();
      };
      let R;
      const upd = () => {
        R = rangeLambda(1 / S.far, 1 / S.med, 1 / S.near, S.first);
        ro.html = `ξ range <b>[${R.lo.toFixed(3)}, ${R.hi.toFixed(3)}]</b> = depths ${(1 / R.hi).toFixed(2)} m … ${(1 / R.lo).toFixed(2)} m (ratio ${(R.hi / R.lo).toFixed(1)}) · d̄ = ${R.dmin.toFixed(2)} m · λ = <b>${R.lambda.toFixed(3)}</b>${S.first ? " (first keyframe: fixed)" : ""}`;
        c.redraw();
      };
      upd();
      c.draw = (ctx) => {
        const t = L.theme();
        const p2 = 1 / S.far, p50 = 1 / S.med, p98 = 1 / S.near;
        const xmax = Math.max(1.6 * p98, 6 * p50, R.hi) * 1.08;
        const x0 = 16, x1 = c.w - 16, ay = c.h * 0.26;
        const X = (v) => x0 + (v / xmax) * (x1 - x0);
        L.draw.line(ctx, x0, ay, x1, ay, t.line, 1.5);
        L.draw.text(ctx, "inverse depth ξ →", x1, ay + 30, t.faint, { size: 11, align: "right" });
        L.draw.text(ctx, "0", x0, ay + 16, t.faint, { size: 11, align: "center" });
        if (!S.first) {
          for (const v of [p50 / 6, 6 * p50]) L.draw.line(ctx, X(v), ay - 26, X(v), ay + 8, t.bad, 1.2, [4, 3]);
          L.draw.text(ctx, "cap", X(6 * p50), ay - 30, t.bad, { size: 11, align: "center" });
        }
        ctx.save(); ctx.fillStyle = t.accent; ctx.globalAlpha = 0.85;
        ctx.fillRect(X(R.lo), ay - 5, Math.max(2, X(R.hi) - X(R.lo)), 10);
        ctx.restore();
        [[p2, "p₂"], [p50, "p₅₀"], [p98, "p₉₈"]].forEach(([v, n], i) => {
          L.draw.dot(ctx, X(v), ay - 16, 4.5, t.accent2);
          L.draw.text(ctx, n, X(v), ay - 24 - (i % 2) * 11, t.accent2, { size: 11, align: "center", bold: true });
        });
        L.draw.line(ctx, X(0.5 * p2), ay - 16, X(p2), ay - 16, t.accent2, 1, [2, 2]);
        L.draw.line(ctx, X(p98), ay - 16, X(1.6 * p98), ay - 16, t.accent2, 1, [2, 2]);
        L.draw.text(ctx, "ξmin", X(R.lo), ay + 20, t.accent, { size: 11, align: "center" });
        L.draw.text(ctx, "ξmax", X(R.hi), ay + 20, t.accent, { size: 11, align: "center" });
        // λ(d̄) plot
        const pl = L.plot(c, { x0: 0, x1: 8, y0: 0, y1: 1, pad: [c.h * 0.46, 12, 22, 34] });
        pl.axes(ctx, { xlabel: "d̄ (m)", ylabel: "λ", xticks: 8, yticks: 2 });
        const pts = [];
        for (let k = 0; k <= 80; k++) { const d = k / 10; pts.push([pl.X(d), pl.Y(1 / (1 + 0.5 * d))]); }
        L.draw.path(ctx, pts, t.accent, 2);
        const d = clamp(R.dmin, 0, 8);
        L.draw.dot(ctx, pl.X(d), pl.Y(R.lambda), 6, S.first ? t.accent3 : t.accent2);
      };
    }

    // ================================================================ which frames, re-solve schedule
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Which frames go into a keyframe's cost volume</h3>
      <p>The set $\mathcal I(r)$ of eq. (2) is built in two ways:</p>
      <ul>
        <li><b>Seeding.</b> When keyframe $r$ starts at frame $f$, the up to 30 previous frames that were tracked and are close to it are averaged in immediately, so it can be solved at once instead of waiting for new frames.</li>
        <li><b>Accumulating.</b> Afterwards every new frame is added if it tracked well: tracking succeeded <i>and</i> used at least 80% of the predicted pixels (<code>min_used_for_mapping</code>). A frame with a hand in view gets its pose from the other pixels but does not pollute the map. At most 250 frames per keyframe.</li>
      </ul>
      <div class="eq-card"><div class="eq-label">Seeding test for a past frame m (slam.rs)</div>
      $$\text{tracked}(m)\ \wedge\ f-m\le30\ \wedge\ \frac{\|\mathbf c_m-\mathbf c_r\|}{D}\le0.3\ \wedge\ \angle(R_{rm})\le15^\circ$$
      <div class="parts">
        <span>$\|\mathbf c_m-\mathbf c_r\|$</span><span>distance between the two camera centres (baseline)</span>
        <span>$D$</span><span>a typical scene depth from the new range, $2/(\xi_\text{min}/0.5+\xi_\text{max}/1.6)$: baseline is judged relative to depth</span>
        <span>$15^\circ$</span><span>the rotation between the two cameras must be small, so the views overlap</span>
      </div></div>
      <h3>Re-solving as frames arrive (§2.2.6)</h3>
      <p>The paper notes that solver iterations can be interleaved with cost-volume updates, so a surface is usable early. This implementation re-runs the full primal–dual solve of chapter 10 on a schedule:</p>
      <ul>
        <li>right after seeding: the first solve; the keyframe joins the tracking model immediately (<code>publish_after_frames = 0</code>);</li>
        <li>then after every 20 frames added (<code>resolve_every</code>); each solve replaces the keyframe's depth map in the model;</li>
        <li>when a new keyframe starts, the old one gets a last solve if it has unsolved frames, then it is frozen.</li>
      </ul>
      <p>Only the newest keyframe is ever being mapped; older ones are fixed parts of the model.</p>

      <h3>Displaying the model: many keyframes, one scene</h3>
      <p>There is no volumetric fusion: each keyframe stays its own triangle mesh (chapter 11's mesh), placed in the world by $T_{wr}$. Drawing all meshes with one z-buffer makes the nearest surface win wherever they overlap. The same holds for tracking: the prediction renders all keyframes into the virtual camera. For display only (<code>viz.rs</code>):</p>
      <ul>
        <li>a mesh vertex every 2nd pixel, after a 3×3 median of the depth map (removes single-pixel spikes; the tracking model is untouched);</li>
        <li>a triangle is dropped only if it is oblique (beyond 80°) <i>and</i> spans a real depth jump (more than 2% per grid step), so noisy but continuous surfaces stay;</li>
        <li>textured with $I_r$, or shaded with a headlight like the paper's figures.</li>
      </ul>

      <h3>Augmented reality: a cube on the table</h3>
      <p>The viewer's AR cube is a live test of tracking: if the pose drifts or jitters, the cube slides. Anchoring (<code>ar.rs</code>), once the first keyframe exists:</p>
<pre><code>anchor(keyframe, pixel = image centre):
  march along the pixel's ray: t = 0.02 · 1.012^i
    until the point is behind the keyframe surface (its z ≥ 1/ξ at its pixel)
  bisect 30 times between the last two samples → hit point
  pts ← confident keyframe points (confidence ≥ 0.3) within 64 px of the hit
  plane ← RANSAC (300 tries, tolerance 1% of hit distance), refit to inliers
  normal ← points toward the camera
  cube: base on the plane under the ray, side = 0.12 × distance
every frame:
  draw the cube at the tracked pose T_wl;
  hide a cube pixel if the model's predicted depth there is nearer:
      z_cube > 1.03 · z_model    (z_model = 1/ξ_v from this frame's prediction)
</code></pre>
    `);

    // ================================================================ widget 6 (AR, side view)
    {
      const img = makeImg(64, 56);
      const segs = [[[-1, 0], [10, 0], 0], [[10, 0], [10, 4.2], 0], ...boxSegs(3.5, 0, 4.0, 0.9, 1), ...boxSegs(7.7, 0, 8.7, 0.3, 2)];
      const kpos = [2.0, 3.8], kaim = [6.3, 0];
      const kcam = cam(kpos[0], kpos[1], Math.atan2(kaim[1] - kpos[1], kaim[0] - kpos[0]));
      const kf = makeKeyframe(kcam, img, segs);
      const S = { tap: 32, live: [8.2, 2.3], cube: null };
      const anchor = () => {
        // march along the tapped keyframe ray until behind the keyframe surface, then bisect
        const d = rayDir(kcam, img, S.tap);
        const surfZ = (P) => { const l = toLocal(kcam, P); if (l[1] <= 1e-3) return null; const u = Math.round((img.f * l[0]) / l[1] + img.c); if (u < 0 || u >= img.n || !(kf.z[u] > 0)) return null; return { z: l[1], s: kf.z[u] }; };
        let prev = 0, hit = null;
        for (let i = 0; i < 600; i++) {
          const tt = 0.02 * 1.012 ** i, q = surfZ([kcam.x + d[0] * tt, kcam.y + d[1] * tt]);
          if (q && q.z >= q.s) {
            let a = prev, b = tt;
            for (let k = 0; k < 30; k++) { const m = (a + b) / 2, r = surfZ([kcam.x + d[0] * m, kcam.y + d[1] * m]); if (r && r.z >= r.s) b = m; else a = m; }
            hit = { t: b, P: [kcam.x + d[0] * b, kcam.y + d[1] * b] };
            break;
          }
          prev = tt;
        }
        if (!hit) { S.cube = null; return; }
        // RANSAC line through nearby model points, then least-squares refit
        const pts = kf.pts.filter((P) => P && Math.hypot(P[0] - hit.P[0], P[1] - hit.P[1]) < 1.5);
        const tol = 0.01 * hit.t;
        let best = [];
        const rr = L.rng(11);
        for (let it = 0; it < 300 && pts.length >= 2; it++) {
          const A = pts[Math.floor(rr.next() * pts.length)], B = pts[Math.floor(rr.next() * pts.length)];
          const e = [B[0] - A[0], B[1] - A[1]], ln = Math.hypot(...e);
          if (ln < 1e-9) continue;
          const n = [-e[1] / ln, e[0] / ln];
          const inl = pts.filter((P) => Math.abs(n[0] * (P[0] - A[0]) + n[1] * (P[1] - A[1])) < tol);
          if (inl.length > best.length) best = inl;
        }
        if (best.length < 3) { S.cube = null; return; }
        const mx = best.reduce((s, P) => s + P[0], 0) / best.length, my = best.reduce((s, P) => s + P[1], 0) / best.length;
        let sxx = 0, syy = 0, sxy = 0;
        for (const P of best) { sxx += (P[0] - mx) ** 2; syy += (P[1] - my) ** 2; sxy += (P[0] - mx) * (P[1] - my); }
        const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);
        const tdir = [Math.cos(ang), Math.sin(ang)];
        let n = [-tdir[1], tdir[0]];
        if (n[0] * (kcam.x - mx) + n[1] * (kcam.y - my) < 0) n = [-n[0], -n[1]];
        // base where the ray meets the fitted line
        const den = n[0] * d[0] + n[1] * d[1];
        const tb = Math.abs(den) > 1e-6 ? (n[0] * (mx - kcam.x) + n[1] * (my - kcam.y)) / den : hit.t;
        const base = [kcam.x + d[0] * tb, kcam.y + d[1] * tb];
        const size = 0.12 * Math.hypot(base[0] - kcam.x, base[1] - kcam.y);
        const h = size / 2;
        const c0 = [base[0] - tdir[0] * h, base[1] - tdir[1] * h], c1 = [base[0] + tdir[0] * h, base[1] + tdir[1] * h];
        const c2 = [c1[0] + n[0] * size, c1[1] + n[1] * size], c3 = [c0[0] + n[0] * size, c0[1] + n[1] * size];
        S.cube = { corners: [c0, c1, c2, c3], segs: [[c0, c1], [c1, c2], [c2, c3], [c3, c0]], hit: hit.P, inliers: best.length, size };
      };
      anchor();
      let view = null;
      const compute = () => {
        const cc = S.cube ? S.cube.corners.reduce((s, P) => [s[0] + P[0] / 4, s[1] + P[1] / 4], [0, 0]) : [6, 0.3];
        const lc = cam(S.live[0], S.live[1], Math.atan2(cc[1] - S.live[1], cc[0] - S.live[0]));
        const pred = predict([kf], lc, img);
        const px = [];
        let cubeN = 0, hidden = 0;
        for (let u = 0; u < img.n; u++) {
          const d = rayDir(lc, img, u), tr = cast(segs, [lc.x, lc.y], d);
          let zc = Infinity;
          if (S.cube) for (const sg of S.cube.segs) zc = Math.min(zc, hitT([lc.x, lc.y], d, sg[0], sg[1]));
          const zm = pred.xi[u] > 0 ? 1 / pred.xi[u] : Infinity;
          let kind = tr.seg >= 0 ? segs[tr.seg][2] : -1, show = false;
          if (zc < Infinity) { cubeN++; if (zc > 1.03 * zm) hidden++; else show = true; }
          px.push({ ztrue: tr.t, kind, show, hasModel: pred.xi[u] > 0, zc, zm });
        }
        view = { lc, px, cubeN, hidden, cov: pred.cov };
      };
      const fig = L.figure(root, "<b>AR occlusion, seen from the side.</b> The purple camera is the keyframe that mapped the table, the cup and the book; the cube is anchored where its tapped ray meets the table. Drag the blue live camera: the strips show what it sees and where the model has surface. Put the cup between camera and cube (drag low and to the left): the model hides the cube. The lower strip shows where this view has model surface; only there can anything hide the cube.");
      const c = L.canvas(fig.el, { aspect: 0.66 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      L.slider(ctl, { label: "tap pixel (keyframe)", min: 8, max: 58, step: 1, value: S.tap, oninput: (v) => { S.tap = v; anchor(); compute(); info(); c.redraw(); } });
      const ro = L.readout(fig.el); fig.add(ro.el);
      const info = () => {
        if (!view) compute();
        ro.html = S.cube
          ? `cube side ${S.cube.size.toFixed(2)} (0.12 × distance) · plane inliers ${S.cube.inliers} · live view: model coverage ${(view.cov * 100).toFixed(0)}% · cube pixels hidden <b>${view.hidden}/${view.cubeN}</b>`
          : "the tapped ray did not hit the model";
      };
      const geo = () => {
        const wx0 = -1.2, wx1 = 10.4, wy0 = -0.3, wy1 = 4.3;
        const sceneH = c.h * 0.64;
        const sc = Math.min((c.w - 8) / (wx1 - wx0), sceneH / (wy1 - wy0));
        const ox = (c.w - (wx1 - wx0) * sc) / 2 - wx0 * sc, oy = 4 + wy1 * sc;
        return { sc, ox, oy, sceneH, P: (q) => [ox + q[0] * sc, oy - q[1] * sc], inv: (x, y) => [(x - ox) / sc, (oy - y) / sc] };
      };
      L.drag(c, () => { const g = geo(); const p = g.P(S.live); return [{ x: p[0], y: p[1] }]; }, (_, p) => {
        const g = geo(), w = g.inv(p.x, p.y);
        S.live = [clamp(w[0], -0.9, 9.7), clamp(w[1], 0.4, 4.1)];
        compute(); info();
      });
      compute(); info();
      c.draw = (ctx) => {
        const t = L.theme();
        if (!view) compute();
        const g = geo(), P = g.P;
        // keyframe rays (faint) and its mesh
        for (let u = 0; u < img.n; u += 4) {
          if (!kf.pts[u]) continue;
          ctx.save(); ctx.globalAlpha = 0.25; L.draw.line(ctx, ...P([kcam.x, kcam.y]), ...P(kf.pts[u]), t.accent4, 1); ctx.restore();
        }
        const col = [t.fg, t.accent2, t.muted];
        for (const sg of segs) L.draw.line(ctx, ...P(sg[0]), ...P(sg[1]), col[sg[2]], 2.2);
        for (let j = 0; j + 1 < img.n; j++) if (kf.keep[j]) L.draw.line(ctx, ...P(kf.pts[j]), ...P(kf.pts[j + 1]), t.accent4, 3.2);
        L.draw.dot(ctx, ...P([kcam.x, kcam.y]), 6, t.accent4);
        if (S.cube) {
          const cp = S.cube.corners.map(P);
          ctx.save(); ctx.fillStyle = t.accent3; ctx.globalAlpha = 0.8;
          ctx.beginPath(); cp.forEach((q, i) => (i ? ctx.lineTo(...q) : ctx.moveTo(...q))); ctx.closePath(); ctx.fill();
          ctx.restore();
          L.draw.dot(ctx, ...P(S.cube.hit), 3, t.fg);
          L.draw.line(ctx, ...P([kcam.x, kcam.y]), ...P(S.cube.hit), t.accent4, 1.2, [4, 3]);
        }
        // live camera frustum
        const lc = view.lc, a = rayDir(lc, img, 0), b = rayDir(lc, img, img.n - 1), L1 = 0.9;
        L.draw.path(ctx, [P([lc.x + a[0] * L1, lc.y + a[1] * L1]), P([lc.x, lc.y]), P([lc.x + b[0] * L1, lc.y + b[1] * L1])], t.accent, 2);
        L.draw.handle(ctx, ...P(S.live), t.accent);
        // strips
        const sy = g.sceneH + 22, sh = Math.max(14, (c.h - sy - 30) / 2), sx0 = 8, sw = c.w - 16, pw = sw / img.n;
        L.draw.text(ctx, "live camera image (green: cube drawn)", sx0, sy - 5, t.muted, { size: 11 });
        view.px.forEach((q, u) => {
          let fill;
          if (q.show) fill = t.accent3;
          else if (q.kind === 1) fill = t.accent2;
          else if (q.kind === 2) fill = t.muted;
          else fill = t.faint;
          ctx.fillStyle = fill;
          ctx.fillRect(sx0 + u * pw, sy, pw + 0.5, sh);
          if (!q.show && q.zc < Infinity) { ctx.fillStyle = t.fg; ctx.fillRect(sx0 + u * pw, sy + sh - 3, pw + 0.5, 3); }
        });
        const sy2 = sy + sh + 20;
        L.draw.text(ctx, "model surface in this view (blue: none)", sx0, sy2 - 5, t.muted, { size: 11 });
        view.px.forEach((q, u) => {
          ctx.fillStyle = q.hasModel ? t.accent4 : t.accent;
          ctx.fillRect(sx0 + u * pw, sy2, pw + 0.5, sh * 0.6);
        });
      };
    }

    // ================================================================ GPU
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Why it runs on a GPU</h3>
      <p>Almost every step is the same small computation repeated for every pixel, independently: a cost-volume update (per pixel × layer), a primal–dual iteration (per pixel), a tracking residual (per pixel). A GPU runs thousands of these at once. In WGSL a <b>compute shader</b> is the per-pixel function; it runs in <b>workgroups</b> of threads (64 here) that can share a small fast memory.</p>
      <h4>Reductions: summing over all pixels</h4>
      <p>Gauss–Newton needs sums over the whole image ($H=\sum J^\top J$, $\mathbf g=\sum J^\top f$). Many threads cannot all add into one number at once, so the sum is a tree (<code>track_common.wgsl</code>):</p>
      <ol>
        <li>128 workgroups × 64 threads = 8192 threads. Thread $i$ handles pixels $i, i+8192, i+16384,\dots$ and keeps 36 running sums in registers: 21 for the upper triangle of $H$, 6 for $\mathbf g$, $\sum f^2$, 4 pixel counters (rejected, in view, used, with model), and 4 sums for gain/bias.</li>
        <li>Inside a workgroup the 64 partial sums are halved repeatedly: threads $i<s$ add slot $i+s$, for $s=32,16,\dots,1$: 6 steps.</li>
        <li>Each workgroup writes one row of 36 numbers; the solver kernel sums the 128 rows.</li>
      </ol>
    `);

    // ================================================================ widget 4: reduction tree
    {
      const N = 16;
      const S = { vals: [], step: 0 };
      const rr = L.rng(5);
      const fresh = () => { S.vals = Array.from({ length: N }, () => 1 + Math.floor(rr.next() * 6)); S.step = 0; };
      fresh();
      const rows = () => {
        const out = [S.vals.slice()];
        let cur = S.vals.slice();
        for (let s = N / 2; s >= 1; s /= 2) {
          const nx = cur.slice();
          for (let i = 0; i < s; i++) nx[i] = cur[i] + cur[i + s];
          out.push(nx);
          cur = nx;
        }
        return out;
      };
      const fig = L.figure(root, "<b>A tree reduction.</b> 16 threads hold partial sums. Press Step: in each step, every thread $i < s$ adds the value of thread $i+s$, then $s$ halves. After $\\log_2 16 = 4$ steps thread 0 holds the total.");
      const c = L.canvas(fig.el, { aspect: 0.5, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ro = L.readout(fig.el);
      L.button(ctl, "Step", () => { S.step = Math.min(4, S.step + 1); upd(); }, "btn primary");
      L.button(ctl, "New numbers", () => { fresh(); upd(); });
      fig.add(ro.el);
      const upd = () => {
        const R = rows(), total = S.vals.reduce((a, b) => a + b, 0);
        ro.html = S.step === 0
          ? `start: 16 partial sums, total should be ${total}`
          : `step ${S.step}: s = ${N >> S.step}: threads 0…${(N >> S.step) - 1} add slot i + ${N >> S.step} · thread 0 now holds <b>${R[S.step][0]}</b>${S.step === 4 ? ` = ${total} ✓` : ""}`;
        c.redraw();
      };
      upd();
      c.draw = (ctx) => {
        const t = L.theme();
        const R = rows();
        const pad = 6, bw = (c.w - 2 * pad) / N, bh = Math.min(26, (c.h - 10) / 5 - 12), gapY = (c.h - 5 * bh - 8) / 4;
        for (let k = 0; k <= S.step; k++) {
          const y = 4 + k * (bh + gapY), active = N >> k;
          if (k > 0) {
            const s = N >> k, yp = 4 + (k - 1) * (bh + gapY) + bh;
            for (let i = 0; i < s; i++) {
              L.draw.line(ctx, pad + (i + 0.5) * bw, yp, pad + (i + 0.5) * bw, y, t.faint, 1);
              L.draw.line(ctx, pad + (i + s + 0.5) * bw, yp, pad + (i + 0.5) * bw, y, t.accent2, 1);
            }
          }
          for (let i = 0; i < N; i++) {
            const on = i < active;
            ctx.save();
            ctx.fillStyle = on ? (k === S.step ? t.accent : t.panel2) : t.panel;
            ctx.strokeStyle = on ? t.accent : t.line;
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.roundRect(pad + i * bw + 1, y, bw - 2, bh, 4); ctx.fill(); ctx.stroke();
            ctx.restore();
            if (on) L.draw.text(ctx, String(R[k][i]), pad + (i + 0.5) * bw, y + bh / 2 + 4, k === S.step ? "#fff" : t.fg, { size: bw > 26 ? 12 : 11, align: "center", mono: true });
          }
        }
      };
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <h4>The real bottleneck: round trips between CPU and GPU</h4>
      <p>The CPU and GPU are separate processors with separate memory. The CPU queues work; to <i>see</i> a result (say $H$ and $\mathbf g$, to solve for $\psi$ and decide the next step), it must wait for the GPU to finish everything queued, copy the result into a readable buffer and map it (<code>read_buffers</code> in <code>gpu.rs</code>). During that wait the GPU sits idle, and while the CPU then solves and queues the next pass, the GPU is idle again.</p>
      <p>A straightforward tracker reads back once per Gauss–Newton iteration: with 20 rotation iterations and 20 + 20 + 15 + 10 pose iterations, that is up to about 85 round trips per frame, each costing far more than the few hundred microseconds of actual GPU work. This implementation's fix (<code>gn.wgsl</code>): <b>the solver also runs on the GPU</b>.</p>
      <ul>
        <li>A tiny "step" kernel (one workgroup) sums the 128 partial rows, applies the LM accept/reject rule, solves the 6×6 system by Cholesky, computes $\exp(\psi)$ and writes the next candidate pose into a GPU buffer.</li>
        <li>The next residual pass reads its pose from that buffer, so no CPU is involved between iterations.</li>
        <li>All levels and iterations are queued in one submission; when a level converges early the kernels see a <code>done</code> flag and do nothing.</li>
        <li>Per frame: one readback for the rotation, one for prediction + alignment together.</li>
      </ul>
    `);

    // ================================================================ widget 5: round-trip timeline
    {
      const S = { work: 0.15, lat: 1.5, pred: 0.4 };
      const NROT = 20, N6 = 65;
      const fig = L.figure(root, "<b>Where the time goes (a simple model).</b> Top: one readback per iteration. Bottom: the whole Gauss–Newton loop on the GPU, two readbacks per frame. Blue: GPU busy. Red: waiting for a round trip. Change the per-iteration GPU work and the round-trip latency.");
      const c = L.canvas(fig.el, { aspect: 0.42, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ro = L.readout(fig.el);
      L.slider(ctl, { label: "GPU work / iteration", min: 0.02, max: 1, step: 0.01, value: S.work, fmt: (v) => v.toFixed(2) + " ms", oninput: (v) => { S.work = v; upd(); } });
      L.slider(ctl, { label: "round-trip latency", min: 0.1, max: 5, step: 0.1, value: S.lat, fmt: (v) => v.toFixed(1) + " ms", oninput: (v) => { S.lat = v; upd(); } });
      fig.add(ro.el);
      const totals = () => {
        const n = NROT + N6;
        return { old: S.pred + S.lat + n * (S.work + S.lat), neu: S.pred + n * S.work + 2 * S.lat, n };
      };
      const upd = () => {
        const T = totals();
        ro.html = `per iteration readback: <b>${T.old.toFixed(1)} ms</b> (${(1000 / T.old).toFixed(0)} fps max, ${T.n + 1} round trips) · GPU-resident: <b>${T.neu.toFixed(1)} ms</b> (${(1000 / T.neu).toFixed(0)} fps max, 2 round trips)`;
        c.redraw();
      };
      upd();
      c.draw = (ctx) => {
        const t = L.theme();
        const T = totals();
        const x0 = 8, x1 = c.w - 8, X = (ms) => x0 + (ms / T.old) * (x1 - x0);
        const lane = (y, blocks, label, total) => {
          L.draw.text(ctx, `${label}: ${total.toFixed(1)} ms`, x0, y - 6, t.fg, { size: 12, bold: true });
          ctx.fillStyle = t.panel2; ctx.fillRect(x0, y, x1 - x0, 18);
          for (const [a, b, kind] of blocks) {
            ctx.fillStyle = kind === "gpu" ? t.accent : t.bad;
            ctx.fillRect(X(a), kind === "gpu" ? y : y + 7, Math.max(0.6, X(b) - X(a)), kind === "gpu" ? 18 : 4);
          }
        };
        const oldB = [];
        let tm = 0;
        oldB.push([tm, tm + S.pred, "gpu"]); tm += S.pred; oldB.push([tm, tm + S.lat, "wait"]); tm += S.lat;
        for (let i = 0; i < T.n; i++) { oldB.push([tm, tm + S.work, "gpu"]); tm += S.work; oldB.push([tm, tm + S.lat, "wait"]); tm += S.lat; }
        const newB = [];
        tm = 0;
        newB.push([tm, tm + NROT * S.work, "gpu"]); tm += NROT * S.work; newB.push([tm, tm + S.lat, "wait"]); tm += S.lat;
        newB.push([tm, tm + S.pred + N6 * S.work, "gpu"]); tm += S.pred + N6 * S.work; newB.push([tm, tm + S.lat, "wait"]);
        const h = c.h;
        lane(h * 0.3, oldB, "readback every iteration", T.old);
        lane(h * 0.72, newB, "loop on the GPU", T.neu);
        L.draw.text(ctx, "time within one frame →", x1, h - 4, t.faint, { size: 11, align: "right" });
      };
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <p>The same principle applies to mapping: the cost-volume update, the whole primal–dual loop and the θ schedule are queued on the GPU; results come back only when a keyframe is solved. The lesson generalises: <b>on a GPU, count round trips before counting flops.</b></p>
      <div class="note"><b>What you now have.</b> KLT + two-view bootstrap → keyframe 0 → per frame: rotation pre-alignment, prediction, robust coarse-to-fine alignment → map well-tracked frames into the active keyframe, re-solve every 20 → new keyframe when coverage &lt; 0.92. That is the whole of DTAM as implemented here. The final chapter tests it end to end.</div>
    `);

    // ================================================================ quiz
    const f3 = (v) => L.fmt(v, 3), f4 = (v) => L.fmt(v, 4);
    L.quiz(root, "system", [
      {
        id: "lambda", type: "num",
        gen: (r) => {
          const d = r.pick([0.3, 0.5, 0.8, 1, 1.5, 2, 3, 4, 6]);
          const lam = 1 / (1 + 0.5 * d);
          return {
            q: String.raw`The model predicts that the nearest surface in view of a new keyframe is $${d}$ m away. What data-term weight $\lambda$ does it get (3 decimals)?`,
            answer: lam, rtol: 0.002,
            explain: String.raw`$\lambda=1/(1+0.5\,\bar d)=1/(1+0.5\cdot${d})=${f4(lam)}$.`,
          };
        },
      },
      {
        id: "range", type: "num",
        gen: (r) => {
          const p50 = r.pick([0.25, 0.4, 0.5, 0.8, 1]);
          const p2 = +(p50 * r.pick([0.1, 0.3, 0.5, 0.7])).toFixed(4), p98 = +(p50 * r.pick([1.3, 2, 3, 5])).toFixed(4);
          const lo = Math.max(0.5 * p2, p50 / 6), hi = Math.min(1.6 * p98, 6 * p50);
          return {
            q: String.raw`At a new keyframe's pose the prediction's inverse depths have percentiles $p_2=${p2}$, $p_{50}=${p50}$, $p_{98}=${p98}$. Compute its range $[\xi_\text{min}, \xi_\text{max}]$ (4 decimals).`,
            answer: [lo, hi], labels: ["$\\xi_\\text{min}$", "$\\xi_\\text{max}$"], rtol: 0.002, tol: 1e-4,
            explain: String.raw`$\xi_\text{min}=\max(0.5\cdot${p2},\ ${p50}/6)=\max(${f4(0.5 * p2)}, ${f4(p50 / 6)})=${f4(lo)}$. $\xi_\text{max}=\min(1.6\cdot${p98},\ 6\cdot${p50})=\min(${f4(1.6 * p98)}, ${f4(6 * p50)})=${f4(hi)}$.`,
          };
        },
      },
      {
        id: "lambda2", type: "num",
        gen: (r) => {
          const p50 = r.pick([0.25, 0.4, 0.5, 1]);
          const p98 = +(p50 * r.pick([1.5, 2, 3, 4, 5])).toFixed(4);
          const hi = Math.min(1.6 * p98, 6 * p50), d = 1.6 / hi, lam = 1 / (1 + 0.5 * d);
          return {
            q: String.raw`A new (not the first) keyframe's prediction has $p_{50}=${p50}$ and $p_{98}=${p98}$. This implementation takes $\bar d=1.6/\xi_\text{max}$. What is $\lambda$ (3 decimals)?`,
            answer: lam, rtol: 0.002,
            explain: String.raw`$\xi_\text{max}=\min(1.6\cdot${p98},\ 6\cdot${p50})=${f4(hi)}$${1.6 * p98 > 6 * p50 ? " (capped)" : ""}. $\bar d=1.6/${f4(hi)}=${f4(d)}$ m. $\lambda=1/(1+0.5\cdot${f4(d)})=${f4(lam)}$.`,
          };
        },
      },
      {
        id: "covmax", type: "num",
        gen: (r) => {
          const [w, h] = r.pick([[256, 192], [256, 144], [192, 144], [224, 168], [208, 156]]);
          const n = Math.floor(0.08 * w * h);
          return {
            q: String.raw`The prediction image is $${w}\times${h}$ pixels. What is the largest number of pixels without predicted surface for which a new keyframe is <i>not</i> triggered by coverage (threshold 0.92)?`,
            answer: n, tol: 0.5,
            explain: String.raw`A keyframe is triggered when coverage $<0.92$, i.e. when more than $8\%$ of pixels are empty: $0.08\cdot${w * h}=${f3(0.08 * w * h)}$. The largest whole number not above that is $${n}$.`,
          };
        },
      },
      {
        id: "spacing", type: "num",
        gen: (r) => {
          const f0 = r.int(100, 400), dd = r.int(2, 14);
          const ans = Math.max(f0 + dd, f0 + 8);
          return {
            q: String.raw`The last keyframe was created at frame $${f0}$. From frame $${f0 + dd}$ on, coverage stays below 0.92 and every frame tracks well. At which frame is the next keyframe created?`,
            answer: ans, tol: 0.1,
            explain: String.raw`It needs coverage $<0.92$ (from frame ${f0 + dd}) and at least 8 frames since the last keyframe (from frame ${f0 + 8}). Both hold first at frame $${ans}$.`,
          };
        },
      },
      {
        id: "seed", type: "num",
        gen: (r) => {
          const D = r.pick([0.8, 1, 1.5, 2]), f = r.int(200, 500);
          const ks = r.shuffle([3, 8, 14, 22, 29, 35]).slice(0, 5).sort((a, b) => a - b);
          const rows = ks.map((k) => ({ k, tracked: r.next() < 0.8, dist: +(D * r.pick([0.05, 0.1, 0.2, 0.28, 0.35, 0.5])).toFixed(3), ang: r.pick([2, 5, 9, 14, 18, 25]) }));
          const ok = rows.filter((x) => x.tracked && x.k <= 30 && x.dist / D <= 0.3 && x.ang <= 15);
          const list = rows.map((x) => `frame ${f - x.k}: ${x.tracked ? "tracked" : "<b>not</b> tracked"}, ${x.dist} m from the new keyframe, rotated ${x.ang}°`).join("</li><li>");
          return {
            q: String.raw`A keyframe starts at frame $${f}$; its typical depth is $D=${D}$ m. How many of these past frames seed its cost volume?<ul><li>${list}</li></ul>`,
            answer: ok.length, tol: 0.1,
            explain: String.raw`A frame seeds if it tracked, is at most 30 frames old, has baseline $\le0.3D=${f3(0.3 * D)}$ m and rotation $\le15^\circ$. That holds for ${ok.length ? ok.map((x) => "frame " + (f - x.k)).join(", ") : "none of them"}: $${ok.length}$.`,
          };
        },
      },
      {
        id: "resolve", type: "num",
        gen: (r) => {
          const F = r.int(100, 300), a = r.int(3, 30), k = r.int(2, 6);
          // simulate: frames F+1.. are added unless in [F+a, F+a+k-1]
          let added = 0, since = 0, solves = 0, fr = F, ans = 0;
          while (solves < 2) {
            fr++;
            if (fr >= F + a && fr < F + a + k) continue;
            added++; since++;
            if (since >= 20) { since = 0; solves++; ans = fr; }
          }
          return {
            q: String.raw`A keyframe is started (seeded and solved) at frame $${F}$. Every later frame is added to its cost volume, except frames $${F + a}$ to $${F + a + k - 1}$, which tracked poorly (under 80% of pixels used). No other keyframe starts. At which frame does the <i>second</i> re-solve happen (re-solve every 20 added frames)?`,
            answer: ans, tol: 0.1,
            explain: String.raw`The keyframe needs 40 added frames. Without gaps that is frame ${F}+40; the ${k} skipped frames push it back by ${k}: frame $${ans}$.`,
          };
        },
      },
      {
        id: "lost", type: "num",
        gen: (r) => {
          const F = r.int(300, 900), j = r.pick([0, 0, 5, 9, 12, 14]);
          const ans = j ? F + j + 15 : F + 15;
          const q = j
            ? String.raw`Frame $${F}$ was tracked. Frames $${F + 1}$ to $${F + j - 1}$ fail, frame $${F + j}$ tracks, and every frame after that fails. At which frame does the system enter the Lost state?`
            : String.raw`Frame $${F}$ was tracked and every frame after it fails. At which frame does the system enter the Lost state?`;
          return {
            q, answer: ans, tol: 0.1,
            explain: j
              ? String.raw`The success at frame ${F + j} resets the counter to 0. The 15th consecutive failure after it is frame $${F + j}+15=${ans}$.`
              : String.raw`Lost after 15 consecutive failures: frames ${F + 1} … ${ans}, so frame $${ans}$.`,
          };
        },
      },
      {
        id: "threads", type: "num",
        gen: (r) => {
          const [w, h] = r.pick([[512, 384], [512, 288], [256, 192], [384, 288]]), L0 = r.pick([0, 1]);
          const wl = Math.ceil(w / 2 ** L0), hl = Math.ceil(h / 2 ** L0), n = wl * hl, per = Math.ceil(n / 8192);
          return {
            q: String.raw`Mapping resolution is $${w}\times${h}$. A tracking residual pass on pyramid level ${L0} runs 128 workgroups of 64 threads with a grid-stride loop. At most how many pixels does one thread process?`,
            answer: per, tol: 0.1,
            explain: String.raw`Level ${L0} is $${wl}\times${hl}=${n}$ pixels. Threads: $128\cdot64=8192$. $\lceil ${n}/8192\rceil=${per}$.`,
          };
        },
      },
      {
        id: "redsteps", type: "num",
        gen: (r) => {
          const k = r.pick([5, 6, 7, 8]), n = 2 ** k;
          return {
            q: String.raw`A workgroup of $${n}$ threads tree-reduces one accumulator as in the widget. How many halving steps does it take, and how many additions are performed in total?`,
            answer: [k, n - 1], labels: ["steps", "additions"], tol: 0.1,
            explain: String.raw`Steps: $\log_2 ${n}=${k}$. Additions: $${n / 2}+${n / 4}+\dots+1=${n - 1}$ (each addition removes one partial sum, from ${n} down to 1).`,
          };
        },
      },
      {
        id: "accum", type: "num",
        gen: (r) => {
          const n = r.pick([2, 3, 4, 6, 7, 8]);
          return {
            q: String.raw`A Gauss–Newton problem has $${n}$ parameters. How many distinct numbers must each pixel add to the sums $H=\sum J^\top J$ and $\mathbf g=\sum J^\top f$ together (use the symmetry of $H$)?`,
            answer: (n * (n + 1)) / 2 + n, tol: 0.1,
            explain: String.raw`$H$ is symmetric: $${n}\cdot${n + 1}/2=${(n * (n + 1)) / 2}$ distinct entries; $\mathbf g$ has ${n}. Total $${(n * (n + 1)) / 2 + n}$. (For 6DOF: 21 + 6 = 27 of the 36 accumulators.)`,
          };
        },
      },
      {
        id: "roundtrip", type: "num",
        gen: (r) => {
          const lat = r.pick([0.5, 0.8, 1, 1.5, 2, 3]), rot = r.pick([10, 20]), it = r.pick([[20, 20, 15, 10], [10, 10, 10, 5], [15, 15, 10, 10]]);
          const n = rot + it.reduce((a, b) => a + b, 0), saved = (n + 1 - 2) * lat;
          return {
            q: String.raw`Model: a tracker reads back once per Gauss–Newton iteration (${rot} rotation iterations, then ${it.join(" + ")} pose iterations) plus once for the prediction. The GPU-resident version reads back twice per frame. Each round trip costs $${lat}$ ms of waiting. How many ms per frame does the GPU-resident version save (worst case, no early exits)?`,
            answer: saved, rtol: 0.005,
            explain: String.raw`Round trips before: $${rot}+${it.join("+")}+1=${n + 1}$. After: 2. Saved: $${n - 1}\cdot${lat}=${f3(saved)}$ ms.`,
          };
        },
      },
      {
        id: "aroccl", type: "num",
        gen: (r) => {
          const xi = r.pick([0.4, 0.5, 0.8, 1.25, 2]);
          return {
            q: String.raw`At one pixel the model's prediction has inverse depth $\xi_v=${xi}$. What is the largest depth a cube pixel there can have and still be drawn (3 decimals)?`,
            answer: 1.03 / xi, rtol: 0.002,
            explain: String.raw`The model depth is $1/${xi}=${f4(1 / xi)}$. The cube is hidden when $z_\text{cube}>1.03\,z_\text{model}$, so it is drawn up to $1.03\cdot${f4(1 / xi)}=${f4(1.03 / xi)}$. The 3% slack stops the surface the cube stands on from hiding it.`,
          };
        },
      },
      {
        id: "failframe", type: "mc",
        q: "In the Dense state, a frame fails tracking (both attempts). What happens to it?",
        choices: [
          "It gets the motion-model pose, is not added to any cost volume, and increments the consecutive-failure counter",
          "It is added to the cost volume with a lower weight",
          "The system immediately switches to Lost and resets the map",
          "It becomes a new keyframe, since the model clearly does not explain it",
        ],
        answer: 0,
        explain: "Only well-tracked frames are mapped, and keyframes need a tracked pose. Lost only follows 15 failures in a row.",
      },
      {
        id: "whycov", type: "mc",
        q: "Why is 'coverage of the predicted image' a natural keyframe trigger for DTAM, compared to feature-based rules like 'distance travelled'?",
        choices: [
          "The dense prediction directly measures how much of the current view the model can explain, which is exactly when new geometry is needed",
          "Coverage is cheaper to compute than distance travelled",
          "Coverage never changes when the camera only rotates",
          "Distance travelled is unknown in a monocular system",
        ],
        answer: 0,
        explain: "Rotating or moving towards unmapped space both reduce coverage; hovering over mapped space does not, however far you move. (Distance is known up to scale, so that distractor is wrong.)",
      },
      {
        id: "lambdawhy", type: "mc",
        q: "Why does a keyframe of a far-away scene get a smaller $\\lambda$?",
        choices: [
          "For similar camera motion a far scene produces less parallax, so the data term is less informative and more regularisation is wanted",
          "Far pixels are darker, so photometric errors are smaller",
          "The cost volume has fewer layers for far scenes",
          "A smaller λ makes the solver converge in fewer iterations",
        ],
        answer: 0,
        explain: "λ multiplies the data term in eq. (6). Weak data (little parallax) → trust smoothness more → smaller λ.",
      },
      {
        id: "fusion", type: "mc",
        q: "How are several keyframes combined into one model, for display and for tracking?",
        choices: [
          "Each stays a separate mesh in world coordinates; rendering them all with one z-buffer lets the nearest surface win",
          "Their depth maps are averaged into a voxel grid",
          "Only the newest keyframe is used; older ones are discarded",
          "Their cost volumes are merged into one before solving",
        ],
        answer: 0,
        explain: "No volumetric fusion. Both the viewer and the tracker's prediction draw every keyframe mesh; overlaps are resolved by depth.",
      },
      {
        id: "gpuwhy", type: "multi",
        q: "Why is running the whole Gauss–Newton loop on the GPU faster than solving each step on the CPU? (select all)",
        choices: [
          "Each GPU→CPU readback forces the CPU to wait for all queued GPU work and a buffer map; per-iteration readbacks multiply that latency",
          "While the CPU waits for or processes a readback, the GPU sits idle",
          "With the solver on the GPU, a whole frame's alignment is one submission and one readback",
          "A 6×6 Cholesky solve is too hard for a CPU",
          "GPUs compute square roots more accurately",
        ],
        answer: [0, 1, 2],
        explain: "The 6×6 solve is trivial anywhere; what costs is the synchronisation. Removing it keeps the GPU busy back to back.",
      },
    ]);
  }

  DTAM.chapter({
    id: "system",
    order: 12,
    title: "Putting it together: the live system",
    subtitle: "State machine, keyframes, λ, re-solving, display, AR and the GPU",
    minutes: 45,
    render,
  });
})();
