// Chapter 13: cumulative final exam + implementation checklist.
DTAM.chapter({
  id: "exam",
  order: 13,
  title: "Final exam and implementation checklist",
  subtitle: "Every step of the pipeline by hand, then the one-page build plan",
  minutes: 90,
  render(root, L) {
    const fm = (v, d = 3) => L.fmt(v, d);
    // wrap negatives in parentheses for products / sums in worked solutions
    const P = (v, d = 3) => (v < 0 ? `(${fm(v, d)})` : fm(v, d));
    const vec = (a, d = 3) => `(${a.map((v) => fm(v, d)).join(",\\ ")})`;
    const deg = Math.PI / 180;
    const la = L.la;

    root.insertAdjacentHTML("beforeend", String.raw`
      <style>
        #exam .st-card { font: 15px/1.5 var(--font); margin-top: 10px; }
        #exam .st-card .st-title { font-weight: 700; font-size: 16px; margin-bottom: 4px; }
        #exam .st-card .st-io { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; }
        #exam .st-card .st-io > :nth-child(odd) { color: var(--muted); font-weight: 600; }
        #exam .st-card .st-demo { margin-top: 6px; padding: 8px 10px; border-radius: 8px; background: var(--panel-2); border: 1px solid var(--line); overflow-x: auto; }
        #exam .st-card .katex-display { margin: .4em 0; }
        #exam .done-box { background: var(--accent-soft); border: 1px solid color-mix(in srgb, var(--accent) 35%, transparent); border-radius: var(--radius); padding: 14px 18px; margin: 26px auto; }
        #exam .done-box h4 { margin: 0 0 6px; font-size: 19px; }
        #exam .ck { display: grid; grid-template-columns: minmax(9em, 1.1fr) 2fr 2fr; border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; font: 14.5px/1.45 var(--font); background: var(--panel); }
        #exam .ck > div { padding: 8px 10px; border-top: 1px solid var(--line); min-width: 0; overflow-x: auto; }
        #exam .ck > .h { font-weight: 700; color: var(--muted); background: var(--panel-2); border-top: none; text-transform: uppercase; font-size: 12px; letter-spacing: .06em; }
        #exam .ck > .n { font-weight: 700; }
        #exam .ck > .n small { display: block; font-weight: 400; color: var(--faint); }
        #exam .ck > .p { color: var(--muted); }
        #exam .ck > .p b { color: var(--fg); font-weight: 600; }
        @media (max-width: 640px) {
          #exam .ck { grid-template-columns: 1fr; }
          #exam .ck > .h { display: none; }
          #exam .ck > .n { border-top: 2px solid var(--line); background: var(--panel-2); }
          #exam .ck > .e, #exam .ck > .p { border-top: none; padding-top: 2px; }
        }
      </style>
      <p>This is the whole pipeline in one sitting: pixels, cameras, corners, two-view geometry, the cost volume, the primal–dual solver, dense tracking and the keyframe logic. Every numeric question draws new numbers each try, and the solutions show every step.</p>
      <p>Work through the parts in order. If a question stalls you, the chapter it comes from is named in brackets. The implementation checklist at the end sums up everything you need to build.</p>
    `);

    // =====================================================================
    // Widget: pipeline recap stepper with a live mini example per stage.
    // =====================================================================
    const theta5 = (b1, b2) => {
      let th = 0.2, it = 0;
      while (th > 1e-4) {
        const b = th >= 1e-3 ? b1 : b2;
        const f = 1 - b * it;
        it++;
        if (f <= 0 || it > 5000) break;
        th *= f;
      }
      return it;
    };
    const itersImpl = theta5(2.5e-4, 2.5e-5);
    const itersPaper = theta5(1e-3, 1e-4);

    const stages = [
      { name: "Frame in", boot: true, ch: 1,
        ins: "RGB video frame", outs: "luma image + 2×2 box pyramid",
        eq: String.raw`Y = 0.2126R + 0.7152G + 0.0722B,\quad u_{l} = \frac{u+0.5}{2^{l}} - 0.5`,
        params: "Mapping resolution: longest side ≤ 512 px (256 on phones).",
        demo: (r) => {
          const R = r.int(0, 255), G = r.int(0, 255), B = r.int(0, 255), u = r.int(0, 300), l = r.int(1, 3);
          const Y = (0.2126 * R + 0.7152 * G + 0.0722 * B) / 255;
          return String.raw`RGB $(${R}, ${G}, ${B})$ → $Y = ${fm(Y)}$. Column $u = ${u}$ at level ${l} → $u_${l} = ${fm((u + 0.5) / 2 ** l - 0.5, 4)}$.`;
        } },
      { name: "KLT corners + tracks", boot: true, ch: 6,
        ins: "luma pyramids of consecutive frames", outs: "corner tracks over ~60 frames",
        eq: String.raw`\lambda_{\min}(M) = \tfrac{a+c}{2} - \sqrt{\left(\tfrac{a-c}{2}\right)^2 + b^2},\quad M\boldsymbol\delta = \mathbf b`,
        params: "5×5 Shi–Tomasi window, best corner per 16×16 tile, ≤ 800 tracks; LK: 4 levels, 15×15 window, ≤ 20 iterations, forward–backward error ≤ 1 px.",
        demo: (r) => {
          const a = r.int(1, 20), c = r.int(1, 20), b = r.int(-8, 8);
          const lm = (a + c) / 2 - Math.sqrt(((a - c) / 2) ** 2 + b * b);
          return String.raw`$M = \begin{pmatrix}${a}&${b}\\${b}&${c}\end{pmatrix}$ → $\lambda_{\min} = ${fm(lm)}$ ${lm > 0.1 * Math.max(a, c) ? "(corner-like)" : "(edge or flat: weak)"}.`;
        } },
      { name: "Self-calibration", boot: true, ch: 7,
        ins: "tracks of frame pairs", outs: "focal length $f$",
        eq: String.raw`E = K^\top F K,\quad \text{cost}(f) = \frac{\sigma_1 - \sigma_2}{\sigma_1}`,
        params: "F: normalised 8-point + RANSAC (800 iterations, 1 px Sampson). Scan hFOV 10°–150°. Fallback: 60° hFOV.",
        demo: (r) => {
          const w = r.pick([640, 1280, 1920]), h = r.pick([50, 60, 70, 80, 90]);
          const f = w / 2 / Math.tan((h / 2) * deg);
          return String.raw`Image width ${w}, hFOV ${h}° → $f = \frac{${w}/2}{\tan ${h / 2}°} = ${fm(f, 1)}$ px.`;
        } },
      { name: "Feature bootstrap", boot: true, ch: 7,
        ins: "tracks + $f$", outs: "poses $T_{wc}$ of ~60 frames, sparse 3D points",
        eq: String.raw`\dot{\mathbf x}_2^\top E\,\dot{\mathbf x}_1 = 0,\quad E = [\mathbf t]_\times R`,
        params: "Initial pair: ≥ 10 frames apart, median flow ≥ 25 px, H/F inlier ratio < 0.85, triangulation angle ≥ 1.5°. BA: 40 LM iterations, Huber 1.5 px. Scale: median depth = 1.",
        demo: (r) => {
          const f = r.pick([500, 600, 800]), b = r.pick([0.1, 0.2, 0.5]), D = r.int(5, 60);
          return String.raw`Side-by-side cameras, $f = ${f}$, baseline ${b}, disparity ${D} px → $z = fb/D = ${fm((f * b) / D)}$.`;
        } },
      { name: "Keyframe setup", boot: false, ch: 12,
        ins: "reference image, $T_{wr}$, predicted $\\xi$ percentiles", outs: "range $[\\xi_{min},\\xi_{max}]$, $\\lambda$",
        eq: String.raw`\lambda = \frac{1}{1 + 0.5\,d_{min}},\quad \xi_{min} = \max(0.5\,\xi_{2\%},\ \xi_{50\%}/6),\ \ \xi_{max} = \min(1.6\,\xi_{98\%},\ 6\,\xi_{50\%})`,
        params: "λ = 1 for the first keyframe. Seeded with ≤ 30 earlier tracked frames within 0.3 × depth and 15°.",
        demo: (r) => {
          const x50 = r.float(0.3, 1.5, 2), x2 = +(x50 * r.float(0.1, 0.8, 2)).toFixed(3), x98 = +(x50 * r.float(1.3, 5, 2)).toFixed(3);
          const lo = Math.max(0.5 * x2, x50 / 6), hi = Math.min(1.6 * x98, 6 * x50);
          const dmin = 1.6 / hi, lam = 1 / (1 + 0.5 * dmin);
          return String.raw`$\xi_{2,50,98\%} = ${fm(x2)}, ${fm(x50)}, ${fm(x98)}$ → range $[${fm(lo)}, ${fm(hi)}]$, $d_{min} = 1.6/\xi_{max} = ${fm(dmin)}$, $\lambda = ${fm(lam)}$.`;
        } },
      { name: "Cost volume", boot: false, ch: 8,
        ins: "keyframe $I_r$, frames $I_m$ with $T_{wm}$", outs: "$C_r(\\mathbf u, d)$ for $S$ layers",
        eq: String.raw`C_r(\mathbf u,d) = \frac{1}{|\mathcal I(r)|}\sum_{m}\big\|I_r(\mathbf u) - I_m\big(\pi(KT_{mr}\pi^{-1}(\mathbf u,d))\big)\big\|_1`,
        params: "S = 64 (32 on phones), linear in ξ. RGB L1. A frame counts for a pixel only if its whole epipolar segment is in view; a voxel needs ≥ 3 views.",
        demo: (r) => {
          const a = [0, 1, 2].map(() => r.float(0, 1, 2)), b = [0, 1, 2].map(() => r.float(0, 1, 2));
          const rho = a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0), n = r.int(2, 9), cb = r.float(0.1, 0.8, 2);
          return String.raw`$I_r = ${vec(a, 2)}$, $I_m = ${vec(b, 2)}$ → $\rho = ${fm(rho)}$; the running average over ${n} frames goes $${cb} \to ${fm((n * cb + rho) / (n + 1))}$.`;
        } },
      { name: "Initialisation", boot: false, ch: 10,
        ins: "cost volume", outs: "$\\mathbf d^0 = \\mathbf a^0$, $\\mathbf q^0 = 0$, $C^{min}_\\mathbf u$, $C^{max}_\\mathbf u$",
        eq: String.raw`d^0_\mathbf u = a^0_\mathbf u = \arg\min_{d} C(\mathbf u, d)\ \ \text{(if localised)}`,
        params: "Localised: samples within τ = max(0.01, 0.1(C_max − C_min)) of C_min span ≤ 6 layers. Other pixels: push–pull fill.",
        demo: (r) => {
          const n = 12, k0 = r.int(1, 10), wide = r.next() < 0.5;
          const row = Array.from({ length: n }, (_, k) => +(wide ? 0.3 + 0.02 * r.next() : 0.15 + 0.05 * Math.abs(k - k0) + 0.02 * r.next()).toFixed(2));
          const cmin = Math.min(...row), cmax = Math.max(...row), tau = Math.max(0.01, 0.1 * (cmax - cmin));
          const idx = row.map((c, k) => (c <= cmin + tau ? k : -1)).filter((k) => k >= 0);
          const width = idx[idx.length - 1] - idx[0] + 1;
          return String.raw`Row ${row.join(", ")}: arg min at layer ${row.indexOf(cmin)}, trough width ${width} layers → ${width <= 6 ? "start from the arg min" : "not localised: fill by push–pull"}.`;
        } },
      { name: "Regularised solve", boot: false, ch: "9–10",
        ins: "cost volume, $g(\\mathbf u)$, $\\mathbf d^0$", outs: "inverse depth map $\\xi_r$",
        eq: String.raw`\mathbf q \leftarrow \Pi\!\left(\tfrac{\mathbf q + \sigma_q g\nabla d}{1+\sigma_q\epsilon}\right),\ d \leftarrow \tfrac{d + \sigma_d(\nabla\!\cdot\!(g\mathbf q) + a/\theta)}{1+\sigma_d/\theta},\ a \leftarrow \arg\min \tfrac{(d-a)^2}{2\theta} + \lambda C`,
        params: String.raw`α = 100, β = 1.6, ε = 1e-4, σ_q = 0.5, σ_d = 0.25, θ: 0.2 → 1e-4 with β = 2.5e-4 (θ ≥ 1e-3) then 2.5e-5: ${itersImpl} iterations (paper's β: ${itersPaper}).`,
        demo: (r) => {
          const n = r.int(1, 400), lam = r.pick([0.5, 0.8, 1]), dc = r.float(0.1, 0.6, 2);
          let th = 0.2;
          for (let k = 0; k < n; k++) th *= 1 - (th >= 1e-3 ? 2.5e-4 : 2.5e-5) * k;
          const rr = Math.sqrt(2 * th * lam * dc);
          return String.raw`After ${n} iterations $\theta = ${fm(th, 5)}$; with $\lambda = ${lam}$, $C^{max}-C^{min} = ${dc}$ the search band is $r = \sqrt{2\theta\lambda\Delta C} = ${fm(rr, 4)}$.`;
        } },
      { name: "Model prediction", boot: false, ch: 11,
        ins: "keyframe meshes, guess $T_{wv}$", outs: "$I_v$, $\\xi_v$, coverage",
        eq: String.raw`\mathbf u_v = \pi\big(K\,T_{vr}\,\pi^{-1}(\mathbf u, \xi_r(\mathbf u))\big)`,
        params: "Mesh on the pixel grid, z-buffered; triangles whose normal is > 85° from the keyframe ray are culled.",
        demo: (r) => {
          const f = 500, cx = 256, u = r.int(100, 400), xi = r.pick([0.25, 0.5, 1, 2]), tx = r.pick([-0.2, -0.1, 0.1, 0.2]);
          const x = (u - cx) / f / xi, uv = (f * (x + tx)) / (1 / xi) + cx;
          return String.raw`$f = ${f}$, $c_x = ${cx}$, column ${u} at $\xi = ${xi}$, virtual camera shifted so $\mathbf t_{vr} = (${tx}, 0, 0)$ → column $${fm(uv, 2)}$ (moved $f\,t_x\,\xi = ${fm(f * tx * xi, 2)}$ px).`;
        } },
      { name: "Dense tracking", boot: false, ch: 11,
        ins: "live frame $I_l$, prediction", outs: "$T_{wl}$, used fraction, RMSE, gain/bias",
        eq: String.raw`F(\psi) = \tfrac12\sum_\mathbf u f_\mathbf u(\psi)^2,\quad J^\top J\,\psi = -J^\top\mathbf f,\quad T_{lv} \leftarrow T_{lv}\,T(\psi)`,
        params: "Rotation pre-alignment (levels 3, 2; 10 iterations). 6DOF: 4 levels, iterations 20/20/15/10, thresholds 0.25/0.18/0.12/0.09 (coarse → fine), LM damping from 1e-4.",
        demo: (r) => {
          const li = r.int(0, 3), tau = [0.25, 0.18, 0.12, 0.09][li];
          const res = Array.from({ length: 8 }, () => r.float(-0.3, 0.3, 2));
          const used = res.filter((v) => Math.abs(v) <= tau).length;
          return String.raw`Level ${3 - li} (threshold ${tau}); residuals ${res.join(", ")} → ${used} used, ${8 - used} rejected.`;
        } },
      { name: "Keyframe / lost logic", boot: false, ch: 12,
        ins: "tracking stats, coverage", outs: "add frame to cost volume? new keyframe? lost?",
        eq: String.raw`\text{new KF} \iff \text{coverage} < 0.92 \ \wedge\ \ge 8\ \text{frames since last}`,
        params: "Tracking OK: used > 0.5 and RMSE < 0.08. Mapping needs used ≥ 0.8. Re-solve every 20 frames. Lost after 15 failures.",
        demo: (r) => {
          const cov = r.float(0.8, 1, 2), since = r.int(2, 20), used = r.float(0.3, 1, 2), rmse = r.float(0.02, 0.12, 3);
          const ok = used > 0.5 && rmse < 0.08;
          return String.raw`coverage ${cov}, ${since} frames since last KF, used ${used}, RMSE ${rmse} → tracking ${ok ? "OK" : "failed"}; ${ok && used >= 0.8 ? "frame joins the cost volume; " : ""}${ok && cov < 0.92 && since >= 8 ? "<b>new keyframe</b>" : "no new keyframe"}.`;
        } },
    ];

    let sel = 0;
    let demoRng = L.rng(12345);
    let demoHtml = stages[0].demo(demoRng);
    const fig = L.figure(root, "<b>Pipeline recap.</b> Tap a dot or use Prev / Next to step through the stages in the order the program runs. <i>New numbers</i> reruns the stage's example with fresh inputs. Green: runs once (bootstrap); blue: the dense loop.");
    const c = L.canvas(fig.el, { aspect: 0.3, maxHeight: 190, scroll: true });
    fig.add(c.el);
    const ctl = L.controls(fig.el);
    fig.add(ctl);
    const card = L.el("div", { class: "st-card" });
    fig.add(card);
    const show = () => {
      const s = stages[sel];
      card.innerHTML = String.raw`<div class="st-title">${sel + 1}. ${s.name} <span style="color:var(--faint);font-weight:400">(chapter ${s.ch})</span></div>
        <div class="st-io"><span>In</span><span>${s.ins}</span><span>Out</span><span>${s.outs}</span></div>
        $$${s.eq}$$
        <div style="color:var(--muted)">${s.params}</div>
        <div class="st-demo"><b>Example:</b> ${demoHtml}</div>`;
      L.typeset(card);
      c.redraw();
    };
    const go = (i) => {
      sel = (i + stages.length) % stages.length;
      demoHtml = stages[sel].demo(demoRng);
      show();
    };
    L.button(ctl, "◀ Prev", () => go(sel - 1));
    L.button(ctl, "Next ▶", () => go(sel + 1), "btn primary");
    L.button(ctl, "New numbers", () => {
      demoRng = L.rng((Math.random() * 4294967295) >>> 0 || 7);
      demoHtml = stages[sel].demo(demoRng);
      show();
    });
    const geom = () => {
      const n = stages.length, pad = Math.max(16, c.w * 0.04);
      const sp = (c.w - 2 * pad) / (n - 1);
      return { n, pad, sp, rad: Math.max(8, Math.min(14, sp * 0.36)), y: c.h * 0.5 };
    };
    c.draw = (ctx) => {
      const t = L.theme();
      const { n, pad, sp, rad, y } = geom();
      const X = (i) => pad + i * sp;
      // loop arc: from "keyframe / lost logic" back to "keyframe setup"
      const x0 = X(4), x1 = X(n - 1), top = Math.max(6, y - rad - c.h * 0.3);
      ctx.save();
      ctx.strokeStyle = t.accent; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x1, y - rad); ctx.bezierCurveTo(x1, top, x0, top, x0, y - rad - 4); ctx.stroke();
      ctx.restore();
      L.draw.arrow(ctx, x0 + 0.5, y - rad - 9, x0, y - rad - 2, t.accent, 1.5, 7);
      L.draw.text(ctx, "every frame", (x0 + x1) / 2, top + 12, t.muted, { size: 11, align: "center" });
      for (let i = 0; i < n - 1; i++) L.draw.line(ctx, X(i) + rad, y, X(i + 1) - rad, y, t.line, 2);
      for (let i = 0; i < n; i++) {
        const col = stages[i].boot ? t.accent3 : t.accent;
        ctx.save();
        ctx.fillStyle = i === sel ? col : t.panel;
        ctx.strokeStyle = col; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(X(i), y, i === sel ? rad + 2 : rad, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.restore();
        L.draw.text(ctx, String(i + 1), X(i), y + 4, i === sel ? "#fff" : t.fg, { size: 11, align: "center", bold: true });
      }
      // selected stage name under its dot, kept inside the canvas
      let size = 14;
      const label = stages[sel].name;
      ctx.font = `600 ${size}px ${t.font}`;
      while (size > 11 && ctx.measureText(label).width > c.w - 8) ctx.font = `600 ${--size}px ${t.font}`;
      const wl = ctx.measureText(label).width;
      const lx = Math.max(4 + wl / 2, Math.min(c.w - 4 - wl / 2, X(sel)));
      L.draw.text(ctx, label, lx, Math.min(c.h - 6, y + rad + 22), t.fg, { size, align: "center", bold: true });
    };
    c.el.addEventListener("pointerdown", (e) => {
      const p = c.pos(e);
      const { n, pad, sp, y } = geom();
      const i = Math.round((p.x - pad) / sp);
      if (i >= 0 && i < n && Math.abs(p.y - y) < c.h * 0.35) go(i);
    });
    show();

    // =====================================================================
    // The exam
    // =====================================================================
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Part A: pixels, cameras, motion</h3>
      <p>Conventions throughout: pixel centres at integer coordinates, $u$ = column, $v$ = row, $\mathbf x_w = T_{wc}\mathbf x_c = R_{wc}\mathbf x_c + \mathbf c_w$. Intensities are in $[0,1]$.</p>
    `);

    L.quiz(root, "exam", [
      { id: "luma", type: "num",
        gen: (r) => {
          const R = r.int(0, 255), G = r.int(0, 255), B = r.int(0, 255);
          const Y = (0.2126 * R + 0.7152 * G + 0.0722 * B) / 255;
          return { data: { R, G, B },
            q: String.raw`[ch. 1] A pixel has 8-bit colour $(R,G,B) = (${R}, ${G}, ${B})$. Using this implementation's Rec. 709 weights $(0.2126, 0.7152, 0.0722)$, what is its luma in $[0,1]$? (3 decimals)`,
            answer: Y, tol: 0.002,
            explain: String.raw`$Y = (0.2126\cdot${R} + 0.7152\cdot${G} + 0.0722\cdot${B})/255 = ${fm(0.2126 * R + 0.7152 * G + 0.0722 * B, 2)}/255 = ${fm(Y, 4)}$. Green dominates because the eye is most sensitive to it.` };
        } },
      { id: "bilinear", type: "num",
        gen: (r) => {
          const u0 = r.int(3, 60), v0 = r.int(3, 60);
          const a = r.float(0, 1, 1), b = r.float(0, 1, 1), cc = r.float(0, 1, 1), e = r.float(0, 1, 1);
          const fx = r.pick([0.25, 0.5, 0.75, 0.2, 0.4, 0.6, 0.8]), fy = r.pick([0.25, 0.5, 0.75, 0.2, 0.4, 0.6, 0.8]);
          const top = (1 - fx) * a + fx * b, bot = (1 - fx) * cc + fx * e, val = (1 - fy) * top + fy * bot;
          return { data: { a, b, c: cc, e, fx, fy },
            q: String.raw`[ch. 1] $I(${u0},${v0}) = ${a}$, $I(${u0 + 1},${v0}) = ${b}$, $I(${u0},${v0 + 1}) = ${cc}$, $I(${u0 + 1},${v0 + 1}) = ${e}$. What is the bilinear sample $I(${fm(u0 + fx)}, ${fm(v0 + fy)})$? (3 decimals)`,
            answer: val, tol: 0.002,
            explain: String.raw`Fractions $t_u = ${fx}$, $t_v = ${fy}$. Top row: $(1-${fx})\cdot${a} + ${fx}\cdot${b} = ${fm(top, 4)}$. Bottom row: $(1-${fx})\cdot${cc} + ${fx}\cdot${e} = ${fm(bot, 4)}$. Blend rows: $(1-${fy})\cdot${fm(top, 4)} + ${fy}\cdot${fm(bot, 4)} = ${fm(val, 4)}$.` };
        } },
      { id: "gradient", type: "num",
        gen: (r) => {
          const l = r.float(0, 1, 2), rt = r.float(0, 1, 2), up = r.float(0, 1, 2), dn = r.float(0, 1, 2);
          const gx = (rt - l) / 2, gy = (dn - up) / 2, m = Math.hypot(gx, gy);
          return { data: { l, rt, up, dn },
            q: String.raw`[ch. 1] Around pixel $(u,v)$: $I(u-1,v) = ${l}$, $I(u+1,v) = ${rt}$, $I(u,v-1) = ${up}$, $I(u,v+1) = ${dn}$. Give the central-difference gradient $(I_u, I_v)$ and its magnitude $\|\nabla I\|$.`,
            answer: [gx, gy, m], labels: ["$I_u$", "$I_v$", "$\\|\\nabla I\\|$"], tol: 0.002,
            explain: String.raw`$I_u = \frac{${rt} - ${l}}{2} = ${fm(gx, 4)}$, $I_v = \frac{${dn} - ${up}}{2} = ${fm(gy, 4)}$ (row $v+1$ is below). $\|\nabla I\| = \sqrt{${P(gx, 4)}^2 + ${P(gy, 4)}^2} = ${fm(m, 4)}$.` };
        } },
      { id: "project", type: "num",
        gen: (r) => {
          const f = r.pick([400, 500, 600]), cx = r.pick([320, 256, 160]), cy = r.pick([240, 192, 120]);
          const z = r.pick([1, 2, 4, 5]);
          const x = +(r.float(-0.4, 0.4, 1) * z).toFixed(2), y = +(r.float(-0.3, 0.3, 1) * z).toFixed(2);
          const u = (f * x) / z + cx, v = (f * y) / z + cy;
          return { data: { f, cx, cy, x, y, z },
            q: String.raw`[ch. 3] $K = \begin{pmatrix}${f}&0&${cx}\\0&${f}&${cy}\\0&0&1\end{pmatrix}$. Where does the camera-frame point $\mathbf x = (${x}, ${y}, ${z})$ project, $\mathbf u = \pi(K\mathbf x)$?`,
            answer: [u, v], labels: ["$u$", "$v$"], tol: 0.01,
            explain: String.raw`$K\mathbf x = (${f}\cdot${P(x)} + ${cx}\cdot${z},\ ${f}\cdot${P(y)} + ${cy}\cdot${z},\ ${z})$. Divide by the last entry: $u = ${f}\cdot${P(x)}/${z} + ${cx} = ${fm(u)}$, $v = ${f}\cdot${P(y)}/${z} + ${cy} = ${fm(v)}$.` };
        } },
      { id: "backproject", type: "num",
        gen: (r) => {
          const f = r.pick([400, 500, 600]), cx = r.pick([320, 256]), cy = r.pick([240, 192]);
          const u = r.int(20, 600), v = r.int(20, 400), d = r.pick([0.25, 0.5, 2, 4, 0.8]);
          const X = [(u - cx) / f / d, (v - cy) / f / d, 1 / d];
          return { data: { f, cx, cy, u, v, d },
            q: String.raw`[ch. 3] Same kind of camera: $f_x = f_y = ${f}$, $(c_x, c_y) = (${cx}, ${cy})$. Back-project pixel $\mathbf u = (${u}, ${v})$ at inverse depth $d = ${d}$: find $\mathbf x = \pi^{-1}(\mathbf u, d) = \frac1d K^{-1}\dot{\mathbf u}$.`,
            answer: X, labels: ["$x$", "$y$", "$z$"], tol: 0.002, rtol: 0.001,
            explain: String.raw`$K^{-1}\dot{\mathbf u} = \big(\frac{${u} - ${cx}}{${f}}, \frac{${v} - ${cy}}{${f}}, 1\big) = (${fm((u - cx) / f, 4)}, ${fm((v - cy) / f, 4)}, 1)$: the ray at depth 1. Scale by $1/d = ${fm(1 / d)}$: $\mathbf x = ${vec(X, 4)}$.` };
        } },
      { id: "transfer", type: "num",
        gen: (r) => {
          const f = 500, cx = 320, cy = 240;
          let u, v, d, t, xr, xm, um;
          do {
            u = r.int(100, 540); v = r.int(80, 400); d = r.pick([0.25, 0.5, 1, 2]);
            t = [r.pick([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]), r.pick([-0.1, 0, 0.1]), r.pick([-0.2, -0.1, 0, 0.1, 0.2])];
            xr = [(u - cx) / f / d, (v - cy) / f / d, 1 / d];
            xm = la.add(xr, t);
            um = [(f * xm[0]) / xm[2] + cx, (f * xm[1]) / xm[2] + cy];
          } while (xm[2] < 0.4 || um[0] < 0 || um[0] > 639 || um[1] < 0 || um[1] > 479);
          return { data: { f, cx, cy, u, v, d, t },
            q: String.raw`[ch. 4] Reference and overlapping camera share $K$ ($f = ${f}$, $(c_x,c_y) = (${cx},${cy})$). $T_{mr}$ has $R_{mr} = I$ and translation $\mathbf t_{mr} = ${vec(t)}$. Where does reference pixel $\mathbf u = (${u}, ${v})$ at inverse depth $d = ${d}$ land in frame $m$, $\mathbf u_m = \pi(KT_{mr}\pi^{-1}(\mathbf u, d))$? (2 decimals)`,
            answer: um, labels: ["$u_m$", "$v_m$"], tol: 0.02,
            explain: String.raw`Back-project: $\mathbf x_r = ${vec(xr, 4)}$. Move into $m$: $\mathbf x_m = \mathbf x_r + \mathbf t_{mr} = ${vec(xm, 4)}$. Project: $u_m = ${f}\cdot${P(xm[0], 4)}/${fm(xm[2], 4)} + ${cx} = ${fm(um[0], 2)}$, $v_m = ${f}\cdot${P(xm[1], 4)}/${fm(xm[2], 4)} + ${cy} = ${fm(um[1], 2)}$. This is the operation behind every cost-volume sample.` };
        } },
      { id: "poses", type: "num",
        gen: (r) => {
          const ang = r.pick([90, 180, -90]);
          const Ra = la.rotZ(ang * deg).map((row) => row.map((x) => Math.round(x)));
          const ca = [r.int(-3, 3), r.int(-3, 3), r.int(-2, 2)], cb = [r.int(-3, 3), r.int(-3, 3), r.int(-2, 2)];
          const D = la.sub(cb, ca);
          const tab = la.matVec(la.T(Ra), D);
          const mat = (M) => String.raw`\begin{pmatrix}${M.map((row) => row.join("&")).join("\\\\")}\end{pmatrix}`;
          return { data: { ang, ca, cb },
            q: String.raw`[ch. 4] Camera $a$: $R_{wa} = R_z(${ang}°) = ${mat(Ra)}$, $\mathbf c_{a} = ${vec(ca)}$. Camera $b$: $R_{wb} = I$, $\mathbf c_{b} = ${vec(cb)}$. Using $T_{ab} = T_{wa}^{-1}T_{wb}$, where is camera $b$'s centre expressed in camera $a$'s frame (the translation of $T_{ab}$)?`,
            answer: tab, labels: ["$x$", "$y$", "$z$"], tol: 1e-6,
            explain: String.raw`$T_{wa}^{-1} = \begin{pmatrix}R_{wa}^\top & -R_{wa}^\top\mathbf c_a\\0&1\end{pmatrix}$, so the translation of $T_{ab}$ is $R_{wa}^\top\mathbf c_b - R_{wa}^\top\mathbf c_a = R_{wa}^\top(\mathbf c_b - \mathbf c_a)$. $\mathbf c_b - \mathbf c_a = ${vec(D)}$, and $R_{wa}^\top = ${mat(la.T(Ra))}$ gives $${vec(tab)}$. (Its rotation is $R_{ab} = R_{wa}^\top$.)` };
        } },
      { id: "rodrigues", type: "num",
        gen: (r) => {
          const ax = r.int(0, 2), th = r.pick([30, 45, 60, 90, 120, 150]);
          const k = [0, 0, 0]; k[ax] = 1;
          let p;
          do { p = [r.int(-3, 3), r.int(-3, 3), r.int(-3, 3)]; } while (la.norm(la.cross(k, p)) === 0);
          const c0 = Math.cos(th * deg), s0 = Math.sin(th * deg);
          const kxp = la.cross(k, p), kp = la.dot(k, p);
          const out = [0, 1, 2].map((i) => p[i] * c0 + kxp[i] * s0 + k[i] * kp * (1 - c0));
          const name = "xyz"[ax];
          return { data: { ax, th, p },
            q: String.raw`[ch. 4] Rotate $\mathbf p = ${vec(p)}$ with the rotation vector $\boldsymbol\omega = \theta\mathbf k$, $\mathbf k$ = the unit ${name} axis, $\theta = ${th}°$. Use Rodrigues: $R\mathbf p = \mathbf p\cos\theta + (\mathbf k\times\mathbf p)\sin\theta + \mathbf k(\mathbf k\cdot\mathbf p)(1-\cos\theta)$. (2 decimals)`,
            answer: out, labels: ["$x$", "$y$", "$z$"], tol: 0.01,
            explain: String.raw`$\cos\theta = ${fm(c0, 4)}$, $\sin\theta = ${fm(s0, 4)}$, $\mathbf k\times\mathbf p = ${vec(kxp)}$, $\mathbf k\cdot\mathbf p = ${kp}$. Sum: $${fm(c0, 4)}\,${vec(p)} + ${fm(s0, 4)}\,${vec(kxp)} + ${fm(kp * (1 - c0), 4)}\,${vec(k)} = ${vec(out)}$. The ${name} component is unchanged, as a rotation about the ${name} axis should leave it.` };
        } },
    ], { title: "Part A · pixels, cameras, motion" });

    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Part B: corners and two-view bootstrap</h3>
      <p>Structure tensor $M = \sum\begin{pmatrix}I_u^2 & I_uI_v\\ I_uI_v & I_v^2\end{pmatrix}$; the essential matrix relates normalised points $\dot{\mathbf x} = K^{-1}\dot{\mathbf u}$ when $\mathbf x_2 = R\mathbf x_1 + \mathbf t$.</p>
    `);

    L.quiz(root, "exam", [
      { id: "shitomasi", type: "num",
        gen: (r) => {
          const a = r.int(1, 20), cc = r.int(1, 20), b = r.int(-8, 8);
          const m = (a + cc) / 2, h = Math.sqrt(((a - cc) / 2) ** 2 + b * b);
          return { data: { a, b, c: cc },
            q: String.raw`[ch. 6] A window's structure tensor is $M = \begin{pmatrix}${a}&${b}\\${b}&${cc}\end{pmatrix}$. What is its Shi–Tomasi score, the smallest value of $\mathbf n^\top M\mathbf n$ over unit $\mathbf n$? (3 decimals)`,
            answer: m - h, tol: 0.002,
            explain: String.raw`$\lambda_{\min} = \frac{a+c}{2} - \sqrt{\left(\frac{a-c}{2}\right)^2 + b^2} = ${fm(m)} - \sqrt{${fm(((a - cc) / 2) ** 2)} + ${b * b}} = ${fm(m)} - ${fm(h, 4)} = ${fm(m - h, 4)}$. Large only when the window has strong gradients in two different directions (a corner).` };
        } },
      { id: "lk", type: "num",
        gen: (r) => {
          const sxx = r.int(4, 12), syy = r.int(4, 12), sxy = r.int(-3, 3);
          const bx = r.float(-2, 2, 1), by = r.float(-2, 2, 1);
          const det = sxx * syy - sxy * sxy;
          const du = (syy * bx - sxy * by) / det, dv = (-sxy * bx + sxx * by) / det;
          return { data: { sxx, syy, sxy, bx, by },
            q: String.raw`[ch. 6] One Lucas–Kanade iteration. Over the window: $\sum I_u^2 = ${sxx}$, $\sum I_uI_v = ${sxy}$, $\sum I_v^2 = ${syy}$, and with $e = I_r(\mathbf q) - I_l(\mathbf q + \mathbf d)$: $\sum I_u e = ${bx}$, $\sum I_v e = ${by}$. Solve $M\boldsymbol\delta = \mathbf b$ for the update $\boldsymbol\delta$. (3 decimals)`,
            answer: [du, dv], labels: ["$\\delta_u$", "$\\delta_v$"], tol: 0.002,
            explain: String.raw`$\det M = ${sxx}\cdot${syy} - ${P(sxy)}^2 = ${det}$. $M^{-1} = \frac{1}{${det}}\begin{pmatrix}${syy}&${-sxy}\\${-sxy}&${sxx}\end{pmatrix}$. $\delta_u = (${syy}\cdot${P(bx)} - ${P(sxy)}\cdot${P(by)})/${det} = ${fm(du, 4)}$, $\delta_v = (-${P(sxy)}\cdot${P(bx)} + ${sxx}\cdot${P(by)})/${det} = ${fm(dv, 4)}$. Add $\boldsymbol\delta$ to $\mathbf d$ and repeat.` };
        } },
      { id: "epipolar", type: "num",
        gen: (r) => {
          let t;
          do { t = [r.int(-2, 2), r.int(-2, 2), r.int(-2, 2)]; } while (la.norm(t) === 0);
          const rot = r.next() < 0.5;
          const x1 = [r.float(-0.5, 0.5, 1), r.float(-0.5, 0.5, 1), 1], x2 = [r.float(-0.5, 0.5, 1), r.float(-0.5, 0.5, 1), 1];
          const Rx1 = rot ? [-x1[1], x1[0], 1] : x1.slice();
          const txr = la.cross(t, Rx1);
          const val = la.dot(x2, txr);
          return { data: { t, rot, x1, x2 },
            q: String.raw`[ch. 7] Two views with $R = ${rot ? String.raw`R_z(90°) = \begin{pmatrix}0&-1&0\\1&0&0\\0&0&1\end{pmatrix}` : "I"}$ and $\mathbf t = ${vec(t)}$, so $E = [\mathbf t]_\times R$. Evaluate the epipolar constraint $\dot{\mathbf x}_2^\top E\,\dot{\mathbf x}_1$ for the normalised points $\dot{\mathbf x}_1 = ${vec(x1)}$, $\dot{\mathbf x}_2 = ${vec(x2)}$ (a true match would give 0).`,
            answer: val, tol: 1e-4,
            explain: String.raw`$E\dot{\mathbf x}_1 = \mathbf t\times(R\dot{\mathbf x}_1)$. $R\dot{\mathbf x}_1 = ${vec(Rx1)}$; $\mathbf t\times R\dot{\mathbf x}_1 = ${vec(txr, 4)}$; dot with $\dot{\mathbf x}_2$: $${fm(val, 4)}$. ${Math.abs(val) < 1e-9 ? "Exactly 0: consistent with this motion." : "Not 0, so the pair does not fit this motion exactly; RANSAC turns such values into a pixel distance (Sampson) and thresholds it."}` };
        } },
      { id: "triangulate", type: "num",
        gen: (r) => {
          const f = r.pick([500, 600, 800]), b = r.pick([0.1, 0.2, 0.5]), z = r.pick([1, 2, 2.5, 4, 5]), cx = 320;
          const D = (f * b) / z, u1 = r.int(150, 500), u2 = u1 - D, x = ((u1 - cx) * z) / f;
          return { data: { f, b, u1, u2, cx },
            q: String.raw`[ch. 7] Camera 2 has the same orientation as camera 1, its centre moved by $b = ${b}$ along camera 1's $x$ axis. $f = ${f}$, $c_x = ${cx}$. A point is seen at column $u_1 = ${u1}$ in camera 1 and $u_2 = ${fm(u2)}$ in camera 2. Triangulate its depth $z$ and its $x$ coordinate (camera 1 frame).`,
            answer: [z, x], labels: ["$z$", "$x$"], tol: 0.002, rtol: 0.002,
            explain: String.raw`$u_1 = f x/z + c_x$ and $u_2 = f(x - b)/z + c_x$, so the disparity is $u_1 - u_2 = fb/z$: $${fm(D)} = ${f}\cdot${b}/z$ gives $z = ${fm(f * b)}/${fm(D)} = ${fm(z)}$. Then $x = (u_1 - c_x)z/f = ${u1 - cx}\cdot${z}/${f} = ${fm(x, 4)}$.` };
        } },
      { id: "ransac", type: "num",
        gen: (r) => {
          const w = r.pick([0.5, 0.6, 0.7, 0.8, 0.9]), p = r.pick([0.95, 0.99, 0.999]), s = 8;
          const ratio = Math.log(1 - p) / Math.log(1 - w ** s);
          return { data: { w, p, s },
            q: String.raw`[ch. 7] 8-point RANSAC for $F$: a fraction $w = ${w}$ of the matches are inliers. How many random samples $N$ are needed to draw at least one all-inlier sample with probability $p = ${p}$? (round up)`,
            answer: Math.ceil(ratio), tol: 1,
            explain: String.raw`One sample is all-inlier with probability $w^8 = ${fm(w ** s, 5)}$. We need $1 - (1 - w^8)^N \ge p$, i.e. $N = \left\lceil \frac{\ln(1-p)}{\ln(1-w^8)} \right\rceil = \lceil ${fm(Math.log(1 - p), 4)}/${fm(Math.log(1 - w ** s), 5)} \rceil = \lceil ${fm(ratio, 2)} \rceil = ${Math.ceil(ratio)}$. (This implementation simply runs a fixed 800–1000 iterations.)` };
        } },
    ], { title: "Part B · corners and bootstrap" });

    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Part C: the cost volume and the regularised solve</h3>
      <div class="eq-card"><div class="eq-label">Paper eqs. (6)–(7), (14) · the energies you are minimising</div>
      $$E_\xi = \sum_\mathbf u g(\mathbf u)\|\nabla\xi(\mathbf u)\|_\epsilon + \lambda C(\mathbf u, \xi(\mathbf u)),\qquad E_\text{aux}(\mathbf u, d_\mathbf u, a_\mathbf u) = \frac{1}{2\theta}(d_\mathbf u - a_\mathbf u)^2 + \lambda C(\mathbf u, a_\mathbf u)$$
      <div class="parts"><span>$\|x\|_\epsilon$</span><span>Huber: $\frac{x^2}{2\epsilon}$ if $|x|\le\epsilon$, else $|x| - \frac\epsilon2$ (eq. 4)</span><span>$g(\mathbf u)$</span><span>$e^{-\alpha\|\nabla I_r\|^\beta}$ (eq. 5)</span><span>$\nabla$</span><span>forward differences, 0 at the last pixel</span></div></div>
    `);

    L.quiz(root, "exam", [
      { id: "cost", type: "num",
        gen: (r) => {
          const a = [0, 1, 2].map(() => r.float(0, 1, 2)), b = [0, 1, 2].map(() => r.float(0, 1, 2));
          const rho = a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);
          const n = r.int(2, 12), cb = r.float(0.05, 0.9, 2);
          const cn = (n * cb + rho) / (n + 1);
          return { data: { a, b, n, cb },
            q: String.raw`[ch. 8] Reference pixel colour $I_r(\mathbf u) = ${vec(a, 2)}$ (RGB). At layer $d$ the new frame $m$ samples $I_m(\mathbf u_m) = ${vec(b, 2)}$. (i) What is $\rho$ (eq. 3, RGB L1 as in this implementation)? (ii) The voxel's average over the ${n} frames so far is $${cb}$. What is $C_r(\mathbf u,d)$ after averaging in frame $m$ (eq. 2)?`,
            answer: [rho, cn], labels: ["$\\rho$", "new $C$"], tol: 0.001,
            explain: String.raw`(i) $\rho = |${a[0]} - ${b[0]}| + |${a[1]} - ${b[1]}| + |${a[2]} - ${b[2]}| = ${fm(rho)}$. (ii) Running average: $C \leftarrow \frac{n\,C + \rho}{n+1} = \frac{${n}\cdot${cb} + ${fm(rho)}}{${n + 1}} = ${fm(cn, 4)}$. No old images are needed, only the sum (or mean) and the count.` };
        } },
      { id: "layers", type: "num",
        gen: (r) => {
          const S = r.pick([32, 64]), xmin = r.pick([0.1, 0.2, 0.25]), step = r.pick([0.02, 0.05, 0.1]);
          const xmax = +(xmin + step * (S - 1)).toFixed(4);
          const k = r.int(1, S - 2), j = r.int(1, S - 2);
          const xk = xmin + k * step, z = 1 / (xmin + j * step);
          return { data: { S, xmin, xmax, k, z },
            q: String.raw`[ch. 8] A keyframe uses $S = ${S}$ layers, linear in inverse depth from $\xi_{min} = ${xmin}$ (layer 0) to $\xi_{max} = ${fm(xmax, 4)}$ (layer ${S - 1}). (i) What inverse depth does layer $k = ${k}$ hold? (ii) Which layer index holds depth $z = ${fm(z, 4)}$?`,
            answer: [xk, j], labels: ["$\\xi_k$", "layer"], tol: 1e-3,
            explain: String.raw`Spacing $\Delta = (\xi_{max} - \xi_{min})/(S-1) = ${fm(xmax - xmin, 4)}/${S - 1} = ${step}$. (i) $\xi_k = ${xmin} + ${k}\cdot${step} = ${fm(xk, 4)}$. (ii) $\xi = 1/${fm(z, 4)} = ${fm(1 / z, 4)}$, index $(\xi - \xi_{min})/\Delta = (${fm(1 / z, 4)} - ${xmin})/${step} = ${j}$.` };
        } },
      { id: "huber", type: "num",
        gen: (r) => {
          const eps = r.pick([0.05, 0.1, 0.2]);
          const inside = r.next() < 0.5;
          const x = inside ? r.sign() * +(eps * r.pick([0.2, 0.4, 0.5, 0.6, 0.8])).toFixed(3) : r.sign() * r.float(eps * 1.5, 1, 2);
          const hv = Math.abs(x) <= eps ? (x * x) / (2 * eps) : Math.abs(x) - eps / 2;
          return { data: { eps, x },
            q: String.raw`[ch. 9] Huber norm (eq. 4) with $\epsilon = ${eps}$: what is $\|${fm(x)}\|_\epsilon$? (4 decimals)`,
            answer: hv, tol: 1e-4,
            explain: Math.abs(x) <= eps
              ? String.raw`$|${fm(x)}| \le ${eps}$: quadratic part, $\frac{x^2}{2\epsilon} = \frac{${fm(x * x, 6)}}{${fm(2 * eps)}} = ${fm(hv, 5)}$.`
              : String.raw`$|${fm(x)}| > ${eps}$: linear part, $|x| - \frac\epsilon2 = ${fm(Math.abs(x))} - ${fm(eps / 2)} = ${fm(hv, 5)}$. The two parts meet with equal value and slope at $|x| = \epsilon$.` };
        } },
      { id: "edgeweight", type: "num",
        gen: (r) => {
          const [p, q, h] = r.pick([[3, 4, 5], [6, 8, 10], [5, 12, 13], [8, 6, 10], [12, 5, 13]]);
          const s = r.pick([0.001, 0.002, 0.004, 0.005]);
          const gx = +(p * s).toFixed(4), gy = +(q * s * r.sign()).toFixed(4), m = h * s;
          const g = Math.exp(-100 * m ** 1.6);
          return { data: { gx, gy },
            q: String.raw`[ch. 9] At a keyframe pixel the luma gradient is $\nabla I_r = (${gx}, ${gy})$. With this implementation's $\alpha = 100$, $\beta = 1.6$, what is the edge weight $g = e^{-\alpha\|\nabla I_r\|^\beta}$? (3 decimals)`,
            answer: g, tol: 0.002,
            explain: String.raw`$\|\nabla I_r\| = \sqrt{${P(gx, 4)}^2 + ${P(gy, 4)}^2} = ${fm(m, 4)}$. $${fm(m, 4)}^{1.6} = e^{1.6\ln ${fm(m, 4)}} = ${fm(m ** 1.6, 5)}$. $g = e^{-100\cdot${fm(m ** 1.6, 5)}} = ${fm(g, 4)}$. Stronger edges give smaller $g$, so smoothing across them costs less.` };
        } },
      { id: "energy", type: "num",
        gen: (r) => {
          const eps = 0.1;
          const xi = [r.float(0.3, 1.5, 2)];
          for (let i = 0; i < 2; i++) xi.push(+(xi[i] + r.sign() * r.pick([0.04, 0.06, 0.08, 0.2, 0.3, 0.5])).toFixed(2));
          const g = [r.float(0.2, 1, 1), r.float(0.2, 1, 1)], C = [0, 1, 2].map(() => r.float(0.05, 0.6, 2)), lam = r.pick([0.5, 1, 2]);
          const hub = (x) => (Math.abs(x) <= eps ? (x * x) / (2 * eps) : Math.abs(x) - eps / 2);
          const h1 = hub(xi[1] - xi[0]), h2 = hub(xi[2] - xi[1]);
          const reg = g[0] * h1 + g[1] * h2, dat = lam * (C[0] + C[1] + C[2]);
          return { data: { xi, g, C, lam, eps },
            q: String.raw`[ch. 9] A 3-pixel, 1D keyframe row has $\xi = (${xi.join(", ")})$, weights $g_1 = ${g[0]}$, $g_2 = ${g[1]}$ ($g_3$ is irrelevant: the last forward difference is 0), costs at those inverse depths $C_1, C_2, C_3 = ${C.join(", ")}$, $\epsilon = ${eps}$, $\lambda = ${lam}$. Evaluate the energy (eq. 6) $E = \sum_i g_i\|\xi_{i+1} - \xi_i\|_\epsilon + \lambda\sum_i C_i$. (4 decimals)`,
            answer: reg + dat, tol: 2e-4,
            explain: String.raw`Differences $${fm(xi[1] - xi[0])}$ and $${fm(xi[2] - xi[1])}$ give Huber values $${fm(h1, 5)}$ and $${fm(h2, 5)}$. Regulariser: $${g[0]}\cdot${fm(h1, 5)} + ${g[1]}\cdot${fm(h2, 5)} = ${fm(reg, 5)}$. Data: $${lam}\cdot(${C.join(" + ")}) = ${fm(dat, 4)}$. $E = ${fm(reg + dat, 5)}$.` };
        } },
      { id: "dual", type: "num",
        gen: (r) => {
          const sq = 0.5, eps = r.pick([0.01, 0.1]);
          const q = [r.float(-0.8, 0.8, 1), r.float(-0.8, 0.8, 1)], g = r.float(0.3, 1, 1);
          const d = r.float(0.5, 1.5, 1), dr = +(d + r.float(-2, 2, 1)).toFixed(1), dd = +(d + r.float(-2, 2, 1)).toFixed(1);
          const grad = [dr - d, dd - d];
          const qt = [0, 1].map((i) => (q[i] + sq * g * grad[i]) / (1 + sq * eps));
          const n = Math.hypot(...qt), qn = qt.map((v) => v / Math.max(1, n));
          return { data: { q, g, d, dr, dd, eps, sq },
            q: String.raw`[ch. 10] Dual step (eqs. 10–11) at an interior pixel: $\mathbf q = ${vec(q)}$, $g = ${g}$, $d = ${d}$, right neighbour $d = ${dr}$, neighbour below $d = ${dd}$, $\sigma_q = ${sq}$, $\epsilon = ${eps}$. Compute $\mathbf q \leftarrow \Pi\big((\mathbf q + \sigma_q g\nabla d)/(1+\sigma_q\epsilon)\big)$ with $\Pi(\mathbf x) = \mathbf x/\max(1, \|\mathbf x\|)$. (3 decimals)`,
            answer: qn, labels: ["$q_u$", "$q_v$"], tol: 0.002,
            explain: String.raw`Forward differences: $\nabla d = (${dr} - ${d},\ ${dd} - ${d}) = ${vec(grad)}$. Ascent: $(\mathbf q + ${fm(sq * g)}\nabla d)/${fm(1 + sq * eps)} = ${vec(qt, 4)}$, length $${fm(n, 4)}$. ${n > 1 ? String.raw`Longer than 1, so divide by it: $${vec(qn, 4)}$.` : "Already inside the unit disc: no change from the projection."}` };
        } },
      { id: "primal", type: "num",
        gen: (r) => {
          const sd = 0.25, th = r.pick([0.2, 0.1, 0.05]);
          const d = r.float(0.4, 1.4, 2), a = +(d + r.float(-0.2, 0.2, 2)).toFixed(2);
          const qm = r.float(-1, 1, 1), qi = r.float(-1, 1, 1), gm = r.float(0.2, 1, 1), gi = r.float(0.2, 1, 1);
          const div = gi * qi - gm * qm;
          const dn = (d + sd * (div + a / th)) / (1 + sd / th);
          return { data: { d, a, th, qm, qi, gm, gi, sd },
            q: String.raw`[ch. 10] Primal step (eq. 12) on a one-row image, interior pixel $i$: $d_i = ${d}$, $a_i = ${a}$, $\theta = ${th}$, $\sigma_d = ${sd}$, duals $q_{i-1} = ${qm}$, $q_i = ${qi}$, weights $g_{i-1} = ${gm}$, $g_i = ${gi}$. With $\nabla\!\cdot(g\mathbf q)_i = g_iq_i - g_{i-1}q_{i-1}$, compute $d_i \leftarrow \frac{d_i + \sigma_d(\nabla\cdot(g\mathbf q)_i + a_i/\theta)}{1 + \sigma_d/\theta}$. (4 decimals)`,
            answer: dn, tol: 2e-4,
            explain: String.raw`Divergence (backward difference, i.e. $-(A G)^\top\mathbf q$): $${gi}\cdot${P(qi)} - ${gm}\cdot${P(qm)} = ${fm(div, 4)}$. $a/\theta = ${fm(a / th, 4)}$. Numerator: $${d} + ${sd}\cdot(${fm(div, 4)} + ${fm(a / th, 4)}) = ${fm(d + sd * (div + a / th), 5)}$. Denominator: $1 + ${sd}/${th} = ${fm(1 + sd / th, 4)}$. $d_i = ${fm(dn, 5)}$: pulled toward $a_i$ and smoothed by the divergence.` };
        } },
      { id: "auxsearch", type: "num",
        gen: (r) => {
          const xs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
          let C, d, th, lam, E, best, gap;
          do {
            C = xs.map(() => r.float(0.05, 0.5, 2));
            d = r.float(0.3, 1.1, 2); th = r.pick([0.05, 0.1, 0.2]); lam = r.pick([0.5, 1]);
            E = xs.map((x, k) => (d - x) ** 2 / (2 * th) + lam * C[k]);
            const sorted = E.slice().sort((p, q) => p - q);
            best = E.indexOf(sorted[0]); gap = sorted[1] - sorted[0];
          } while (gap < 0.01);
          return { data: { C, d, th, lam },
            q: String.raw`[ch. 10] Point-wise search (eqs. 13–14). A pixel's cost row over layers $\xi = ${xs.join(", ")}$ is $C = ${C.join(", ")}$. Current $d_\mathbf u = ${d}$, $\theta = ${th}$, $\lambda = ${lam}$. Which layer value $a_\mathbf u$ minimises $E_\text{aux} = \frac{(d_\mathbf u - a)^2}{2\theta} + \lambda C(\mathbf u, a)$? (before any Newton refinement)`,
            answer: xs[best], tol: 1e-6,
            explain: String.raw`$E_\text{aux}$ per layer: ${E.map((e) => fm(e, 3)).join(", ")}. Smallest at $a = ${xs[best]}$ ($E = ${fm(E[best], 4)}$). The coupling term keeps $a$ near $d$; the data term pulls it to a low-cost layer.` };
        } },
      { id: "band", type: "num",
        gen: (r) => {
          const th = r.pick([0.2, 0.1, 0.05, 0.01, 0.001]), lam = r.pick([0.5, 0.8, 1]);
          const cmin = r.float(0.02, 0.2, 2), cmax = +(cmin + r.float(0.1, 0.8, 2)).toFixed(2);
          const rr = Math.sqrt(2 * th * lam * (cmax - cmin));
          return { data: { th, lam, cmin, cmax },
            q: String.raw`[ch. 10] Search band (eqs. 15–17): a pixel has $C^{min} = ${cmin}$, $C^{max} = ${cmax}$; $\theta = ${th}$, $\lambda = ${lam}$. How far from $d_\mathbf u$ can the minimiser $a_\mathbf u$ possibly be, $r = \sqrt{2\theta\lambda(C^{max} - C^{min})}$? (4 decimals)`,
            answer: rr, tol: 2e-4, rtol: 0.002,
            explain: String.raw`Choosing $a = d$ costs at most $\lambda C^{max}$, and any $a$ costs at least $\frac{(d-a)^2}{2\theta} + \lambda C^{min}$. So a better $a$ needs $\frac{(d-a)^2}{2\theta} \le \lambda(C^{max} - C^{min})$: $r = \sqrt{2\cdot${th}\cdot${lam}\cdot${fm(cmax - cmin)}} = ${fm(rr, 5)}$. Only layers within $\pm r$ of $d$ are tested, and the band shrinks as $\theta \to 0$.` };
        } },
      { id: "newton", type: "num",
        gen: (r) => {
          const step = r.pick([0.02, 0.05, 0.1]), k = r.int(3, 20), a = +(0.1 + k * step).toFixed(3);
          const e0 = r.float(0.1, 0.5, 2), em = +(e0 + r.float(0.02, 0.3, 2)).toFixed(2), ep = +(e0 + r.float(0.02, 0.3, 2)).toFixed(2);
          const grad = (ep - em) / (2 * step), hess = (ep - 2 * e0 + em) / (step * step);
          const ah = a - grad / hess;
          return { data: { step, a, e0, em, ep },
            q: String.raw`[ch. 10] Sub-sample refinement (eq. 18). The best layer is $a = ${a}$ with $E_\text{aux} = ${e0}$; its neighbours $a \mp ${step}$ have $E_\text{aux} = ${em}$ (below) and $${ep}$ (above). Apply one Newton step with numerical derivatives: $\hat a = a - \frac{\nabla E}{\nabla^2E}$. (4 decimals)`,
            answer: ah, tol: 1e-4,
            explain: String.raw`$\nabla E = \frac{${ep} - ${em}}{2\cdot${step}} = ${fm(grad, 4)}$, $\nabla^2E = \frac{${ep} - 2\cdot${e0} + ${em}}{${step}^2} = ${fm(hess, 3)}$. $\hat a = ${a} - ${P(grad, 4)}/${fm(hess, 3)} = ${fm(ah, 5)}$: the vertex of the parabola through the three samples, always within half a layer of $a$ here.` };
        } },
      { id: "theta", type: "num",
        gen: (r) => {
          const b = r.pick([0.01, 0.02, 0.05]), n = r.int(3, 5), t0 = 0.2;
          let th = t0; const fs = [];
          for (let k = 0; k < n; k++) { fs.push(1 - b * k); th *= 1 - b * k; }
          return { data: { b, n, t0 },
            q: String.raw`[ch. 10] The schedule is $\theta_{n+1} = \theta_n(1 - \beta n)$ starting at $n = 0$ with $\theta_0 = ${t0}$. With an (exaggerated) $\beta = ${b}$, what is $\theta_${n}$? (5 decimals)`,
            answer: th, tol: 2e-5,
            explain: String.raw`$\theta_${n} = ${t0}\cdot${fs.map((f) => fm(f, 3)).join("\\cdot")} = ${fm(th, 6)}$. The first step keeps $\theta$ unchanged ($n = 0$); later steps shrink it faster and faster. The real values ($\beta = 2.5\times10^{-4}$, then $2.5\times10^{-5}$ below $\theta = 10^{-3}$) take ${itersImpl} iterations to reach $10^{-4}$.` };
        } },
      { id: "lambda", type: "num",
        gen: (r) => {
          const x98 = r.pick([0.4, 0.5, 0.8, 1, 1.25, 2, 2.5, 4]);
          const dmin = 1 / x98, lam = 1 / (1 + 0.5 * dmin);
          return { data: { x98 },
            q: String.raw`[ch. 12] A new keyframe is created. The largest (98th-percentile) predicted inverse depth is $\xi = ${x98}$, so the nearest scene depth is $d_{min} = 1/\xi$. What data weight does it get, $\lambda = 1/(1 + 0.5\,d_{min})$? (4 decimals)`,
            answer: lam, tol: 1e-4,
            explain: String.raw`$d_{min} = 1/${x98} = ${fm(dmin, 4)}$; $\lambda = 1/(1 + 0.5\cdot${fm(dmin, 4)}) = ${fm(lam, 5)}$. A more distant scene gets smaller $\lambda$, i.e. more smoothing, because the same camera motion gives it less parallax (a weaker data term). The first keyframe uses $\lambda = 1$.` };
        } },
    ], { title: "Part C · cost volume and solver" });

    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Part D: dense tracking and the system</h3>
      <div class="eq-card"><div class="eq-label">Paper eqs. (19)–(21) · tracking</div>
      $$F(\psi) = \frac12\sum_\mathbf u f_\mathbf u(\psi)^2,\quad f_\mathbf u(\psi) = I_l\big(\pi(KT_{lv}(\psi)\pi^{-1}(\mathbf u, \xi_v(\mathbf u)))\big) - I_v(\mathbf u),\quad T_{lv}(\psi) = T_{lv}\exp\Big(\sum_i\psi_i\,\text{gen}_i\Big)$$
      <div class="parts"><span>$\psi$</span><span>$(\mathbf v, \boldsymbol\omega)$: translation first, then rotation</span><span>$\partial\mathbf x/\partial\psi$</span><span>$[\,I \mid -[\mathbf x]_\times\,]$ at $\psi = 0$</span></div></div>
    `);

    L.quiz(root, "exam", [
      { id: "gn1d", type: "num",
        gen: (r) => {
          let J, res, sjj;
          do {
            J = [0, 1, 2, 3].map(() => r.float(-0.5, 0.5, 2));
            res = [0, 1, 2, 3].map(() => r.float(-0.2, 0.2, 2));
            sjj = la.dot(J, J);
          } while (sjj < 0.15);
          const sjr = la.dot(J, res), dlt = -sjr / sjj;
          return { data: { J, res },
            q: String.raw`[ch. 5, 11] Align the live image to the prediction by a single horizontal shift $s$ (one parameter). At the current $s$, four pixels have residuals $f_i = ${res.join(", ")}$ and derivatives $J_i = \partial f_i/\partial s = ${J.join(", ")}$ (the live image's gradient). What Gauss–Newton step $\delta$ solves $\big(\sum J_i^2\big)\delta = -\sum J_if_i$? (4 decimals)`,
            answer: dlt, tol: 2e-4, rtol: 0.002,
            explain: String.raw`$\sum J_i^2 = ${fm(sjj, 4)}$, $\sum J_if_i = ${fm(sjr, 4)}$. $\delta = -${P(sjr, 4)}/${fm(sjj, 4)} = ${fm(dlt, 5)}$. It minimises the linearised cost $\sum(f_i + J_i\delta)^2$; with 6 parameters the same sums become the $6\times6$ system $J^\top J\psi = -J^\top\mathbf f$.` };
        } },
      { id: "jacobian", type: "num",
        gen: (r) => {
          const f = r.pick([400, 500]), z = r.pick([1, 2, 4]);
          const x = r.float(-1, 1, 1), y = r.float(-1, 1, 1), gx = r.float(-0.05, 0.05, 3), gy = r.float(-0.05, 0.05, 3);
          const a = [(gx * f) / z, (gy * f) / z, -(gx * f * x + gy * f * y) / (z * z)];
          const X = [x, y, z], rot = la.cross(X, a);
          const J = [...a, ...rot];
          const ci = r.int(0, 5);
          const names = ["v_x", "v_y", "v_z", "\\omega_x", "\\omega_y", "\\omega_z"];
          return { data: { f, x, y, z, gx, gy, ci },
            q: String.raw`[ch. 11] First tracking iteration ($T_{lv} = I$, so $\mathbf x_l = \mathbf x_v$). A predicted pixel back-projects to $\mathbf x = (${x}, ${y}, ${z})$; $f_x = f_y = ${f}$; the live image gradient at its projection is $(I_u, I_v) = (${gx}, ${gy})$. What is the Jacobian entry $\partial f_\mathbf u/\partial ${names[ci]}$? Chain: $\nabla I\cdot\frac{\partial\pi}{\partial\mathbf x}\cdot[\,I\mid-[\mathbf x]_\times]$. (4 decimals)`,
            answer: J[ci], tol: 2e-4, rtol: 0.002,
            explain: String.raw`Image gradient × projection derivative $\begin{pmatrix}f/z&0&-fx/z^2\\0&f/z&-fy/z^2\end{pmatrix}$ gives $\mathbf a = \big(I_uf/z,\ I_vf/z,\ -(I_ufx + I_vfy)/z^2\big) = ${vec(a, 4)}$. Translation entries are $\mathbf a$ itself; rotation entries are $\mathbf a^\top(-[\mathbf x]_\times) = \mathbf x\times\mathbf a = ${vec(rot, 4)}$. So $\partial f/\partial ${names[ci]} = ${fm(J[ci], 5)}$.` };
        } },
      { id: "outliers", type: "num",
        gen: (r) => {
          const li = r.int(0, 3), tau = [0.25, 0.18, 0.12, 0.09][li];
          const res = [];
          while (res.length < 6) { const v = r.float(-0.4, 0.4, 2); if (Math.abs(Math.abs(v) - tau) > 0.005) res.push(v); }
          const used = res.filter((v) => Math.abs(v) <= tau), rej = res.length - used.length;
          const cost = (used.reduce((s, v) => s + v * v, 0) + rej * tau * tau) / res.length;
          return { data: { tau, res },
            q: String.raw`[ch. 11] On pyramid level ${3 - li} (${["coarsest", "second-coarsest", "second-finest", "finest"][li]}) this implementation rejects residuals with $|f_\mathbf u| > ${tau}$. Six in-view pixels have residuals $${res.join(", ")}$. (i) How many are used? (ii) What is the truncated-quadratic cost $\frac1N\big(\sum_\text{used} f^2 + n_\text{rejected}\,\tau^2\big)$ that the LM safeguard compares? (5 decimals)`,
            answer: [used.length, cost], labels: ["used", "cost"], tol: 1e-5,
            explain: String.raw`Used ($|f| \le ${tau}$): $${used.join(", ") || "none"}$, so ${used.length}. $\sum f^2 = ${fm(used.reduce((s, v) => s + v * v, 0), 5)}$, plus ${rej} × $${tau}^2 = ${fm(rej * tau * tau, 5)}$. Divide by 6: $${fm(cost, 6)}$. Rejected pixels (occluders, moving objects) cost a constant, so they cannot pull the pose.` };
        } },
      { id: "keyframe", type: "num",
        gen: (r) => {
          let cov, ans;
          do {
            const rate = r.float(0.005, 0.02, 3);
            cov = Array.from({ length: 14 }, (_, i) => Math.min(1, +(1 - rate * (i + 1) + r.float(-0.015, 0.015, 3)).toFixed(2)));
            const dip = r.int(2, 6); cov[dip - 1] = r.pick([0.88, 0.9, 0.91]);
            cov = cov.map((v) => (v === 0.92 ? 0.93 : v));
            ans = -1;
            for (let f = 8; f <= 14; f++) if (cov[f - 1] < 0.92) { ans = f; break; }
          } while (ans < 0);
          return { data: { cov },
            q: String.raw`[ch. 12] The last keyframe was created at frame 0. Tracking succeeds on every frame and the active keyframe is already in the model. Predicted coverage for frames 1–14: ${cov.join(", ")}. At which frame does this implementation start the next keyframe?`,
            answer: ans, tol: 0,
            explain: String.raw`Rule: coverage $< 0.92$ <i>and</i> at least 8 frames since the last keyframe. Early dips (before frame 8) are ignored. The first frame $f \ge 8$ with coverage below 0.92 is frame ${ans} (coverage ${cov[ans - 1]}).` };
        } },
      { id: "why-inverse", type: "mc",
        q: "[ch. 3, 8] Why does DTAM space its cost-volume layers evenly in inverse depth $\\xi = 1/z$ rather than in depth $z$?",
        choices: [
          "Equal $\\xi$ steps give near-equal pixel steps along the epipolar line, and far points ($\\xi \\to 0$) stay in range",
          "Inverse depth makes the photometric error convex in $\\xi$, so a single Gauss–Newton step from any starting layer finds the global minimum",
          "Depth cannot be negative, so storing its reciprocal avoids sign errors when the camera moves backwards",
          "The GPU stores reciprocals more accurately than ordinary floats",
        ],
        answer: 0,
        explain: "Pixel displacement is proportional to baseline × ξ, so uniform ξ steps sample the image uniformly; z-spacing would waste layers far away and undersample near. The cost stays non-convex either way." },
      { id: "why-c2f", type: "mc",
        q: "[ch. 5, 6, 11] Why do KLT and dense tracking start on the coarsest pyramid level?",
        choices: [
          "Coarse levels contain more independent pixels per window, so the normal equations there are better conditioned and cheaper to solve",
          "Large motions become small at low resolution, where the linearisation holds; finer levels refine",
          "The finest level is dominated by noise, so its image gradients cannot be trusted until the coarse levels have smoothed them",
          "It removes the need for outlier rejection",
        ],
        answer: 1,
        explain: "Gauss–Newton / LK assume the image is locally linear over the motion. Halving resolution halves the motion in pixels and smooths the image, widening the basin of convergence." },
      { id: "why-theta", type: "mc",
        q: "[ch. 9, 10] Why is $\\theta$ driven toward 0 during the solve?",
        choices: [
          "A small $\\theta$ makes the data term $C(\\mathbf u, a)$ convex, so the point-wise search can be replaced by gradient descent",
          "Shrinking $\\theta$ enlarges the effective step sizes $\\sigma_q, \\sigma_d$, so the primal–dual iterations converge faster",
          "The coupling $\\frac{1}{2\\theta}(d - a)^2$ then forces $d = a$, so the result minimises the original energy (6)",
          "It reduces the number of layers $S$ needed",
        ],
        answer: 2,
        explain: "Eq. (7) is only a relaxation of eq. (6). As θ→0 the relaxation tightens; the search band r also shrinks, which speeds up later iterations." },
      { id: "why-rot", type: "mc",
        q: "[ch. 11] What does the rotation-only pre-alignment between consecutive live frames buy?",
        choices: [
          "It replaces the 6DOF alignment on the fine levels, where rotation and translation can no longer be told apart",
          "It estimates the scene depth range for the next keyframe from how far consecutive frames rotate",
          "It corrects the camera's gain and bias",
          "Rotation moves the whole image a lot but needs no depth, so a cheap 3-parameter fit gives 6DOF a good start",
        ],
        answer: 3,
        explain: "Paper §2.3.1. Rotation-induced motion is depth-independent (R K⁻¹u), so it can be solved image-to-image before the model is involved. Here: levels 3 and 2, 10 iterations each." },
      { id: "what-breaks", type: "multi",
        q: "[ch. 8–12] Which of these would <b>degrade</b> the depth maps? (select all)",
        choices: [
          "Setting $g(\\mathbf u) = 1$ everywhere (no edge weighting)",
          "Averaging a frame into the cost volume even though its tracking pose is badly wrong",
          "Running the Newton refinement once after the solve instead of inside each iteration",
          "Using the running average instead of storing every frame's image",
        ],
        answer: [0, 1, 2],
        explain: "No g: depth is smoothed across object boundaries. A wrong pose puts the minimum at the wrong layer. Newton after convergence fits a parabola dominated by the (huge) coupling term, so it adds nothing (paper §2.2.5). The running average gives exactly the same C as storing the frames." },
    ], { title: "Part D · tracking and system" });

    // =====================================================================
    // Finish + checklist
    // =====================================================================
    root.insertAdjacentHTML("beforeend", String.raw`
      <div class="done-box">
        <h4>You've finished the guide</h4>
        <p style="margin:6px 0">If you solved every question you have done each computation DTAM does, by hand: projecting, warping, matching, averaging, regularising, searching, aligning and deciding. The rest is loops and GPU buffers.</p>
        <p style="margin:6px 0">To build it, go through the checklist below from top to bottom, and test each stage against the hand calculations from the exam before moving to the next.</p>
      </div>

      <h3>Implementation checklist</h3>
      <p>Every component in pipeline order, with its equations and this implementation's values (from <code>dtam/mod.rs</code>, <code>slam.rs</code>, <code>tracker/mod.rs</code>, <code>sfm.rs</code>, <code>calib.rs</code>).</p>
      <div class="ck wide">
        <div class="h">Component</div><div class="h">Key equations</div><div class="h">Values here</div>

        <div class="n">1. Input<small>frame.rs, pyr.wgsl</small></div>
        <div class="e">$Y = 0.2126R + 0.7152G + 0.0722B$; 2×2 box downsample; level coords $u_l = (u + 0.5)/2^l - 0.5$, $c_{x,l}$ likewise, $f_l = f/2^l$</div>
        <div class="p">Mapping/tracking image: longest side <b>≤ 512</b> px (<b>256</b> on phones)</div>

        <div class="n">2. Corners<small>tracker/, corners.wgsl</small></div>
        <div class="e">$M = \sum\nabla I\nabla I^\top$, score $\lambda_{\min}(M)$</div>
        <div class="p"><b>5×5</b> window, best per <b>16×16</b> tile, ≥ <b>0.02</b> × best score, spacing <b>12</b> px, ≤ <b>800</b> tracks, border <b>12</b> px</div>

        <div class="n">3. KLT tracking<small>klt.wgsl</small></div>
        <div class="e">$M\boldsymbol\delta = \sum\nabla I\,(I_r - I_l)$, iterate, coarse to fine</div>
        <div class="p"><b>4</b> levels, <b>15×15</b> window, ≤ <b>20</b> iterations; drop if forward–backward error > <b>1</b> px or mean residual > <b>0.08</b></div>

        <div class="n">4. Focal length<small>calib.rs</small></div>
        <div class="e">8-point $F$ + RANSAC, Sampson distance; $E = K^\top FK$, minimise $(\sigma_1 - \sigma_2)/\sigma_1$ over $f$</div>
        <div class="p">Pairs <b>10/20/30</b> frames apart, ≥ <b>60</b> matches, median flow ≥ <b>8</b> px; RANSAC <b>800</b>, <b>1</b> px; skip if H/F inliers ≥ <b>0.9</b>; scan hFOV <b>10°–150°</b>; fallback <b>60°</b></div>

        <div class="n">5. Bootstrap<small>sfm.rs</small></div>
        <div class="e">$E = [\mathbf t]_\times R$ → 4 candidates, cheirality; DLT triangulation; PnP; bundle adjustment (LM, Schur)</div>
        <div class="p">Window <b>60</b> frames (retry dropping <b>15</b>); pair ≥ <b>10</b> frames apart, flow ≥ <b>25</b> px, H/F < <b>0.85</b>, angle ≥ <b>1.5°</b>; BA <b>40</b> its, Huber <b>1.5</b> px, drop > <b>3</b> px; median depth = <b>1</b></div>

        <div class="n">6. Keyframe setup<small>slam.rs</small></div>
        <div class="e">$\xi_{min} = \max(0.5\xi_{2\%}, \xi_{50\%}/6)$, $\xi_{max} = \min(1.6\xi_{98\%}, 6\xi_{50\%})$; $\lambda = \frac{1}{1 + 0.5d_{min}}$</div>
        <div class="p">KF 0: middle bootstrap frame, $\lambda = 1$. New KF seeded with ≤ <b>30</b> earlier frames within <b>0.3</b> × depth and <b>15°</b></div>

        <div class="n">7. Cost volume<small>cost_update.wgsl</small></div>
        <div class="e">(2) $C_r = \frac{1}{|\mathcal I(r)|}\sum_m\|\rho_r\|_1$, (3) $\rho_r = I_r(\mathbf u) - I_m(\pi(KT_{mr}\pi^{-1}(\mathbf u, d)))$</div>
        <div class="p">$S$ = <b>64</b> (<b>32</b> phones), RGB L1, bilinear; frame counts only if the whole epipolar segment is in view; voxel needs ≥ <b>3</b> views; ≤ <b>250</b> frames</div>

        <div class="n">8. Initialise<small>cost_minmax.wgsl, mapping.rs</small></div>
        <div class="e">$d^0 = a^0 = \arg\min_d C$, $\mathbf q^0 = 0$; $C^{min}, C^{max}$ per pixel</div>
        <div class="p">Localised if trough (τ = max(0.01, 0.1ΔC)) ≤ <b>6</b> layers; others: push–pull fill</div>

        <div class="n">9. Edge weights<small>weights.wgsl</small></div>
        <div class="e">(5) $g = e^{-\alpha\|\nabla I_r\|^\beta}$, central differences on luma</div>
        <div class="p">$\alpha$ = <b>100</b>, $\beta$ = <b>1.6</b></div>

        <div class="n">10. Solve loop<small>dual/primal/aux.wgsl</small></div>
        <div class="e">(10–11) $\mathbf q \leftarrow \Pi\big(\frac{\mathbf q + \sigma_qg\nabla d}{1 + \sigma_q\epsilon}\big)$; (12) $d \leftarrow \frac{d + \sigma_d(\nabla\cdot(g\mathbf q) + a/\theta)}{1 + \sigma_d/\theta}$; (13–14) $a$ = arg min $E_\text{aux}$ within (17) $r = \sqrt{2\theta\lambda\Delta C}$; (18) Newton; $\theta_{n+1} = \theta_n(1 - \beta n)$</div>
        <div class="p">$\epsilon$ = <b>1e-4</b>, $\sigma_q$ = <b>0.5</b>, $\sigma_d$ = <b>0.25</b>, $\theta$: <b>0.2 → 1e-4</b>, $\beta$ = <b>2.5e-4</b> (θ ≥ 1e-3) / <b>2.5e-5</b> → <b>${itersImpl}</b> iterations; Newton step clamped to ±1 layer</div>

        <div class="n">11. Model prediction<small>predict.wgsl</small></div>
        <div class="e">Render each keyframe's grid mesh at $T_{vr} = T_{wv}^{-1}T_{wr}$ with a z-buffer → $I_v$, $\xi_v$</div>
        <div class="p">Cull triangles with normal > <b>85°</b> from the keyframe ray; coverage and ξ percentiles (2, 50, 98 %)</div>

        <div class="n">12. Pose guess<small>slam.rs, track_rot.wgsl</small></div>
        <div class="e">Rotation-only alignment $R \leftarrow \exp(\boldsymbol\omega)R$; constant-velocity translation</div>
        <div class="p">Levels <b>3, 2</b>, <b>10</b> iterations each, threshold <b>0.2</b>; on failure retry from last good pose</div>

        <div class="n">13. 6DOF tracking<small>track6.wgsl, gn.wgsl</small></div>
        <div class="e">(19–21) $J = \nabla I_l\,\frac{\partial\pi}{\partial\mathbf x}R_{lv}[\,I \mid -[\mathbf x_v]_\times]$; $(J^\top J + \mu\,\text{diag})\psi = -J^\top\mathbf f$; $T_{lv} \leftarrow T_{lv}T(\psi)$</div>
        <div class="p"><b>4</b> levels, iterations <b>20/20/15/10</b>, thresholds <b>0.25/0.18/0.12/0.09</b> (coarse → fine); LM μ from <b>1e-4</b>, ×10 on reject, ÷3 on accept; gain/bias $aI_l + b$, $a \in [0.5, 2]$</div>

        <div class="n">14. Decisions<small>slam.rs</small></div>
        <div class="e">OK: used > 0.5, RMSE < 0.08. Map: used ≥ 0.8. New KF: coverage < 0.92, ≥ 8 frames since last</div>
        <div class="p">Re-solve every <b>20</b> frames (10 before joining the model); lost after <b>15</b> failures; ≤ <b>64</b> keyframes (web: 32, phones 10)</div>
      </div>
      <p>Paper values you may meet elsewhere: $\beta = 10^{-3}/10^{-4}$ (${itersPaper} iterations). This implementation uses ¼ of that: slower, but it fills textureless regions better.</p>
    `);
  },
});
