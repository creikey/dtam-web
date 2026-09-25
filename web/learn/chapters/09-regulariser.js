// Chapter 9: regularisation, the energy DTAM minimises (paper §2.2.1, eqs 4-7).
DTAM.chapter({
  id: "regulariser",
  order: 9,
  title: "Regularisation: the energy DTAM minimises",
  subtitle: "Smooth where the data is silent, sharp at edges: eqs (4)–(7)",
  minutes: 50,
  render(root, L) {
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const huber = (x, eps) => { const a = Math.abs(x); return eps > 0 && a <= eps ? (x * x) / (2 * eps) : a - eps / 2; };

    root.insertAdjacentHTML("beforeend", String.raw`
      <p>Chapter 8 ended with a speckled arg-min depth map: fine on texture, random on blank surfaces. The cure is to ask for <b>two things at once</b>:</p>
      <ul>
        <li><b>fit the data</b>: each pixel's depth should have low cost $C(\mathbf u,\xi(\mathbf u))$;</li>
        <li><b>be smooth</b>: neighbouring pixels should have similar depth, except at object edges.</li>
      </ul>
      <p>Where the data is strong it wins; where it is silent (flat cost rows), smoothness fills in from the neighbours.</p>
      <h3>The energy</h3>
      <div class="eq-card"><div class="eq-label">Paper eq. (6) · the energy</div>
      $$E_\xi = \int_\Omega \Big\{\, g(\mathbf u)\,\big\|\nabla\xi(\mathbf u)\big\|_\epsilon \;+\; \lambda\, C\big(\mathbf u, \xi(\mathbf u)\big) \Big\}\, d\mathbf u$$
      <div class="parts">
        <span>$\int_\Omega \dots d\mathbf u$</span><span>"add up over the image". For pixels: a plain sum over every pixel $\mathbf u$</span>
        <span>$\nabla\xi(\mathbf u)$</span><span>how fast inverse depth changes at $\mathbf u$: $(\xi(u{+}1,v)-\xi(u,v),\ \xi(u,v{+}1)-\xi(u,v))$</span>
        <span>$\|\cdot\|_\epsilon$</span><span>Huber norm of that gradient (eq. 4): the smoothness penalty</span>
        <span>$g(\mathbf u)$</span><span>per-pixel weight (eq. 5): small at image edges, so depth may jump there</span>
        <span>$\lambda C(\mathbf u,\xi(\mathbf u))$</span><span>data term: the cost volume read at the chosen depth, weighted by $\lambda$</span>
      </div></div>
      <p><b>Example</b> (1D, 3 pixels, pure TV so $\|x\|_\epsilon = |x|$). $\xi = (0.30, 0.34, 0.34)$, $g = (1, 0.5, \cdot)$, $C(\mathbf u,\xi(\mathbf u)) = (0.10, 0.05, 0.08)$, $\lambda = 2$. Smoothness: $1\cdot|0.04| + 0.5\cdot|0| = 0.04$ (the last pixel has no right neighbour: gradient 0). Data: $2\cdot(0.10+0.05+0.08) = 0.46$. $E = 0.50$.</p>
<pre><code>// evaluating eq. (6) on a W×H inverse depth map
E = 0
for each pixel (u, v):
    gx = (u+1 < W) ? ξ[u+1,v] − ξ[u,v] : 0     // forward differences
    gy = (v+1 < H) ? ξ[u,v+1] − ξ[u,v] : 0
    E += g[u,v] · huber(sqrt(gx² + gy²), ε)     // eq. (4), (5)
    E += λ · C(u, v, ξ[u,v])                    // cost volume, eq. (2)</code></pre>
      <p>Each part gets its own section below: the smoothness penalty, the edge weight $g$, and the balance $\lambda$.</p>
      <h3>Which smoothness penalty?</h3>
      <p>Take a 1D noisy signal $f$ and look for a clean $\xi$ minimising $\sum_i \text{pen}(\xi_{i+1}-\xi_i) + \frac\lambda2\sum_i(\xi_i - f_i)^2$. Three choices for pen:</p>
      <ul>
        <li><b>quadratic</b> $\frac12x^2$: big jumps are very expensive;</li>
        <li><b>total variation (TV)</b> $|x|$: cost grows only linearly with the jump;</li>
        <li><b>Huber</b> $\|x\|_\epsilon$: quadratic for tiny $|x|\le\epsilon$, linear beyond.</li>
      </ul>
    `);

    // ================================================================ W1: 1D denoising
    {
      const n = 100;
      const truth = Array.from({ length: n }, (_, i) => i < 25 ? 0.3 : i < 50 ? 0.7 : i < 75 ? 0.7 - (0.35 * (i - 49)) / 25 : 0.5);
      let seed = 3, sigma = 0.05, mode = "tv", f = [];
      const makeNoisy = () => {
        const r = L.rng(seed);
        f = truth.map((v) => v + sigma * Math.sqrt(-2 * Math.log(1 - r.next() * 0.999999)) * Math.cos(2 * Math.PI * r.next()));
      };
      /** Huber/TV: accelerated primal-dual (chapter 10 explains how it works). */
      const solveHuber = (lam, eps) => {
        const x = Float64Array.from(f), xb = Float64Array.from(f), p = new Float64Array(n - 1);
        let tau = 0.5, sig = 0.5;
        const iters = Math.min(4000, Math.round(300 + 600 / lam));
        for (let it = 0; it < iters; it++) {
          for (let i = 0; i < n - 1; i++) p[i] = clamp((p[i] + sig * (xb[i + 1] - xb[i])) / (1 + sig * eps), -1, 1);
          const th = 1 / Math.sqrt(1 + 2 * lam * tau);
          for (let i = 0; i < n; i++) {
            const dtp = (i > 0 ? p[i - 1] : 0) - (i < n - 1 ? p[i] : 0);
            const xn = (x[i] - tau * dtp + tau * lam * f[i]) / (1 + tau * lam);
            xb[i] = xn + th * (xn - x[i]);
            x[i] = xn;
          }
          tau *= th; sig /= th;
        }
        return x;
      };
      /** Quadratic: exact solve of (DᵀD + λI) x = λ f (tridiagonal). */
      const solveQuad = (lam) => {
        const b = new Float64Array(n), d = new Float64Array(n), cp = new Float64Array(n);
        for (let i = 0; i < n; i++) { b[i] = lam + (i > 0) + (i < n - 1); d[i] = lam * f[i]; cp[i] = i < n - 1 ? -1 : 0; }
        for (let i = 1; i < n; i++) { const m = -1 / b[i - 1]; b[i] -= m * cp[i - 1]; d[i] -= m * d[i - 1]; }
        const x = new Float64Array(n);
        x[n - 1] = d[n - 1] / b[n - 1];
        for (let i = n - 2; i >= 0; i--) x[i] = (d[i] - cp[i] * x[i + 1]) / b[i];
        return x;
      };
      const fig = L.figure(root, "<b>Quadratic vs TV vs Huber on a noisy 1D signal.</b> Pick a penalty and slide $\\lambda$ to find its best result (lowest error). Quadratic cannot remove the noise without also rounding the step. TV keeps the step but turns the ramp into flat stairs. Huber is quadratic below $\\varepsilon$, so small slopes are no longer forced flat: compare the ramp.");
      const c = L.canvas(fig.el, { aspect: 0.55, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ctl2 = L.controls(fig.el); fig.add(ctl2);
      const out = L.readout(fig.el); fig.add(out.el);
      const btns = {};
      for (const [k, label] of [["quad", "quadratic"], ["tv", "TV"], ["huber", "Huber"]]) {
        btns[k] = L.button(ctl, label, () => { mode = k; upd(); });
      }
      L.button(ctl, "new noise", () => { seed++; makeNoisy(); upd(); });
      const sL = L.slider(ctl2, { label: "$\\lambda$", min: -1.3, max: 2.5, step: 0.05, value: 1, fmt: (v) => L.fmt(10 ** v, 2), oninput: () => upd() });
      const sE = L.slider(ctl2, { label: "$\\varepsilon$ (Huber)", min: -3, max: -0.5, step: 0.05, value: -1.6, fmt: (v) => L.fmt(10 ** v, 3), oninput: () => upd() });
      L.slider(ctl2, { label: "noise", min: 0, max: 0.1, step: 0.005, value: sigma, fmt: (v) => v.toFixed(3), oninput: (v) => { sigma = v; makeNoisy(); upd(); } });
      let x = [];
      const upd = () => {
        const lam = 10 ** sL.value, eps = mode === "huber" ? 10 ** sE.value : 0;
        x = mode === "quad" ? solveQuad(lam) : solveHuber(lam, eps);
        for (const k in btns) btns[k].className = "btn" + (k === mode ? " primary" : "");
        const pen = mode === "quad" ? (v) => (v * v) / 2 : (v) => huber(v, eps);
        let er = 0, ed = 0, se = 0;
        for (let i = 0; i < n; i++) {
          if (i < n - 1) er += pen(x[i + 1] - x[i]);
          ed += (lam / 2) * (x[i] - f[i]) ** 2;
          se += (x[i] - truth[i]) ** 2;
        }
        out.html = `smoothness = ${L.fmt(er, 4)} · data = ${L.fmt(ed, 4)} · energy = <b>${L.fmt(er + ed, 4)}</b> · error vs truth (RMS) = <b>${L.fmt(Math.sqrt(se / n), 4)}</b> (noisy input: ${L.fmt(Math.sqrt(f.reduce((s, v, i) => s + (v - truth[i]) ** 2, 0) / n), 4)})`;
        c.redraw();
      };
      c.draw = (ctx) => {
        const t = L.theme();
        const P = L.plot(c, { x0: 0, x1: n - 1, y0: 0.1, y1: 0.9, pad: [12, 8, 24, 36] });
        P.axes(ctx, { xlabel: "pixel i", xticks: 4, yticks: 4 });
        for (let i = 0; i < n; i++) L.draw.dot(ctx, P.X(i), P.Y(clamp(f[i], 0.1, 0.9)), 2, t.faint);
        L.draw.path(ctx, truth.map((v, i) => [P.X(i), P.Y(v)]), t.muted, 1.2, [4, 4]);
        L.draw.path(ctx, Array.from(x, (v, i) => [P.X(i), P.Y(clamp(v, 0.1, 0.9))]), t.accent, 2.5);
      };
      makeNoisy();
      upd();
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <div class="key">TV charges a rise of height $h$ the same whether it happens in one jump or in many small steps ($|h|$ either way). The quadratic penalty is much cheaper for many small steps: one jump of $0.2$ costs $0.2^2 = 0.04$, four steps of $0.05$ cost $4\cdot0.05^2 = 0.01$. So quadratic smoothing <b>blurs edges</b>; TV <b>keeps them</b>.</div>
      <p>Depth maps are exactly "smooth surfaces with sudden jumps at object boundaries", so DTAM uses TV, softened into the Huber norm.</p>
      <h3>The Huber norm</h3>
      <div class="eq-card"><div class="eq-label">Paper eq. (4) · Huber norm</div>
      $$\|\mathbf x\|_\epsilon = \begin{cases} \dfrac{\|\mathbf x\|_2^2}{2\epsilon} & \text{if } \|\mathbf x\|_2 \le \epsilon \\[2mm] \|\mathbf x\|_2 - \dfrac{\epsilon}{2} & \text{otherwise} \end{cases}$$
      <div class="parts">
        <span>$\mathbf x$</span><span>here the gradient $\nabla\xi(\mathbf u)$, a 2-vector; $\|\mathbf x\|_2 = \sqrt{x_1^2+x_2^2}$ is its length</span>
        <span>$\|\mathbf x\|_2 \le \epsilon$</span><span>tiny slopes: quadratic, so gentle slopes stay smooth (no stairs)</span>
        <span>otherwise</span><span>TV: linear, so jumps stay sharp. $-\epsilon/2$ makes the two pieces meet</span>
        <span>$\epsilon$</span><span>switch point: $10^{-4}$ in the paper and in this implementation</span>
      </div></div>
      <p><b>Example.</b> $\epsilon = 0.1$. $x = 0.05$: $0.05^2/0.2 = 0.0125$. $x = 0.3$: $0.3 - 0.05 = 0.25$. At $x=\epsilon$ both pieces give $\epsilon/2 = 0.05$, and both have slope 1: the join is smooth. For a gradient $(0.03, 0.04)$ and $\epsilon=0.01$: length $0.05 > \epsilon$, so $0.05 - 0.005 = 0.045$.</p>
      <p>The paper's print writes the second branch with a 1-norm; this implementation (like the dual update of chapter 10, which projects onto a disc) uses the length $\|\mathbf x\|_2$, which treats all edge directions alike.</p>
    `);

    // ================================================================ W2: Huber plot
    {
      const fig = L.figure(root, "<b>The Huber norm.</b> Drag the orange point; change $\\varepsilon$. Top: penalty, with $|x|$ (TV) and $x^2/2\\varepsilon$ faint. Bottom: its slope, which is $x/\\varepsilon$ inside $[-\\varepsilon,\\varepsilon]$ and never more than 1 outside, so a big jump pulls no harder than a small one.");
      const c = L.canvas(fig.el, { aspect: 0.75 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      let xv = 0.06;
      const sE = L.slider(ctl, { label: "$\\varepsilon$", min: 0.01, max: 0.4, step: 0.01, value: 0.1, oninput: () => upd() });
      const plots = () => ({
        A: L.plot(c, { x0: -0.5, x1: 0.5, y0: 0, y1: 0.5, pad: [12, 8, c.h * 0.45 + 22, 40] }),
        B: L.plot(c, { x0: -0.5, x1: 0.5, y0: -1.2, y1: 1.2, pad: [c.h * 0.55 + 10, 8, 24, 40] }),
      });
      const upd = () => {
        const e = sE.value, h = huber(xv, e);
        const inside = Math.abs(xv) <= e;
        out.html = `x = ${L.fmt(xv, 3)} · ‖x‖<sub>ε</sub> = <b>${L.fmt(h, 4)}</b> (${inside ? "quadratic branch: x²/2ε" : "linear branch: |x| − ε/2"}) · slope = ${L.fmt(inside ? xv / e : Math.sign(xv), 3)}`;
        c.redraw();
      };
      L.drag(c, () => { const { A } = plots(); return [{ x: A.X(xv), y: A.Y(huber(xv, sE.value)) }]; },
        (_, p) => { const { A } = plots(); xv = clamp(A.invX(p.x), -0.5, 0.5); upd(); });
      c.draw = (ctx) => {
        const t = L.theme(), e = sE.value, { A, B } = plots();
        A.axes(ctx, { ylabel: "penalty", xticks: 4, yticks: 2 });
        B.axes(ctx, { xlabel: "x", ylabel: "slope", xticks: 4, yticks: 2 });
        const xs = Array.from({ length: 201 }, (_, i) => -0.5 + i / 200);
        const clipA = (v) => clamp(v, 0, 0.5);
        ctx.save(); ctx.globalAlpha = 0.12; ctx.fillStyle = t.accent;
        ctx.fillRect(A.X(-e), A.Y(0.5), A.X(e) - A.X(-e), A.Y(0) - A.Y(0.5));
        ctx.fillRect(B.X(-e), B.Y(1.2), B.X(e) - B.X(-e), B.Y(-1.2) - B.Y(1.2));
        ctx.restore();
        L.draw.path(ctx, xs.map((x) => [A.X(x), A.Y(Math.abs(x))]), t.faint, 1.2);
        L.draw.path(ctx, xs.filter((x) => (x * x) / (2 * e) <= 0.5).map((x) => [A.X(x), A.Y(clipA((x * x) / (2 * e)))]), t.faint, 1.2, [4, 4]);
        L.draw.path(ctx, xs.map((x) => [A.X(x), A.Y(clipA(huber(x, e)))]), t.accent, 2.5);
        L.draw.path(ctx, xs.map((x) => [B.X(x), B.Y(clamp(x / e, -1, 1))]), t.accent, 2.5);
        L.draw.path(ctx, xs.filter((x) => Math.abs(x / e) <= 1.2).map((x) => [B.X(x), B.Y(x / e)]), t.faint, 1.2, [4, 4]);
        L.draw.line(ctx, B.X(xv), B.Y(-1.2), B.X(xv), B.Y(1.2), t.accent2, 1, [3, 3]);
        L.draw.dot(ctx, B.X(xv), B.Y(clamp(xv / e, -1, 1)), 4.5, t.accent2);
        L.draw.handle(ctx, A.X(xv), A.Y(clipA(huber(xv, e))), t.accent2);
        L.draw.text(ctx, "±ε", A.X(e) + 4, A.Y(0.45), t.accent, { size: 12 });
      };
      upd();
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Edge weight: let depth jump where the image has an edge</h3>
      <p>Depth edges (the outline of an object against the background) almost always show up as image edges too. So DTAM weakens the smoothness term wherever the reference image has a strong gradient (chapter 1):</p>
      <div class="eq-card"><div class="eq-label">Paper eq. (5) · edge weight</div>
      $$g(\mathbf u) = e^{-\alpha\,\|\nabla I_r(\mathbf u)\|_2^{\beta}}$$
      <div class="parts">
        <span>$\nabla I_r(\mathbf u)$</span><span>image gradient by central differences: $\tfrac12(I(u{+}1,v)-I(u{-}1,v),\ I(u,v{+}1)-I(u,v{-}1))$, intensities in $[0,1]$</span>
        <span>$\|\cdot\|_2^\beta$</span><span>edge strength raised to the power $\beta$</span>
        <span>$\alpha, \beta$</span><span>how quickly the weight falls: $\alpha = 100$, $\beta = 1.6$ in this implementation</span>
        <span>$g$</span><span>$1$ on flat image regions (full smoothing), near $0$ on strong edges (depth may jump)</span>
      </div></div>
      <p><b>Example.</b> $I(u{\pm}1,v) = 0.40, 0.46$ and $I(u,v{\pm}1) = 0.50, 0.58$: gradient $(0.03, 0.04)$, length $0.05$. $0.05^{1.6} = e^{1.6\ln 0.05} = 0.00828$, so $g = e^{-100\cdot0.00828} = e^{-0.828} = 0.437$.</p>
<pre><code>// weights.wgsl: once per keyframe, every pixel in parallel
Y(x, y) = luma of I_r at (clamp(x), clamp(y))   // 0.2126 R + 0.7152 G + 0.0722 B
gx = 0.5 · (Y(x+1, y) − Y(x−1, y))
gy = 0.5 · (Y(x, y+1) − Y(x, y−1))
g[x, y] = exp(−α · pow(sqrt(gx² + gy²), β))</code></pre>
    `);

    // ================================================================ W3: g on a real image
    {
      const fig = L.figure(root, "<b>Edge weight on a real frame.</b> Left: the reference image. Right: $g(\\mathbf u)$ (white = 1, full smoothing; black = 0, depth free to jump). Change $\\alpha$ and $\\beta$; tap either picture to read one pixel. Below: $g$ as a function of edge strength.");
      const c = L.canvas(fig.el, { aspect: 0.5, scroll: true });
      fig.add(c.el);
      const c2 = L.canvas(fig.el, { aspect: 0.32, scroll: true });
      fig.add(c2.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const sA = L.slider(ctl, { label: "$\\alpha$", min: 0, max: 400, step: 5, value: 100, oninput: () => upd() });
      const sB = L.slider(ctl, { label: "$\\beta$", min: 0.5, max: 3, step: 0.1, value: 1.6, oninput: () => upd() });
      let img = null, mag = null, gcv = null, icv = null, probe = { x: 96, y: 60 };
      const gOf = (m) => Math.exp(-sA.value * Math.pow(m, sB.value));
      const upd = () => {
        if (img) {
          const d = gcv.getContext("2d").createImageData(img.w, img.h);
          let mean = 0;
          for (let i = 0; i < img.w * img.h; i++) {
            const g = gOf(mag[i]), v = Math.round(g * 255);
            mean += g;
            d.data[4 * i] = v; d.data[4 * i + 1] = v; d.data[4 * i + 2] = v; d.data[4 * i + 3] = 255;
          }
          gcv.getContext("2d").putImageData(d, 0, 0);
          const i = probe.y * img.w + probe.x;
          out.html = `pixel (${probe.x}, ${probe.y}): |∇I| = <b>${L.fmt(mag[i], 4)}</b> → g = <b>${L.fmt(gOf(mag[i]), 4)}</b> · mean g over the image = ${L.fmt(mean / (img.w * img.h), 3)}`;
        } else out.html = "loading img/frame_120.png …";
        c.redraw(); c2.redraw();
      };
      const layout = () => {
        const s = Math.min((c.w - 12) / 2, c.h - 4);
        return { s, x0: (c.w - 2 * s - 12) / 2, x1: (c.w - 2 * s - 12) / 2 + s + 12, y: (c.h - s) / 2 };
      };
      c.draw = (ctx) => {
        const t = L.theme(), g = layout();
        if (!img) { L.draw.text(ctx, "loading…", c.w / 2, c.h / 2, t.muted, { align: "center" }); return; }
        ctx.save(); ctx.imageSmoothingEnabled = false;
        ctx.drawImage(icv, g.x0, g.y, g.s, g.s);
        ctx.drawImage(gcv, g.x1, g.y, g.s, g.s);
        ctx.restore();
        for (const x0 of [g.x0, g.x1]) {
          const px = x0 + ((probe.x + 0.5) / img.w) * g.s, py = g.y + ((probe.y + 0.5) / img.h) * g.s;
          L.draw.dot(ctx, px, py, 5, "rgba(0,0,0,0)", t.accent2);
        }
      };
      c2.draw = (ctx) => {
        const t = L.theme();
        const P = L.plot(c2, { x0: 0, x1: 0.25, y0: 0, y1: 1, pad: [10, 8, 24, 36] });
        P.axes(ctx, { xlabel: "|∇I|", ylabel: "g", xticks: 5, yticks: 2 });
        L.draw.path(ctx, Array.from({ length: 151 }, (_, i) => { const m = (0.25 * i) / 150; return [P.X(m), P.Y(gOf(m))]; }), t.accent, 2.5);
        if (img) { const m = mag[probe.y * img.w + probe.x]; L.draw.dot(ctx, P.X(Math.min(m, 0.25)), P.Y(gOf(m)), 5, t.accent2); }
      };
      c.el.addEventListener("pointerdown", (e) => {
        if (!img) return;
        const g = layout(), p = c.pos(e);
        for (const x0 of [g.x0, g.x1]) {
          if (p.x >= x0 && p.x < x0 + g.s && p.y >= g.y && p.y < g.y + g.s) {
            probe = { x: clamp(Math.floor(((p.x - x0) / g.s) * img.w), 0, img.w - 1), y: clamp(Math.floor(((p.y - g.y) / g.s) * img.h), 0, img.h - 1) };
            upd();
          }
        }
      });
      (async () => {
        try {
          const im = await L.loadImage("img/frame_120.png");
          const w = im.w, h = im.h, Y = new Float32Array(w * h);
          for (let i = 0; i < w * h; i++) Y[i] = (0.2126 * im.rgba[4 * i] + 0.7152 * im.rgba[4 * i + 1] + 0.0722 * im.rgba[4 * i + 2]) / 255;
          const at = (x, y) => Y[clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)];
          mag = new Float32Array(w * h);
          for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            const gx = 0.5 * (at(x + 1, y) - at(x - 1, y)), gy = 0.5 * (at(x, y + 1) - at(x, y - 1));
            mag[y * w + x] = Math.hypot(gx, gy);
          }
          icv = document.createElement("canvas"); icv.width = w; icv.height = h;
          const d = icv.getContext("2d").createImageData(w, h);
          for (let i = 0; i < w * h; i++) { const v = Math.round(Y[i] * 255); d.data[4 * i] = v; d.data[4 * i + 1] = v; d.data[4 * i + 2] = v; d.data[4 * i + 3] = 255; }
          icv.getContext("2d").putImageData(d, 0, 0);
          gcv = document.createElement("canvas"); gcv.width = w; gcv.height = h;
          img = { w, h };
          probe = { x: Math.min(96, w - 1), y: Math.min(60, h - 1) };
          upd();
        } catch (e) {
          out.html = "could not load img/frame_120.png";
        }
      })();
      upd();
    }

    // ================================================================ synthetic cost volume for W4/W5
    const N = 64, S = 32, XI_MIN = 0.15, XI_MAX = 0.6, STEP = (XI_MAX - XI_MIN) / (S - 1);
    const xiOf = (k) => XI_MIN + k * STEP;
    const trueXi = Array.from({ length: N }, (_, u) => u < 24 ? 0.25 : u < 44 ? 0.45 : u < 50 ? 0.25 : 0.25 + (u - 50) * 0.009);
    const tex = (x) => x < 9.5 ? 0.42 + 0.1 * Math.sin(1.9 * x) + 0.07 * Math.sin(3.7 * x + 1)
      : x < 23.5 ? 0.4
      : (x < 27.5 || (x >= 39.5 && x < 43.5)) ? 0.72 + 0.1 * Math.sin(2.3 * x) + 0.06 * Math.sin(3.9 * x + 0.5)
      : x < 39.5 ? 0.74
      : x < 49.5 ? 0.45 + 0.2 * Math.sin((2 * Math.PI * x) / 3)
      : 0.4 + 0.1 * Math.sin(1.7 * x + 0.7) + 0.07 * Math.sin(3.1 * x);
    const Ir = Array.from({ length: N }, (_, u) => tex(u));
    // Each "frame" m with disparity factor D (px per unit ξ) samples the texture
    // at u - D (ξ - ξ_true): the flatland warp of chapter 8, occlusions ignored.
    const Ds = [-16, -12, -8, -4, 4, 8, 12, 16];
    const Cv = (() => {
      const r = L.rng(5), sig = 0.02;
      const gauss = () => Math.sqrt(-2 * Math.log(1 - r.next() * 0.999999)) * Math.cos(2 * Math.PI * r.next());
      const C = [];
      for (let u = 0; u < N; u++) {
        const row = [];
        for (let k = 0; k < S; k++) {
          let s = 0;
          for (const D of Ds) s += Math.abs(tex(u) - (tex(u - D * (xiOf(k) - trueXi[u])) + sig * gauss()));
          row.push(s / Ds.length);
        }
        C.push(row);
      }
      return C;
    })();
    const argmin = Cv.map((row) => row.indexOf(Math.min(...row)));
    const gW = (use) => Array.from({ length: N }, (_, u) => {
      if (!use) return 1;
      const gr = Math.abs(Ir[Math.min(N - 1, u + 1)] - Ir[Math.max(0, u - 1)]) / 2;
      return Math.exp(-100 * Math.pow(gr, 1.6));
    });
    /** Exact minimiser of the 1D discrete eq. (6) over the S layers (dynamic programming). */
    const solveDP = (lam, gw, eps = 1e-4) => {
      let prev = Cv[0].map((c) => lam * c);
      const back = [];
      for (let u = 1; u < N; u++) {
        const cur = new Array(S), bk = new Array(S);
        for (let k = 0; k < S; k++) {
          let best = Infinity, bj = 0;
          for (let j = 0; j < S; j++) {
            const v = prev[j] + gw[u - 1] * huber(xiOf(k) - xiOf(j), eps);
            if (v < best) { best = v; bj = j; }
          }
          cur[k] = best + lam * Cv[u][k];
          bk[k] = bj;
        }
        back.push(bk);
        prev = cur;
      }
      let k = prev.indexOf(Math.min(...prev));
      const E = prev[k], lab = [k];
      for (let u = N - 1; u > 0; u--) { k = back[u - 1][k]; lab.unshift(k); }
      return { lab, E };
    };
    const rmsOf = (lab) => Math.sqrt(lab.reduce((s, k, u) => s + (xiOf(k) - trueXi[u]) ** 2, 0) / N);

    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>The balance $\lambda$</h3>
      <p>$\lambda$ sets how much the data term counts against smoothness. Large $\lambda$: trust the cost volume, noise and all. Small $\lambda$: trust smoothness, and thin objects get flattened into their background.</p>
      <p>In 1D we can find the <b>exact</b> minimiser of the discretised eq. (6) over the $S$ layers by dynamic programming (sweep left to right, keeping for each layer the cheapest way to reach it). That does not scale to 2D images, which is why DTAM needs the trick at the end of this chapter, but it lets us see what the energy itself prefers.</p>
    `);

    // ================================================================ W4: lambda trade-off with the exact 1D minimiser
    {
      const fig = L.figure(root, "<b>What eq. (6) prefers.</b> A synthetic 1D cost volume (pixel $u$ across, $\\xi$ up; dark = low cost): far background, a nearer box, stripes, and a slanted surface, with textureless stretches on the background and inside the box. Dots: arg min. Blue line: exact minimiser of eq. (6). Dashed: truth. Slide $\\lambda$; turn the edge weight off and try small $\\lambda$.");
      const c = L.canvas(fig.el, { aspect: 0.62, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      let useG = true, sol = null;
      const sL = L.slider(ctl, { label: "$\\lambda$", min: -1.7, max: 1.7, step: 0.05, value: 0, fmt: (v) => L.fmt(10 ** v, 3), oninput: () => upd() });
      L.toggle(ctl, "edge weight $g$", true, (v) => { useG = v; upd(); });
      const upd = () => {
        sol = solveDP(10 ** sL.value, gW(useG));
        out.html = `RMS error: arg min = ${L.fmt(rmsOf(argmin), 4)} · eq. (6) minimiser = <b>${L.fmt(rmsOf(sol.lab), 4)}</b> · its energy E = ${L.fmt(sol.E, 4)}`;
        c.redraw();
      };
      c.draw = (ctx) => {
        const t = L.theme();
        const pl = 34, pr = 6, top = 40, bot = c.h - 20, cw = (c.w - pl - pr) / N, lh = (bot - top) / S;
        const Y = (xi) => bot - ((xi - XI_MIN) / STEP + 0.5) * lh, X = (u) => pl + (u + 0.5) * cw;
        const gw = gW(true);
        for (let u = 0; u < N; u++) {
          ctx.fillStyle = L.gray(Ir[u]); ctx.fillRect(pl + u * cw, 4, cw + 0.6, 12);
          ctx.fillStyle = L.gray(gw[u]); ctx.fillRect(pl + u * cw, 20, cw + 0.6, 12);
          for (let k = 0; k < S; k++) {
            ctx.fillStyle = L.viridisish(Cv[u][k] / 0.3);
            ctx.fillRect(pl + u * cw, bot - (k + 1) * lh, cw + 0.6, lh + 0.6);
          }
        }
        L.draw.text(ctx, "Iᵣ", pl - 4, 14, t.muted, { size: 11, align: "right" });
        L.draw.text(ctx, "g", pl - 4, 30, t.muted, { size: 11, align: "right" });
        const tp = [];
        for (let u = 0; u < N; u++) tp.push([pl + u * cw, Y(trueXi[u])], [pl + (u + 1) * cw, Y(trueXi[u])]);
        L.draw.path(ctx, tp, t.fg, 1.5, [4, 3]);
        for (let u = 0; u < N; u++) L.draw.dot(ctx, X(u), Y(xiOf(argmin[u])), 2.4, t.accent2);
        if (sol) L.draw.path(ctx, sol.lab.map((k, u) => [X(u), Y(xiOf(k))]), t.accent, 3);
        for (const xi of [0.15, 0.3, 0.45, 0.6]) L.draw.text(ctx, String(xi), pl - 4, Y(xi) + 4, t.faint, { size: 11, align: "right" });
        L.draw.text(ctx, "pixel u →", c.w - 6, c.h - 5, t.muted, { size: 12, align: "right" });
      };
      upd();
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <p>What to notice:</p>
      <ul>
        <li>Around $\lambda \approx 0.1$–$1$ the minimiser is almost perfect, even on the blank stretches where the arg min is random.</li>
        <li>Large $\lambda$: it follows the arg min, speckles included.</li>
        <li>Small $\lambda$ without $g$: the box costs "edge length × jump" and gets flattened into the background. With $g$, its edges are nearly free, so it survives.</li>
      </ul>
      <p>How this implementation sets $\lambda$ (paper §2.2.6): the data term is worse for far scenes (less parallax for the same camera motion), so</p>
      <div class="eq-card"><div class="eq-label">Paper §2.2.6 · data weight</div>
      $$\lambda = \frac{1}{1 + 0.5\,\bar d}$$
      <div class="parts">
        <span>$\bar d$</span><span>the nearest scene depth the current model predicts for the new keyframe (in bootstrap units)</span>
        <span>$\lambda$</span><span>$= 1$ for the first keyframe; smaller (more smoothing) for distant scenes</span>
      </div></div>
      <p><b>Example.</b> $\bar d = 2$: $\lambda = 1/(1+1) = 0.5$.</p>
      <h3>Why this energy is hard to minimise</h3>
      <ul>
        <li>The <b>regulariser</b> $g\|\nabla\xi\|_\epsilon$ is <b>convex</b> (bowl-shaped, one valley): easy.</li>
        <li>The <b>data term</b> $C(\mathbf u,\cdot)$ is <b>not</b>: it has several dips (stripes, occlusions, noise) and is only known at $S$ samples. Gradient descent would slide into whichever dip is nearest the start.</li>
        <li>Dynamic programming, as above, works only along a 1D chain.</li>
      </ul>
      <p>The fix (from large-displacement optical flow): split the variable in two.</p>
      <div class="eq-card"><div class="eq-label">Paper eq. (7) · decoupled energy</div>
      $$E_{\xi,a} = \int_\Omega \Big\{ g(\mathbf u)\|\nabla\xi(\mathbf u)\|_\epsilon + \frac{1}{2\theta}\big(\xi(\mathbf u) - a(\mathbf u)\big)^2 + \lambda\,C\big(\mathbf u, a(\mathbf u)\big) \Big\}\,d\mathbf u$$
      <div class="parts">
        <span>$a(\mathbf u)$</span><span>an <b>auxiliary</b> inverse depth map, a second copy of $\xi$ (the paper calls it $\alpha(\mathbf u)$; we write $a$, as the code does, to avoid a clash with the $\alpha$ of eq. 5)</span>
        <span>$\xi(\mathbf u)$</span><span>the smooth copy: only it appears in the regulariser</span>
        <span>$\frac{1}{2\theta}(\xi - a)^2$</span><span>coupling term $Q(\mathbf u)$: penalises the copies disagreeing</span>
        <span>$\theta$</span><span>coupling width. Large: copies may differ. $\theta\to0$: forces $\xi = a$, giving back eq. (6)</span>
        <span>$\lambda C(\mathbf u, a(\mathbf u))$</span><span>the data term now reads the <i>auxiliary</i> copy</span>
      </div></div>
      <p>Now each half is easy on its own:</p>
      <ul>
        <li><b>Fix $a$, solve for $\xi$</b>: $g\|\nabla\xi\|_\epsilon + \frac1{2\theta}(\xi-a)^2$ is convex, the same "TV + stay close to a signal" problem as the 1D widget above (with $\lambda \leftrightarrow 1/\theta$). Chapter 10 solves it with a primal-dual method.</li>
        <li><b>Fix $\xi$, solve for $a$</b>: no neighbours appear, so each pixel is on its own. Just try all $S$ layers and keep the one minimising $\frac1{2\theta}(\xi(\mathbf u)-a)^2 + \lambda C(\mathbf u,a)$.</li>
      </ul>
      <p><b>Example.</b> $\xi(\mathbf u) = 0.40$, $\theta = 0.02$, $\lambda = 1$, and layers $a = 0.30, 0.35, 0.40, 0.45$ with costs $0.05, 0.20, 0.30, 0.25$. Sum per layer: $0.25 + 0.05 = 0.30$; $0.0625 + 0.20 = 0.2625$; $0 + 0.30 = 0.30$; $0.0625+0.25 = 0.3125$. Best: $a = 0.35$.</p>
    `);

    // ================================================================ W5: the point-wise a-search
    {
      const fig = L.figure(root, "<b>The point-wise search for $a$.</b> One pixel's cost row (from the volume above, $\\lambda = 1$). Drag the blue handle to set the current smooth value $\\xi(\\mathbf u)$; shrink $\\theta$. Green: $\\lambda C(a)$. Purple dashed: coupling $(\\xi-a)^2/2\\theta$. Blue: their sum at each layer. Orange: the winning layer. Large $\\theta$: $a$ jumps to the global arg min. Small $\\theta$: $a$ is pinned next to $\\xi$.");
      const c = L.canvas(fig.el, { aspect: 0.6 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      let xiNow = 0.4;
      const lam = 1;
      const sU = L.slider(ctl, { label: "pixel $u$", min: 0, max: N - 1, step: 1, value: 46, oninput: () => upd() });
      const sT = L.slider(ctl, { label: "$\\theta$", min: -4, max: 0, step: 0.05, value: -1, fmt: (v) => (10 ** v).toExponential(1), oninput: () => upd() });
      const P = () => {
        const ymax = lam * Math.max(...Cv[sU.value]) * 1.25 + 0.02;
        return L.plot(c, { x0: XI_MIN, x1: XI_MAX, y0: 0, y1: ymax, pad: [14, 8, 34, 40] });
      };
      const search = () => {
        const th = 10 ** sT.value, row = Cv[sU.value];
        let best = 0, eb = Infinity;
        for (let k = 0; k < S; k++) {
          const e = (xiNow - xiOf(k)) ** 2 / (2 * th) + lam * row[k];
          if (e < eb) { eb = e; best = k; }
        }
        return { best, eb };
      };
      const upd = () => {
        const { best } = search();
        out.html = `ξ(u) = ${L.fmt(xiNow, 3)} · θ = ${(10 ** sT.value).toExponential(1)} → a = <b>${L.fmt(xiOf(best), 3)}</b> · arg min of C alone = ${L.fmt(xiOf(argmin[sU.value]), 3)} · true ξ = ${L.fmt(trueXi[sU.value], 3)}`;
        c.redraw();
      };
      L.drag(c, () => { const p = P(); return [{ x: p.X(xiNow), y: c.h - 12 }]; }, (_, q) => { xiNow = clamp(P().invX(q.x), XI_MIN, XI_MAX); upd(); }, { radius: 24 });
      c.draw = (ctx) => {
        const t = L.theme(), p = P(), row = Cv[sU.value], th = 10 ** sT.value;
        p.axes(ctx, { xlabel: "a", xticks: 3, yticks: 2, fmt: (v) => +v.toFixed(2) });
        const ytop = p.y1;
        const q = [], d = [], s = [];
        for (let k = 0; k < S; k++) {
          const a = xiOf(k), cq = (xiNow - a) ** 2 / (2 * th);
          d.push([p.X(a), p.Y(lam * row[k])]);
          if (cq <= ytop) q.push([p.X(a), p.Y(cq)]);
          s.push([p.X(a), p.Y(Math.min(ytop, cq + lam * row[k]))]);
        }
        const qc = [];
        for (let i = 0; i <= 200; i++) {
          const a = XI_MIN + ((XI_MAX - XI_MIN) * i) / 200, v = (xiNow - a) ** 2 / (2 * th);
          if (v <= ytop) qc.push([p.X(a), p.Y(v)]);
          else if (qc.length) { L.draw.path(ctx, qc.splice(0), t.accent4, 1.5, [5, 4]); }
        }
        L.draw.path(ctx, qc, t.accent4, 1.5, [5, 4]);
        L.draw.path(ctx, d, t.accent3, 2);
        L.draw.path(ctx, s, t.accent, 2.5);
        for (const pt of s) L.draw.dot(ctx, pt[0], pt[1], 2.2, t.accent);
        const { best } = search();
        L.draw.dot(ctx, p.X(xiOf(best)), p.Y(Math.min(ytop, (xiNow - xiOf(best)) ** 2 / (2 * th) + lam * row[best])), 6, t.accent2);
        L.draw.line(ctx, p.X(trueXi[sU.value]), p.Y(0), p.X(trueXi[sU.value]), p.Y(ytop), t.fg, 1, [4, 4]);
        L.draw.line(ctx, p.X(xiNow), p.Y(0), p.X(xiNow), c.h - 12, t.accent, 1.5);
        L.draw.handle(ctx, p.X(xiNow), c.h - 12, t.accent);
        L.draw.text(ctx, "ξ", p.X(xiNow) + 12, c.h - 8, t.accent, { size: 13, bold: true });
      };
      upd();
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <div class="key">Start with a large $\theta$: $a$ is free to jump to the best-matching depth anywhere, which escapes the local dips. Then shrink $\theta$ step by step towards 0, so $\xi$ and $a$ are squeezed together and the result minimises the original eq. (6). This implementation goes from $\theta = 0.2$ down to $10^{-4}$.</div>
<pre><code>// the overall plan (chapter 10 fills in each step)
ξ = a = initial guess (arg min where localised, interpolated elsewhere)
θ = 0.2
while θ > 1e-4:
    ξ ← argmin_ξ Σ g‖∇ξ‖_ε + (ξ − a)² / 2θ           // convex: primal-dual
    for each pixel u:                                  // point-wise
        a(u) ← argmin over layers k of (ξ(u) − ξ_k)² / 2θ + λ C(u, k)
    θ ← smaller θ
return ξ</code></pre>
    `);

    // ================================================================ quiz
    L.quiz(root, "regulariser", [
      { id: "hsmall", type: "num",
        gen: (r) => {
          const e = r.pick([0.1, 0.2, 0.5]), x = +(r.sign() * e * r.pick([0.2, 0.4, 0.5, 0.6, 0.8])).toFixed(3);
          const ans = (x * x) / (2 * e);
          return { q: String.raw`With $\epsilon = ${e}$, what is the Huber norm $\|x\|_\epsilon$ of $x = ${x}$ (4 decimals)?`, answer: ans, tol: 1e-4,
            explain: String.raw`$|x| = ${Math.abs(x)} \le \epsilon$, so the quadratic branch: $x^2/2\epsilon = ${L.fmt(x * x, 6)}/${2 * e} = ${L.fmt(ans, 5)}$.` };
        } },
      { id: "hbig", type: "num",
        gen: (r) => {
          const e = r.pick([0.01, 0.05, 0.1, 0.2]), x = +(r.sign() * r.float(0.25, 0.9, 2)).toFixed(2);
          const ans = Math.abs(x) - e / 2;
          return { q: String.raw`With $\epsilon = ${e}$, what is $\|x\|_\epsilon$ for $x = ${x}$?`, answer: ans, tol: 1e-4,
            explain: String.raw`$|x| = ${Math.abs(x)} > \epsilon$: linear branch, $|x| - \epsilon/2 = ${Math.abs(x)} - ${e / 2} = ${L.fmt(ans, 4)}$.` };
        } },
      { id: "hvec", type: "num",
        gen: (r) => {
          const [a, b, h] = r.pick([[3, 4, 5], [6, 8, 10], [5, 12, 13], [8, 6, 10]]);
          const s = r.pick([0.001, 0.002, 0.005]), e = r.pick([0.01, 0.02, 0.05]);
          const gx = +(a * s).toFixed(4), gy = +(r.sign() * b * s).toFixed(4), len = h * s;
          const ans = huber(len, e);
          return { q: String.raw`The inverse-depth gradient at a pixel is $\nabla\xi = (${gx}, ${gy})$ and $\epsilon = ${e}$. What is $\|\nabla\xi\|_\epsilon$ (eq. 4, using the length $\|\cdot\|_2$)? Give 3 significant figures.`,
            answer: ans, rtol: 0.005,
            explain: String.raw`Length $\sqrt{${gx}^2 + ${gy}^2} = ${L.fmt(len, 5)}$. ${len <= e ? String.raw`That is $\le\epsilon$: $${L.fmt(len, 5)}^2/(2\cdot${e}) = ${L.fmt(ans, 6)}$.` : String.raw`That is $>\epsilon$: $${L.fmt(len, 5)} - ${e}/2 = ${L.fmt(ans, 6)}$.`}` };
        } },
      { id: "gval", type: "num",
        gen: (r) => {
          for (;;) {
            const al = r.pick([10, 20, 50, 100]), be = r.pick([1, 1.6, 2]);
            const l = r.float(0.2, 0.8, 2), rr = +clamp(l + r.sign() * r.float(0.02, 0.3, 2), 0.05, 0.95).toFixed(2);
            const up = r.float(0.2, 0.8, 2), dn = +clamp(up + r.sign() * r.float(0.0, 0.3, 2), 0.05, 0.95).toFixed(2);
            const gx = 0.5 * (rr - l), gy = 0.5 * (dn - up), m = Math.hypot(gx, gy), ex = al * Math.pow(m, be);
            if (ex < 0.1 || ex > 3.5) continue;
            const ans = Math.exp(-ex);
            return { q: String.raw`Around pixel $(u,v)$ of $I_r$: $I(u{-}1,v) = ${l}$, $I(u{+}1,v) = ${rr}$, $I(u,v{-}1) = ${up}$, $I(u,v{+}1) = ${dn}$. With $\alpha = ${al}$, $\beta = ${be}$, what is $g(\mathbf u)$ (3 significant figures)?`,
              answer: ans, rtol: 0.01,
              explain: String.raw`Central differences: $g_x = (${rr} - ${l})/2 = ${L.fmt(gx, 4)}$, $g_y = (${dn} - ${up})/2 = ${L.fmt(gy, 4)}$, length $${L.fmt(m, 5)}$. $${L.fmt(m, 5)}^{${be}} = ${L.fmt(Math.pow(m, be), 5)}$; times $${al}$ gives $${L.fmt(ex, 4)}$; $g = e^{-${L.fmt(ex, 4)}} = ${L.fmt(ans, 4)}$.` };
          }
        } },
      { id: "ghalf", type: "num",
        gen: (r) => {
          const al = r.pick([50, 100, 200]), be = r.pick([1, 1.5, 2]);
          const ans = Math.pow(Math.log(2) / al, 1 / be);
          return { q: String.raw`With $\alpha = ${al}$, $\beta = ${be}$, at what image gradient length $\|\nabla I_r\|$ does the edge weight drop to $g = 0.5$ (4 significant figures)?`,
            answer: ans, rtol: 0.005,
            explain: String.raw`$e^{-\alpha m^\beta} = 0.5 \Rightarrow \alpha m^\beta = \ln 2 \Rightarrow m = (\ln 2/\alpha)^{1/\beta} = (${L.fmt(Math.log(2) / al, 6)})^{${L.fmt(1 / be, 4)}} = ${L.fmt(ans, 5)}$.` };
        } },
      { id: "energy", type: "num",
        gen: (r) => {
          const e = 0.01, lam = r.pick([0.5, 1, 2]);
          const xi = [r.float(0.2, 0.5, 2)];
          for (let i = 0; i < 3; i++) xi.push(+(xi[i] + r.pick([0, 0.004, -0.006, 0.05, -0.08, 0.12])).toFixed(3));
          const g = [r.pick([1, 0.8, 0.5]), r.pick([1, 0.3, 0.05]), r.pick([1, 0.9, 0.6])];
          const C = xi.map(() => r.float(0.02, 0.2, 2));
          let reg = 0;
          for (let i = 0; i < 3; i++) reg += g[i] * huber(xi[i + 1] - xi[i], e);
          const data = lam * C.reduce((s, v) => s + v, 0), ans = reg + data;
          return { q: String.raw`A 1D inverse-depth map of 4 pixels: $\xi = (${xi.join(", ")})$, edge weights $g = (${g.join(", ")}, \cdot)$, data costs $C(\mathbf u,\xi(\mathbf u)) = (${C.join(", ")})$, $\lambda = ${lam}$, $\epsilon = ${e}$. Using forward differences (the last pixel has gradient 0), what is the energy of eq. (6) (4 decimals)?`,
            answer: ans, tol: 2e-4,
            explain: String.raw`Differences: ${xi.slice(1).map((v, i) => L.fmt(v - xi[i], 3)).join(", ")}. Huber (quadratic if $|x|\le0.01$, else $|x| - 0.005$): ${xi.slice(1).map((v, i) => L.fmt(huber(v - xi[i], e), 5)).join(", ")}. Weighted by $g$: ${L.fmt(reg, 5)}. Data: $${lam}\cdot${L.fmt(C.reduce((s, v) => s + v, 0), 3)} = ${L.fmt(data, 4)}$. $E = ${L.fmt(ans, 5)}$.` };
        } },
      { id: "quadtv", type: "num",
        gen: (r) => {
          const h = r.pick([0.2, 0.3, 0.4, 0.6]), k = r.pick([2, 3, 4, 5, 6]);
          return { q: String.raw`Inverse depth rises by $h = ${h}$ across an edge. Version A: in one jump. Version B: in ${k} equal steps. With the quadratic penalty $\sum(\Delta\xi)^2$, what does each version cost? (With TV both would cost $${h}$.)`,
            answer: [h * h, (h * h) / k], labels: ["A", "B"], rtol: 0.005,
            explain: String.raw`A: $${h}^2 = ${L.fmt(h * h, 4)}$. B: $${k}\cdot(${h}/${k})^2 = ${h}^2/${k} = ${L.fmt((h * h) / k, 5)}$. Quadratic smoothing prefers spreading the jump out (blurring the edge); TV is indifferent, and with the data term pulling towards a sharp edge it keeps it.` };
        } },
      { id: "coupling", type: "num",
        gen: (r) => {
          const th = r.pick([0.2, 0.1, 0.05, 0.01, 0.001]), xi = r.float(0.2, 0.8, 2), a = +(xi + r.sign() * r.float(0.01, 0.1, 2)).toFixed(2);
          const ans = (xi - a) ** 2 / (2 * th);
          return { q: String.raw`At one pixel $\xi = ${xi}$ and $a = ${a}$. With $\theta = ${th}$, what is the coupling term $Q = \frac{1}{2\theta}(\xi - a)^2$ (4 significant figures)?`,
            answer: ans, rtol: 0.005,
            explain: String.raw`$(${xi} - ${a})^2 / (2\cdot${th}) = ${L.fmt((xi - a) ** 2, 6)}/${L.fmt(2 * th, 4)} = ${L.fmt(ans, 5)}$. The smaller $\theta$, the more a disagreement costs.` };
        } },
      { id: "asearch", type: "num",
        gen: (r) => {
          for (;;) {
            const th = r.pick([0.01, 0.02, 0.05]), lam = r.pick([0.5, 1, 2]);
            const a0 = r.pick([0.2, 0.3, 0.4]), st = 0.05;
            const as = [0, 1, 2, 3, 4].map((k) => +(a0 + k * st).toFixed(2));
            const xi = +(a0 + r.float(0, 0.2, 2)).toFixed(2);
            const C = as.map(() => r.float(0.02, 0.4, 2));
            const E = as.map((a, k) => (xi - a) ** 2 / (2 * th) + lam * C[k]);
            const srt = E.slice().sort((p, q) => p - q);
            if (srt[1] - srt[0] < 0.01) continue;
            const kb = E.indexOf(srt[0]);
            return { q: String.raw`Point-wise step of eq. (7) at one pixel: $\xi(\mathbf u) = ${xi}$, $\theta = ${th}$, $\lambda = ${lam}$. The layers $a = ${as.join(", ")}$ have costs $C = ${C.join(", ")}$. Which $a$ minimises $\frac{1}{2\theta}(\xi - a)^2 + \lambda C(\mathbf u, a)$?`,
              answer: as[kb], tol: 1e-6,
              explain: String.raw`Totals per layer: ${as.map((a, k) => `$${a}$: ${L.fmt((xi - a) ** 2 / (2 * th), 4)} + ${L.fmt(lam * C[k], 3)} = <b>${L.fmt(E[k], 4)}</b>`).join("; ")}. Smallest at $a = ${as[kb]}$.` };
          }
        } },
      { id: "lambda", type: "num",
        gen: (r) => {
          const d = r.pick([0.5, 1, 1.5, 2, 3, 4, 6]);
          const ans = 1 / (1 + 0.5 * d);
          return { q: String.raw`A new (not the first) keyframe's nearest predicted scene depth is $\bar d = ${d}$. What $\lambda$ does it get (3 decimals)?`,
            answer: ans, tol: 1e-3,
            explain: String.raw`$\lambda = 1/(1 + 0.5\cdot${d}) = 1/${1 + 0.5 * d} = ${L.fmt(ans, 4)}$. Farther scenes get a smaller $\lambda$: more smoothing for a weaker data term.` };
        } },
      { id: "tvedge", type: "mc",
        q: "Why does TV keep sharp depth edges while quadratic smoothing blurs them?",
        choices: [
          "TV charges a rise of height $h$ the same ($|h|$) however it is split, while the quadratic penalty is much cheaper when the rise is spread over many small steps",
          "TV ignores gradients larger than $\\epsilon$",
          "TV is non-convex, so it can snap to edges",
          "TV looks at the image to decide where edges go"],
        answer: 0,
        explain: "Spreading a jump of $h$ over $k$ steps costs $h^2/k$ quadratically but still $h$ with TV. The image-edge information comes from $g$, a separate ingredient." },
      { id: "grole", type: "mc",
        q: "At a pixel on a strong edge of the reference image, the weight $g(\\mathbf u)$ is…",
        choices: [
          "close to 0, so a depth jump there costs almost nothing",
          "close to 1, so the depth there is kept smooth",
          "larger than 1, to sharpen the edge",
          "undefined, because the gradient is too large"],
        answer: 0,
        explain: "$g = e^{-\\alpha\\|\\nabla I\\|^\\beta}$ falls towards 0 as the image gradient grows. Depth edges usually coincide with image edges." },
      { id: "huberwhy", type: "mc",
        q: "Why does DTAM use the Huber norm with a tiny $\\epsilon \\approx 10^{-4}$ rather than pure TV?",
        choices: [
          "To avoid the staircase effect of TV on gently sloped surfaces, while still allowing sharp discontinuities",
          "To make the energy non-convex",
          "Because Huber is cheaper to compute than an absolute value",
          "To allow larger depth jumps than TV does"],
        answer: 0,
        explain: "Below $\\epsilon$ the penalty is quadratic, which smooths tiny slopes instead of forming flat steps. Above it, it is TV. The paper notes TV's stair-casing." },
      { id: "lameffect", type: "mc",
        q: "Increasing $\\lambda$ in eq. (6) makes the solution…",
        choices: [
          "follow the cost volume more closely: closer to the (noisy) arg-min map",
          "smoother, with more pixels filled from their neighbours",
          "ignore the edge weight $g$",
          "converge to the average depth of the image"],
        answer: 0,
        explain: "$\\lambda$ multiplies the data term. Large $\\lambda$: data wins, speckles come back. Small $\\lambda$: smoothness wins, thin structures can be flattened." },
      { id: "hard", type: "multi",
        q: "Why can't eq. (6) simply be minimised by gradient descent from the arg-min map? (select all)",
        choices: [
          "The data term $C(\\mathbf u,\\cdot)$ has several local minima (it is non-convex)",
          "The cost is only known at $S$ discrete layers, so its slope is unreliable",
          "The Huber regulariser is non-convex",
          "The energy has no minimum at all"],
        answer: [0, 1],
        explain: "The regulariser is convex; the trouble is entirely in the data term. Eq. (7) moves the data term onto the auxiliary variable, where it can be searched exhaustively." },
      { id: "decouple", type: "multi",
        q: "Which statements about the decoupled energy, eq. (7), are true? (select all)",
        choices: [
          "With $a$ fixed, the energy is convex in $\\xi$",
          "With $\\xi$ fixed, each pixel's $a$ can be found independently by trying all $S$ layers",
          "As $\\theta\\to0$ the coupling term forces $\\xi = a$, recovering eq. (6)",
          "$\\theta$ should end large so the two copies can stay different",
          "The step for $a$ needs the neighbouring pixels' values of $a$"],
        answer: [0, 1, 2],
        explain: "Large $\\theta$ at the start lets $a$ jump to good matches far away; shrinking it to about $10^{-4}$ ties the copies together. The $a$-step has no neighbour terms, which is what makes it point-wise." },
    ]);
  },
});
