// Chapter 5: least squares, Gauss–Newton, Levenberg–Marquardt, robust costs,
// coarse-to-fine. The optimisation toolbox used by chapters 6, 7 and 11.
DTAM.chapter({
  id: "lsq",
  order: 5,
  title: "Least squares and Gauss–Newton",
  subtitle: "Fitting by minimising squared errors, one linearisation at a time",
  minutes: 55,
  render(root, L) {
    const la = L.la;

    // ================================================================ 1. residuals
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>Almost every step of DTAM is "find the numbers that make predictions match measurements best": corner motion (chapter 6), camera poses and 3D points (chapter 7), and the live camera pose on every frame (chapter 11). They all use one recipe from this chapter.</p>
      <h3>Residuals and the sum of squares</h3>
      <p>A <b>residual</b> is prediction minus measurement for one observation. We want all residuals small at once, so we minimise the <b>sum of their squares</b>.</p>
      <div class="eq-card"><div class="eq-label">Least-squares cost</div>
      $$S(\mathbf p) = \sum_{i=1}^{N} r_i(\mathbf p)^2$$
      <div class="parts">
        <span>$\mathbf p$</span><span>the unknown parameters (e.g. slope and offset of a line; later a camera pose)</span>
        <span>$r_i(\mathbf p)$</span><span>residual of observation $i$: predicted − measured</span>
        <span>$N$</span><span>number of observations (a few points, or every pixel of an image)</span>
        <span>$S$</span><span>total squared error; zero only if every prediction is exact</span>
      </div></div>
      <p>Squaring makes every error positive, punishes big errors more than small ones, and (the real reason) gives a smooth bowl whose bottom we can find with linear algebra.</p>
      <div class="note"><b>Worked example.</b> Line $y = a x + b$ with $a = 2$, $b = 1$; points $(0, 1.5)$, $(1, 2.5)$, $(2, 5.5)$. Predictions $1, 3, 5$. Residuals $1-1.5=-0.5$, $3-2.5=0.5$, $5-5.5=-0.5$. $S = 0.25+0.25+0.25 = 0.75$.</div>
    `);

    // ---- widget 1: line fit
    {
      const fig = L.figure(root, "<b>Fitting a line.</b> Drag the blue data points. The green line is the least-squares fit; the hollow handles set your own line. Can you beat the fit's $S$? (You can't: it is the minimum.)");
      const c = L.canvas(fig.el, { aspect: 0.6 });
      fig.add(c.el);
      const ro = L.readout(fig.el); fig.add(ro.el);
      const pts = [[0.8, 1.6], [2.0, 2.2], [3.1, 3.9], [4.3, 3.6], [5.6, 5.4], [7.0, 5.9]];
      const mine = [[0.5, 3.0], [7.5, 4.0]]; // two points defining the user's line
      let P;
      const fit = () => {
        let sx = 0, sy = 0, sxx = 0, sxy = 0; const n = pts.length;
        for (const [x, y] of pts) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
        const sol = la.solve([[sxx, sx], [sx, n]], [sxy, sy]) || [0, sy / n];
        return { a: sol[0], b: sol[1], sx, sy, sxx, sxy, n };
      };
      const sse = (a, b) => pts.reduce((s, [x, y]) => s + (a * x + b - y) ** 2, 0);
      c.draw = (ctx) => {
        const t = L.theme();
        P = L.plot(c, { x0: 0, x1: 8, y0: 0, y1: 8, pad: [10, 10, 26, 30] });
        P.axes(ctx, { xticks: 8, yticks: 4, fmt: (v) => v.toFixed(0) });
        const F = fit();
        const ma = (mine[1][1] - mine[0][1]) / (mine[1][0] - mine[0][0] || 1e-6), mb = mine[0][1] - ma * mine[0][0];
        // residual bars to the fit
        for (const [x, y] of pts) L.draw.line(ctx, P.X(x), P.Y(y), P.X(x), P.Y(F.a * x + F.b), t.accent4, 2);
        L.draw.line(ctx, P.X(0), P.Y(F.b), P.X(8), P.Y(F.a * 8 + F.b), t.good, 2.5);
        L.draw.line(ctx, P.X(0), P.Y(mb), P.X(8), P.Y(ma * 8 + mb), t.accent2, 2, [6, 5]);
        for (const [x, y] of mine) L.draw.dot(ctx, P.X(x), P.Y(y), 8, t.panel, t.accent2);
        for (const [x, y] of pts) L.draw.handle(ctx, P.X(x), P.Y(y), t.accent);
        ro.html = `fit: a = <b>${F.a.toFixed(3)}</b>, b = <b>${F.b.toFixed(3)}</b>, S = <b>${sse(F.a, F.b).toFixed(3)}</b><br>` +
          `yours (dashed): a = ${ma.toFixed(3)}, b = ${mb.toFixed(3)}, S = <b>${sse(ma, mb).toFixed(3)}</b><br>` +
          `normal equations: [${F.sxx.toFixed(2)} ${F.sx.toFixed(2)}; ${F.sx.toFixed(2)} ${F.n}]·[a b] = [${F.sxy.toFixed(2)} ${F.sy.toFixed(2)}]`;
      };
      L.drag(c, () => (P ? [...pts, ...mine].map(([x, y]) => ({ x: P.X(x), y: P.Y(y) })) : []), (i, p) => {
        const x = Math.max(0.1, Math.min(7.9, P.invX(p.x))), y = Math.max(0, Math.min(8, P.invY(p.y)));
        if (i < pts.length) pts[i] = [x, y];
        else {
          const k = i - pts.length;
          mine[k] = [x, y];
          if (Math.abs(mine[0][0] - mine[1][0]) < 0.3) mine[k][0] = mine[1 - k][0] + (k ? 0.3 : -0.3);
        }
      });
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <p>Recall from chapter 2: an overdetermined linear system $A\mathbf p \approx \mathbf b$ is solved in the least-squares sense by the <b>normal equations</b> $A^\top A\,\mathbf p = A^\top\mathbf b$. For a line, row $i$ of $A$ is $(x_i, 1)$ and $b_i = y_i$:</p>
      <div class="eq-card"><div class="eq-label">Line fit · normal equations written as sums</div>
      $$\begin{pmatrix}\sum x_i^2 & \sum x_i\\ \sum x_i & N\end{pmatrix}\begin{pmatrix}a\\ b\end{pmatrix} = \begin{pmatrix}\sum x_i y_i\\ \sum y_i\end{pmatrix}$$
      <div class="parts">
        <span>$A^\top A$</span><span>2×2, built by adding one small contribution per point</span>
        <span>$A^\top \mathbf b$</span><span>2-vector, also one contribution per point</span>
        <span>$(a, b)$</span><span>slope and offset that minimise $S$</span>
      </div></div>
      <div class="note"><b>Worked example.</b> Points $(0,1)$, $(1,3)$, $(2,4)$: $\sum x^2 = 5$, $\sum x = 3$, $N = 3$, $\sum xy = 0+3+8 = 11$, $\sum y = 8$. Solve $5a + 3b = 11$, $3a + 3b = 8$: subtract → $2a = 3$, $a = 1.5$, $b = (8 - 4.5)/3 \approx 1.167$.</div>
      <p>That worked because the residual $ax_i + b - y_i$ is <b>linear</b> in the unknowns. Most DTAM residuals are not (they go through a projection and an image lookup). The fix: pretend they are linear, near where we are now.</p>

      <h3>Linearising: the first-order Taylor approximation</h3>
      <div class="eq-card"><div class="eq-label">Tangent-line approximation</div>
      $$f(x + \delta) \approx f(x) + f'(x)\,\delta$$
      <div class="parts">
        <span>$x$</span><span>where we are now</span>
        <span>$\delta$</span><span>a small change we are considering</span>
        <span>$f'(x)$</span><span>the slope at $x$ (derivative)</span>
        <span>$\approx$</span><span>good for small $\delta$; the error grows roughly like $\delta^2$</span>
      </div></div>
      <div class="note"><b>Worked example.</b> $f(x) = x^2$ at $x = 3$: $f = 9$, $f' = 6$. For $\delta = 0.1$: approx $9 + 0.6 = 9.6$; true $3.1^2 = 9.61$. For $\delta = 1$: approx $15$, true $16$: the error grew 100×.</div>
    `);

    // ---- widget 2: Taylor
    {
      const fig = L.figure(root, "<b>Tangent line.</b> Drag the point along the curve and move the $\\delta$ slider. The approximation is excellent near the point and drifts away further out.");
      const c = L.canvas(fig.el, { aspect: 0.55 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ro = L.readout(fig.el); fig.add(ro.el);
      const f = (x) => 1.2 + Math.sin(1.1 * x) + 0.15 * x;
      const df = (x) => 1.1 * Math.cos(1.1 * x) + 0.15;
      let x0 = 1.8, P;
      const sd = L.slider(ctl, { label: "$\\delta$", min: -3, max: 3, step: 0.05, value: 1, oninput: () => c.redraw() });
      c.draw = (ctx) => {
        const t = L.theme();
        P = L.plot(c, { x0: -1, x1: 7, y0: -1, y1: 3.5, pad: [10, 10, 26, 30] });
        P.axes(ctx, { xticks: 8, yticks: 3, fmt: (v) => v.toFixed(1) });
        const curve = [];
        for (let i = 0; i <= 200; i++) { const x = -1 + (8 * i) / 200; curve.push([P.X(x), P.Y(f(x))]); }
        L.draw.path(ctx, curve, t.accent, 2.5);
        const s = df(x0), y0 = f(x0);
        L.draw.line(ctx, P.X(-1), P.Y(y0 + s * (-1 - x0)), P.X(7), P.Y(y0 + s * (7 - x0)), t.accent2, 1.5, [6, 5]);
        const d = sd.value, xt = x0 + d, approx = y0 + s * d, tru = f(xt);
        L.draw.line(ctx, P.X(xt), P.Y(approx), P.X(xt), P.Y(tru), t.bad, 3);
        L.draw.dot(ctx, P.X(xt), P.Y(tru), 5, t.accent);
        L.draw.dot(ctx, P.X(xt), P.Y(approx), 5, t.accent2);
        L.draw.handle(ctx, P.X(x0), P.Y(y0), t.accent);
        ro.html = `x = ${x0.toFixed(2)}, f(x) = ${y0.toFixed(3)}, f'(x) = ${s.toFixed(3)}<br>` +
          `f(x+δ) ≈ ${y0.toFixed(3)} + ${s.toFixed(3)}·${d.toFixed(2)} = <b>${approx.toFixed(3)}</b>; true = <b>${tru.toFixed(3)}</b>; error = <b style="color:var(--bad)">${(tru - approx).toFixed(3)}</b>`;
      };
      L.drag(c, () => (P ? [{ x: P.X(x0), y: P.Y(f(x0)) }] : []), (i, p) => { x0 = Math.max(-0.5, Math.min(6.5, P.invX(p.x))); });
    }

    // ================================================================ 3. Gauss–Newton 1D
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Gauss–Newton with one unknown</h3>
      <p>Replace every residual by its tangent line, $r_i(x+\delta) \approx r_i + J_i\,\delta$ with $J_i = r_i'(x)$. The cost becomes a parabola in $\delta$, and a parabola's bottom has a formula:</p>
      <div class="eq-card"><div class="eq-label">One Gauss–Newton step (1 unknown)</div>
      $$S(x+\delta) \approx \sum_i (r_i + J_i\delta)^2 \quad\Rightarrow\quad \delta = -\frac{\sum_i J_i r_i}{\sum_i J_i^2}$$
      <div class="parts">
        <span>$r_i$</span><span>residuals at the current $x$</span>
        <span>$J_i$</span><span>slope of residual $i$ with respect to $x$</span>
        <span>$\sum J_i^2$</span><span>curvature of the parabola ("$H$")</span>
        <span>$\sum J_i r_i$</span><span>half the slope of the cost ("$g$")</span>
        <span>$\delta$</span><span>jump to the parabola's minimum; then set $x \leftarrow x + \delta$ and repeat</span>
      </div></div>
      <p>Why: the derivative of $\sum(r_i + J_i\delta)^2$ is $2\sum J_i(r_i + J_i\delta)$; set it to zero. With a single residual this is Newton's root-finding step $\delta = -r/J$.</p>
      <div class="note"><b>Worked example.</b> Residuals $r = (0.2, -0.1, 0.4)$, slopes $J = (1, 2, -1)$. $\sum J r = 0.2 - 0.2 - 0.4 = -0.4$, $\sum J^2 = 6$, $\delta = 0.4/6 \approx 0.067$.</div>
      <p>Below is the problem the KLT tracker (chapter 6) and dense tracking (chapter 11) really solve, shrunk to 1D: slide a signal by $x$ until it matches a template. Residual $i$ is $I(i + x) - T(i)$, and its slope $J_i = I'(i + x)$ is the <b>image gradient</b>.</p>
    `);

    // ---- widget 3: 1D alignment with GN / LM, basin, blur
    {
      const fig = L.figure(root, "<b>1D alignment.</b> Top: template (grey) and the moving signal slid by the current $x$ (blue). Bottom: the cost $S(x)$, and the dashed parabola Gauss–Newton believes in. Drag the handle to pick a start, press <b>Step</b>. Start far away to fall into a wrong valley; then raise the blur or press <b>Coarse→fine</b>.");
      const c = L.canvas(fig.el, { aspect: 0.8 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ctl2 = L.controls(fig.el); fig.add(ctl2);
      const ro = L.readout(fig.el); fig.add(ro.el);
      const bumps = [[40, 5, 0.7], [58, 2.5, -0.5], [72, 3, 0.55], [88, 2.5, 0.6], [104, 3, -0.55], [118, 2.5, 0.6], [134, 4, -0.5], [150, 2.5, 0.65], [165, 5, -0.4], [100, 30, 0.25]];
      const TRUE = 18; // I(x) = T(x - TRUE): the answer is x = TRUE
      // Blurring a Gaussian bump of width w by σ gives width √(w²+σ²), height scaled by w/√(w²+σ²).
      let prepBlur = -1, prepB = null, prepT = null;
      const prep = (blur) => {
        if (blur === prepBlur) return;
        prepB = bumps.map(([c0, w0, a0]) => { const w = Math.sqrt(w0 * w0 + blur * blur); return [c0, 1 / (w * w), (a0 * w0) / w]; });
        prepBlur = blur;
        prepT = new Float64Array(I1 + 1);
        for (let i = I0; i <= I1; i++) prepT[i] = sig(i, blur, 0)[0];
      };
      const sig = (x, blur, shift) => {
        prep(blur);
        let v = 0, dv = 0;
        for (const [c0, iw2, a] of prepB) {
          const u = x - shift - c0;
          const e = a * Math.exp(-0.5 * u * u * iw2);
          v += e; dv -= u * iw2 * e;
        }
        return [v, dv];
      };
      const I0 = 30, I1 = 170;
      const evalAt = (x, blur) => {
        prep(blur);
        let S = 0, H = 0, g = 0;
        for (let i = I0; i <= I1; i++) {
          const [iv, id] = sig(i + x, blur, TRUE);
          const r = iv - prepT[i];
          S += r * r; H += id * id; g += id * r;
        }
        return { S, H, g };
      };
      let x = -12, hist = [], msg = "", curve = null, curveBlur = -1, timer = 0;
      const blur = L.slider(ctl, { label: "blur σ", min: 0, max: 16, step: 0.5, value: 0, oninput: () => { reset(); } });
      const reset = () => { hist = [x]; msg = ""; c.redraw(); };
      const step = () => {
        const b = blur.value, e = evalAt(x, b);
        const d = -e.g / e.H;
        x += d; hist.push(x);
        msg = `step δ = −g/H = ${(-e.g).toFixed(3)}/${e.H.toFixed(3)} = ${d.toFixed(2)}`;
        c.redraw();
      };
      L.button(ctl2, "Step", step, "btn primary");
      L.button(ctl2, "Run ×10", () => { for (let k = 0; k < 10; k++) step(); });
      L.button(ctl2, "Coarse→fine", () => {
        const my = ++timer;
        const levels = [12, 6, 3, 0];
        let li = 0, it = 0;
        blur.value = levels[0]; reset();
        const tick = () => {
          if (li >= levels.length || my !== timer) return;
          step(); it++;
          if (it >= 5) { li++; it = 0; if (li < levels.length) { blur.value = levels[li]; hist = [x]; } }
          setTimeout(tick, 250);
        };
        tick();
      });
      L.button(ctl2, "Reset", () => { timer++; x = -12; blur.value = 0; reset(); });
      reset();
      let P2;
      c.draw = (ctx) => {
        const t = L.theme(), b = blur.value;
        const top = c.h * 0.38;
        // top plot: signals
        const P1 = L.plot({ w: c.w, h: top }, { x0: 20, x1: 180, y0: -0.8, y1: 1.1, pad: [8, 10, 6, 30] });
        const tp = [], mp = [];
        for (let i = 20; i <= 180; i += 1) { tp.push([P1.X(i), P1.Y(sig(i, b, 0)[0])]); mp.push([P1.X(i), P1.Y(sig(i + x, b, TRUE)[0])]); }
        L.draw.line(ctx, P1.X(I0), 4, P1.X(I0), top, t.line, 1);
        L.draw.line(ctx, P1.X(I1), 4, P1.X(I1), top, t.line, 1);
        L.draw.path(ctx, tp, t.faint, 3);
        L.draw.path(ctx, mp, t.accent, 2);
        L.draw.text(ctx, "T(i)", P1.X(22), 16, t.muted, { size: 12 });
        L.draw.text(ctx, "I(i + x)", P1.X(22), 30, t.accent, { size: 12 });
        // bottom: cost curve
        const X0 = -40, X1 = 70;
        if (curveBlur !== b || !curve) {
          curve = [];
          for (let k = 0; k <= 110; k++) { const xx = X0 + ((X1 - X0) * k) / 110; curve.push([xx, evalAt(xx, b).S]); }
          curveBlur = b;
        }
        const ymax = Math.max(...curve.map((q) => q[1])) * 1.1;
        ctx.save(); ctx.translate(0, top);
        const Pc = L.plot({ w: c.w, h: c.h - top }, { x0: X0, x1: X1, y0: 0, y1: ymax, pad: [12, 10, 26, 30] });
        P2 = { X: Pc.X, Y: (y) => Pc.Y(y) + top, invX: Pc.invX };
        Pc.axes(ctx, { xticks: 11, yticks: 2, fmt: (v) => (Math.abs(v) > 5 ? v.toFixed(0) : +v.toFixed(1)) });
        L.draw.path(ctx, curve.map(([xx, s]) => [Pc.X(xx), Pc.Y(s)]), t.fg, 2);
        L.draw.line(ctx, Pc.X(TRUE), Pc.Y(0), Pc.X(TRUE), Pc.Y(ymax), t.good, 1.5, [3, 4]);
        L.draw.text(ctx, "x", c.w - 14, c.h - top - 6, t.muted, { size: 12 });
        const e = evalAt(x, b);
        // GN parabola: S + 2 g d + H d^2
        const para = [];
        for (let k = -60; k <= 60; k++) {
          const d = k * 0.5, s = e.S + 2 * e.g * d + e.H * d * d;
          if (s <= ymax && x + d >= X0 && x + d <= X1) para.push([Pc.X(x + d), Pc.Y(Math.max(0, s))]);
        }
        L.draw.path(ctx, para, t.accent2, 1.5, [5, 4]);
        for (let k = 1; k < hist.length; k++) {
          const a = hist[k - 1], bb = hist[k];
          const ya = Pc.Y(Math.min(ymax, evalAt(a, b).S)), yb = Pc.Y(Math.min(ymax, evalAt(bb, b).S));
          if (a >= X0 && a <= X1 && bb >= X0 && bb <= X1) L.draw.line(ctx, Pc.X(a), ya, Pc.X(bb), yb, t.accent4, 1.5);
        }
        const xc = Math.max(X0, Math.min(X1, x));
        ctx.restore();
        L.draw.handle(ctx, P2.X(xc), P2.Y(Math.min(ymax, e.S)), t.accent);
        ro.html = `x = <b>${x.toFixed(2)}</b> (true 18), blur ${b}, S = ${e.S.toFixed(3)}, H = ΣJ² = ${e.H.toFixed(3)}, g = ΣJr = ${e.g.toFixed(3)}` +
          `<br>${msg || "Press Step."}`;
      };
      L.drag(c, () => (P2 ? [{ x: P2.X(Math.max(-40, Math.min(70, x))), y: P2.Y(Math.min(1e9, evalAt(x, blur.value).S)) }] : []),
        (i, p) => { timer++; x = Math.max(-40, Math.min(70, P2.invX(p.x))); reset(); }, { radius: 26 });
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <div class="key"><b>Key idea: the basin of convergence.</b> Gauss–Newton only sees the local slope. From a start inside the valley around the true answer it converges in a few steps; from outside it slides into a different valley (a <b>local minimum</b>) and happily stops there. The set of good starts is the <b>basin</b>. Everything later in this chapter is about making the basin wider or the steps safer.</div>

      <h3>Many unknowns: the Jacobian</h3>
      <p>With $n$ unknowns $\mathbf p = (p_1,\dots,p_n)$ each residual has one slope per unknown. Stack them into the <b>Jacobian</b> $J$: one row per residual, one column per unknown.</p>
      <div class="eq-card"><div class="eq-label">Jacobian and linearised residuals</div>
      $$J_{ij} = \frac{\partial r_i}{\partial p_j}, \qquad \mathbf r(\mathbf p + \boldsymbol\delta) \approx \mathbf r(\mathbf p) + J\boldsymbol\delta$$
      <div class="parts">
        <span>$\partial r_i/\partial p_j$</span><span>slope of residual $i$ when only $p_j$ changes (others held fixed)</span>
        <span>$J$</span><span>$N\times n$ matrix; row $i$ is $J_i$, the "gradient" of residual $i$</span>
        <span>$\boldsymbol\delta$</span><span>the update to all unknowns at once</span>
      </div></div>
      <p>Minimising $\|\mathbf r + J\boldsymbol\delta\|^2$ is a linear least-squares problem: chapter 2's normal equations with $A = J$ and $\mathbf b = -\mathbf r$.</p>
      <div class="eq-card"><div class="eq-label">Gauss–Newton step · paper §2.3.1 solves exactly this</div>
      $$\underbrace{J^\top J}_{H}\,\boldsymbol\delta = -\underbrace{J^\top\mathbf r}_{\mathbf g}, \qquad \mathbf p \leftarrow \mathbf p + \boldsymbol\delta$$
      <div class="parts">
        <span>$H = J^\top J$</span><span>$n\times n$, symmetric, "bowl-shaped" (positive definite) when the data pin down every unknown</span>
        <span>$\mathbf g = J^\top \mathbf r$</span><span>$n$-vector; half the gradient of $S$</span>
        <span>$\boldsymbol\delta$</span><span>solve the $n\times n$ system (small: $n = 2$ for KLT, $6$ for a pose)</span>
      </div></div>
      <p>The paper writes the tracking cost as $F(\psi) = \tfrac12\sum f_\mathbf u(\psi)^2$ (eq. 19). The $\tfrac12$ only tidies the derivative; the step is identical.</p>

      <h3>Sums over pixels</h3>
      <p>Look at what $J^\top J$ and $J^\top\mathbf r$ are, entry by entry:</p>
      <div class="eq-card"><div class="eq-label">Accumulating the normal equations</div>
      $$H = \sum_{i} J_i^\top J_i, \qquad \mathbf g = \sum_i J_i^\top r_i$$
      <div class="parts">
        <span>$J_i^\top J_i$</span><span>an $n\times n$ "outer product": entry $(j,k)$ is $J_{ij}J_{ik}$</span>
        <span>$J_i^\top r_i$</span><span>row $i$ scaled by its residual</span>
        <span>$\sum_i$</span><span>each residual (pixel) adds its own piece; order doesn't matter</span>
      </div></div>
      <p>We never build the huge $N\times n$ matrix $J$. For a 640×480 image with 6 unknowns, each pixel adds 21 numbers (the upper triangle of symmetric $H$) plus 6 for $\mathbf g$. On a GPU every pixel computes its piece in parallel and a <b>reduction</b> adds them up (this repo's <code>track6.wgsl</code>: each of 128 workgroups sums its pixels, then <code>gn.wgsl</code> adds the 128 partial sums and solves the 6×6 system).</p>
      <div class="note"><b>Worked example.</b> Two unknowns, three residuals: $J_1 = (1, 0)$, $r_1 = 1$; $J_2 = (0, 2)$, $r_2 = -2$; $J_3 = (1, 1)$, $r_3 = 0$.<br>
      $H = \begin{pmatrix}1&0\\0&0\end{pmatrix} + \begin{pmatrix}0&0\\0&4\end{pmatrix} + \begin{pmatrix}1&1\\1&1\end{pmatrix} = \begin{pmatrix}2&1\\1&5\end{pmatrix}$, $\ \mathbf g = (1,0) + (0,-4) + (0,0) = (1,-4)$.<br>
      Solve $H\boldsymbol\delta = (-1, 4)$: $\det = 9$, $\boldsymbol\delta = \frac19(5\cdot(-1) - 1\cdot 4,\ -1\cdot(-1) + 2\cdot 4) = (-1, 1)$.</div>
      <pre><code>// Gauss–Newton (what every tracker in this repo does)
p = initial guess
repeat until |δ| tiny or max iterations:
    H = 0 (n×n), g = 0 (n)
    for each residual i:                 // each pixel, in parallel on the GPU
        r_i = residual(i, p)
        J_i = d r_i / d p                // 1×n row
        H += J_iᵀ J_i ;  g += J_iᵀ r_i
    solve H δ = −g                       // tiny n×n system
    p = p + δ</code></pre>
      <p>Try it on a 2-unknown problem: locate a phone at $\mathbf p = (x, y)$ from measured distances $d_i$ to towers $\mathbf a_i$. Residual $r_i = \|\mathbf p - \mathbf a_i\| - d_i$; its Jacobian row is the unit vector from the tower to $\mathbf p$: $J_i = (\mathbf p - \mathbf a_i)^\top/\|\mathbf p - \mathbf a_i\|$.</p>
    `);

    // ---- widget 4: trilateration GN in 2D
    {
      const fig = L.figure(root, "<b>Gauss–Newton in 2D.</b> Drag the towers (squares), the true position (star) and the start (blue). Background: the cost $S$ (bright = low). Press <b>Step</b>. With 2 towers there are two exact answers (mirror images): which one you get depends on the basin you start in. With 2 towers, start near the line through them: plain Gauss–Newton can shoot off; tick <b>LM</b> to see rejected steps.");
      const c = L.canvas(fig.el, { aspect: 0.62 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ro = L.readout(fig.el); fig.add(ro.el);
      const W = 10, Hh = 6.2;
      const towers = [[2, 1.5], [8, 2], [5, 5.2]];
      let truth = [6.2, 3.6], start = [1.2, 5.2], p = start.slice(), path = [start.slice()], nT = 3;
      let last = null, mu = 1e-4;
      L.toggle(ctl, "only 2 towers", false, (v) => { nT = v ? 2 : 3; resetP(); });
      const lm = L.toggle(ctl, "LM safeguard", false, () => resetP());
      L.button(ctl, "Step", () => {
        const T = towers.slice(0, nT);
        const H = [[0, 0], [0, 0]], g = [0, 0], rows = [];
        for (const a of T) {
          const d = Math.hypot(truth[0] - a[0], truth[1] - a[1]);
          const dx = p[0] - a[0], dy = p[1] - a[1], rho = Math.max(1e-6, Math.hypot(dx, dy));
          const r = rho - d, J = [dx / rho, dy / rho];
          rows.push({ r, J });
          H[0][0] += J[0] * J[0]; H[0][1] += J[0] * J[1]; H[1][1] += J[1] * J[1];
          g[0] += J[0] * r; g[1] += J[1] * r;
        }
        H[1][0] = H[0][1];
        const damp = lm.checked ? 1 + mu : 1;
        const del = la.solve([[H[0][0] * damp, H[0][1]], [H[1][0], H[1][1] * damp]], [-g[0], -g[1]]);
        last = { rows, H, g, del, note: "" };
        if (del) {
          const q = [p[0] + del[0], p[1] + del[1]], s0 = cost(p[0], p[1]), s1 = cost(q[0], q[1]);
          if (!lm.checked || s1 <= s0) {
            p = q; path.push(p.slice());
            if (lm.checked) { mu = Math.max(mu / 3, 1e-7); last.note = `<br>S ${s0.toFixed(3)} → ${s1.toFixed(3)}: <b style="color:var(--good)">accept</b>, μ → ${mu.toExponential(1)}`; }
            else if (s1 > s0) last.note = `<br><b style="color:var(--bad)">S went up</b> (${s0.toFixed(3)} → ${s1.toFixed(3)})`;
          } else {
            mu *= 10;
            last.note = `<br>S would go ${s0.toFixed(3)} → ${s1.toFixed(3)}: <b style="color:var(--bad)">reject</b>, μ → ${mu.toExponential(1)}`;
          }
        }
        c.redraw();
      }, "btn primary");
      L.button(ctl, "Reset", () => resetP());
      const resetP = () => { p = start.slice(); path = [start.slice()]; last = null; mu = 1e-4; c.redraw(); };
      let S2C, C2S;
      const cost = (x, y) => {
        let s = 0;
        for (const a of towers.slice(0, nT)) {
          const d = Math.hypot(truth[0] - a[0], truth[1] - a[1]);
          s += (Math.hypot(x - a[0], y - a[1]) - d) ** 2;
        }
        return s;
      };
      c.draw = (ctx) => {
        const t = L.theme();
        const sc = Math.min(c.w / W, c.h / Hh), ox = (c.w - W * sc) / 2, oy = (c.h - Hh * sc) / 2;
        S2C = (q) => ({ x: ox + q[0] * sc, y: oy + q[1] * sc });
        C2S = (q) => [(q.x - ox) / sc, (q.y - oy) / sc];
        const gx = 60, gy = Math.round((60 * Hh) / W);
        const vals = [];
        let mx = 0;
        for (let j = 0; j < gy; j++) for (let i = 0; i < gx; i++) {
          const v = Math.sqrt(cost(((i + 0.5) / gx) * W, ((j + 0.5) / gy) * Hh));
          vals.push(v); mx = Math.max(mx, v);
        }
        for (let j = 0; j < gy; j++) for (let i = 0; i < gx; i++) {
          ctx.fillStyle = L.viridisish(1 - Math.pow(vals[j * gx + i] / mx, 0.5));
          ctx.fillRect(ox + (i * W * sc) / gx, oy + (j * Hh * sc) / gy, (W * sc) / gx + 1, (Hh * sc) / gy + 1);
        }
        for (const a of towers.slice(0, nT)) {
          const q = S2C(a), d = Math.hypot(truth[0] - a[0], truth[1] - a[1]);
          ctx.save(); ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.setLineDash([4, 4]);
          ctx.beginPath(); ctx.arc(q.x, q.y, d * sc, 0, 7); ctx.stroke(); ctx.restore();
        }
        L.draw.path(ctx, path.map((q) => { const s = S2C(q); return [s.x, s.y]; }), "#fff", 2);
        for (const q of path) { const s = S2C(q); L.draw.dot(ctx, s.x, s.y, 3, "#fff"); }
        towers.forEach((a, k) => {
          const q = S2C(a);
          ctx.save(); ctx.globalAlpha = k < nT ? 1 : 0.3;
          ctx.fillStyle = t.accent2; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
          ctx.fillRect(q.x - 8, q.y - 8, 16, 16); ctx.strokeRect(q.x - 8, q.y - 8, 16, 16); ctx.restore();
        });
        const tq = S2C(truth);
        L.draw.text(ctx, "★", tq.x, tq.y + 7, "#fff", { size: 22, align: "center" });
        const sq = S2C(start);
        L.draw.handle(ctx, sq.x, sq.y, t.accent);
        const pq = S2C(p);
        L.draw.dot(ctx, pq.x, pq.y, 6, t.accent4, "#fff");
        let h = `p = (${p[0].toFixed(3)}, ${p[1].toFixed(3)}), S = <b>${cost(p[0], p[1]).toFixed(4)}</b>, steps: ${path.length - 1}`;
        if (last) {
          h += "<br>" + last.rows.map((q, k) => `tower ${k + 1}: r = ${q.r.toFixed(3)}, J = (${q.J[0].toFixed(3)}, ${q.J[1].toFixed(3)})`).join("<br>");
          h += `<br>H = [${last.H[0][0].toFixed(3)} ${last.H[0][1].toFixed(3)}; ${last.H[1][0].toFixed(3)} ${last.H[1][1].toFixed(3)}], g = (${last.g[0].toFixed(3)}, ${last.g[1].toFixed(3)})`;
          h += last.del ? `, δ = (${last.del[0].toFixed(3)}, ${last.del[1].toFixed(3)})` : ", H singular: no step";
          h += last.note;
        }
        ro.html = h;
      };
      L.drag(c, () => (S2C ? [...towers.map(S2C), S2C(truth), S2C(start)] : []), (i, q) => {
        const s = C2S(q);
        s[0] = Math.max(0.1, Math.min(W - 0.1, s[0])); s[1] = Math.max(0.1, Math.min(Hh - 0.1, s[1]));
        if (i < 3) towers[i] = s; else if (i === 3) truth = s; else start = s;
        p = start.slice(); path = [start.slice()]; last = null; mu = 1e-4;
      });
    }

    // ================================================================ LM
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Levenberg–Marquardt: steps that never make things worse</h3>
      <p>Gauss–Newton trusts its parabola completely. Far from the answer the parabola is wrong and the step can overshoot so the cost goes <b>up</b>. Levenberg–Marquardt (LM) adds a <b>damping</b> factor $\mu$ and checks every step.</p>
      <div class="eq-card"><div class="eq-label">Damped normal equations (this repo's form)</div>
      $$\big(H + \mu\,\mathrm{diag}(H)\big)\,\boldsymbol\delta = -\mathbf g$$
      <div class="parts">
        <span>$\mu \approx 0$</span><span>plain Gauss–Newton step</span>
        <span>$\mu$ large</span><span>each diagonal entry is multiplied by $1+\mu$: a short step, roughly downhill (gradient descent)</span>
        <span>$\mathrm{diag}(H)$</span><span>only the diagonal entries $H_{jj}$; so this multiplies each $H_{jj}$ by $(1+\mu)$</span>
      </div></div>
      <p>The accept/reject rule:</p>
      <ul>
        <li>Solve for $\boldsymbol\delta$, evaluate the cost at $\mathbf p + \boldsymbol\delta$.</li>
        <li>Cost did not increase → <b>accept</b>: move there, make $\mu$ smaller (trust the parabola more).</li>
        <li>Cost increased → <b>reject</b>: stay put, make $\mu$ larger, solve again from the same $H$, $\mathbf g$.</li>
      </ul>
      <p>This repo (<code>gn.wgsl</code>): $\mu$ starts at $10^{-4}$; accept → $\mu \leftarrow \max(\mu/3,\ 10^{-7})$; reject → $\mu \leftarrow 10\mu$; give up on the level when $\mu > 10^4$.</p>
      <div class="note"><b>Worked example (1 unknown).</b> $H = 4$, $g = -2$. GN: $\delta = 2/4 = 0.5$. Suppose the cost rises. Reject; with $\mu = 1$: $\delta = 2/(4\cdot 2) = 0.25$, half the step. With $\mu = 9$: $\delta = 2/40 = 0.05$.</div>
      <pre><code>// Gauss–Newton with an LM safeguard (gn.wgsl, per pyramid level)
μ = 1e-4
(E, H, g) = evaluate(p)                    // cost + normal equations
repeat for the level's iteration budget:
    solve (H with every H_jj × (1+μ)) δ = −g
    (E', H', g') = evaluate(p + δ)
    if E' ≤ E:  p = p + δ; (E,H,g) = (E',H',g'); μ = max(μ/3, 1e-7)
    else:       μ = 10 μ;  if μ > 1e4: stop    // p, H, g unchanged
    if |δ| < 1e-6: stop</code></pre>
      <p>Go back to the 2D widget: tick <b>only 2 towers</b>, drag the start close to the line through the two towers and press Step. Plain Gauss–Newton can jump far away (the parabola is nearly flat in one direction). With <b>LM safeguard</b> ticked, bad steps are rejected and $\mu$ grows until a short, safe step is accepted.</p>

      <h3>Robust costs: don't let outliers vote</h3>
      <p>Squares grow fast: one residual of 10 costs as much as 100 residuals of 1. In tracking, pixels on a hand waving in front of the camera, or on a shadow, are <b>outliers</b>: the model can't explain them, and with plain squares they drag the answer away. Replace $r^2$ by a function $\rho(r)$ that grows more slowly:</p>
      <div class="eq-card"><div class="eq-label">Truncated quadratic · what dense tracking uses (paper §2.3.2)</div>
      $$\rho_\tau(r) = \min(r^2,\ \tau^2)$$
      <div class="parts">
        <span>$|r| \le \tau$</span><span>ordinary square: the pixel is used</span>
        <span>$|r| > \tau$</span><span>constant $\tau^2$: the pixel is ignored (no pull at all)</span>
        <span>$\tau$</span><span>the outlier threshold. This repo: $0.25, 0.18, 0.12, 0.09$ from coarse to fine level</span>
      </div></div>
      <div class="eq-card"><div class="eq-label">Paper eq. (4) · Huber norm</div>
      $$\|r\|_\epsilon = \begin{cases} \dfrac{r^2}{2\epsilon} & |r| \le \epsilon\\[4pt] |r| - \dfrac{\epsilon}{2} & \text{otherwise}\end{cases}$$
      <div class="parts">
        <span>$|r| \le \epsilon$</span><span>quadratic (smooth near zero)</span>
        <span>$|r| > \epsilon$</span><span>linear: large errors still count, but only proportionally</span>
        <span>$\epsilon/2$</span><span>makes the two pieces meet smoothly at $|r| = \epsilon$ (both equal $\epsilon/2$, same slope 1)</span>
      </div></div>
      <div class="note"><b>Worked example.</b> $\epsilon = 0.2$. $r = 0.1$: $0.01/0.4 = 0.025$. $r = 0.5$: $0.5 - 0.1 = 0.4$. $r = -3$: $3 - 0.1 = 2.9$ (a square would give 9). DTAM uses Huber again in chapter 9, on depth gradients.</div>
      <p><b>How to minimise a robust cost with Gauss–Newton:</b> give every residual a <b>weight</b> $w_i$ and use $H = \sum w_i J_i^\top J_i$, $\mathbf g = \sum w_i J_i^\top r_i$. Recompute weights each iteration ("iteratively reweighted least squares"):</p>
      <table class="mat"><tr><td>cost</td><td>weight $w(r)$</td></tr>
        <tr><td>squares $r^2$</td><td>1</td></tr>
        <tr><td>truncated $\min(r^2,\tau^2)$</td><td>1 if $|r|\le\tau$, else 0</td></tr>
        <tr><td>Huber</td><td>1 if $|r|\le\epsilon$, else $\epsilon/|r|$</td></tr></table>
      <p>(The Huber weight is relative to the inliers' weight; multiplying all weights by a constant doesn't change $\boldsymbol\delta$.)</p>
    `);

    // ---- widget 5: robust line fit
    {
      const fig = L.figure(root, "<b>Robust fitting.</b> Drag the two red outliers. Switch the cost. Dot size = the point's weight in the final iteration. Squares get dragged away; Huber resists; truncated ignores the outliers completely (if it starts close enough).");
      const c = L.canvas(fig.el, { aspect: 0.6 });
      fig.add(c.el);
      const c2 = L.canvas(fig.el, { aspect: 0.3, scroll: true });
      fig.add(c2.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ro = L.readout(fig.el); fig.add(ro.el);
      const inl = [[0.5, 1.2], [1.4, 1.5], [2.3, 2.1], [3.1, 2.3], [4.0, 2.9], [4.9, 3.1], [5.8, 3.7], [6.7, 3.9], [7.5, 4.6]];
      const out = [[6.2, 0.4], [7.2, 0.9]];
      let mode = "sq";
      const thr = L.slider(ctl, { label: "$\\epsilon$ / $\\tau$", min: 0.1, max: 3, step: 0.05, value: 0.6, oninput: () => { c.redraw(); c2.redraw(); } });
      const modeBtns = {};
      for (const [k, lab] of [["sq", "squares"], ["huber", "Huber"], ["trunc", "truncated"]]) {
        modeBtns[k] = L.button(ctl, lab, () => { mode = k; for (const q in modeBtns) modeBtns[q].className = "btn" + (q === mode ? " primary" : ""); c.redraw(); c2.redraw(); });
      }
      modeBtns.sq.className = "btn primary";
      const weight = (r, m, e) => (m === "sq" ? 1 : m === "trunc" ? (Math.abs(r) <= e ? 1 : 0) : Math.abs(r) <= e ? 1 : e / Math.abs(r));
      const irls = () => {
        const all = [...inl, ...out];
        let a = 0, b = 0, ws = all.map(() => 1);
        for (let it = 0; it < 30; it++) {
          let sxx = 0, sx = 0, sw = 0, sxy = 0, sy = 0;
          all.forEach(([x, y], k) => { const w = ws[k]; sxx += w * x * x; sx += w * x; sw += w; sxy += w * x * y; sy += w * y; });
          const sol = la.solve([[sxx, sx], [sx, sw]], [sxy, sy]);
          if (!sol) break;
          [a, b] = sol;
          ws = all.map(([x, y]) => weight(a * x + b - y, mode, thr.value));
        }
        return { a, b, ws };
      };
      let P;
      c.draw = (ctx) => {
        const t = L.theme();
        P = L.plot(c, { x0: 0, x1: 8, y0: 0, y1: 6, pad: [10, 10, 26, 30] });
        P.axes(ctx, { xticks: 8, yticks: 3, fmt: (v) => v.toFixed(0) });
        const F = irls(), all = [...inl, ...out];
        L.draw.line(ctx, P.X(0), P.Y(0.9), P.X(8), P.Y(0.9 + 0.47 * 8), t.faint, 1.5, [3, 4]);
        L.draw.line(ctx, P.X(0), P.Y(F.b), P.X(8), P.Y(F.a * 8 + F.b), t.good, 2.5);
        all.forEach(([x, y], k) => {
          const isOut = k >= inl.length;
          if (isOut) L.draw.handle(ctx, P.X(x), P.Y(y), t.bad);
          L.draw.dot(ctx, P.X(x), P.Y(y), 2 + 5 * F.ws[k], isOut ? "#fff" : t.accent);
        });
        ro.html = `fit: a = <b>${F.a.toFixed(3)}</b>, b = <b>${F.b.toFixed(3)}</b> (dotted: the line the inliers came from, a = 0.47, b = 0.9)<br>outlier weights: ${F.ws.slice(inl.length).map((w) => w.toFixed(2)).join(", ")}`;
      };
      c2.draw = (ctx) => {
        const t = L.theme(), e = thr.value;
        const Q = L.plot(c2, { x0: -4, x1: 4, y0: 0, y1: 4, pad: [8, 10, 22, 30] });
        Q.axes(ctx, { xticks: 8, yticks: 2, fmt: (v) => v.toFixed(0) });
        const f = { sq: (r) => r * r, huber: (r) => (Math.abs(r) <= e ? r * r : 2 * e * Math.abs(r) - e * e), trunc: (r) => Math.min(r * r, e * e) };
        const cols = { sq: t.faint, huber: t.accent2, trunc: t.accent3 };
        for (const k of ["sq", "huber", "trunc"]) {
          const pts = [];
          for (let i = 0; i <= 160; i++) { const r = -4 + i / 20, v = f[k](r); if (v <= 4.2) pts.push([Q.X(r), Q.Y(Math.min(v, 4))]); else if (pts.length) { L.draw.path(ctx, pts, cols[k], k === mode ? 3 : 1.5); pts.length = 0; } }
          L.draw.path(ctx, pts, cols[k], k === mode ? 3 : 1.5);
        }
        L.draw.text(ctx, "ρ(r): grey r², orange Huber, green truncated", 36, 14, t.muted, { size: 11 });
        L.draw.text(ctx, "r", c2.w - 14, c2.h - 4, t.muted, { size: 12 });
      };
      L.drag(c, () => (P ? out.map(([x, y]) => ({ x: P.X(x), y: P.Y(y) })) : []), (i, p) => {
        out[i] = [Math.max(0.1, Math.min(7.9, P.invX(p.x))), Math.max(0, Math.min(6, P.invY(p.y)))];
      });
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <div class="warn"><b>Robust costs shrink the basin.</b> A truncated cost ignores anything beyond $\tau$, including correct pixels that are simply still misaligned at the start. That is why this repo uses a generous $\tau = 0.25$ on the coarsest level and ramps it down to $0.09$ as the alignment improves (paper §2.3.2: "ramps down this threshold as we converge").</div>
      <p>In dense tracking (chapter 11) the robust cost is the mean truncated square over the pixels that land inside the live image:</p>
      <div class="eq-card"><div class="eq-label">This repo's tracking cost (used for LM accept/reject)</div>
      $$E = \frac{1}{N_{\text{in view}}}\Big(\sum_{|r_\mathbf u| \le \tau} r_\mathbf u^2 \;+\; N_{\text{rejected}}\,\tau^2\Big)$$
      <div class="parts">
        <span>$N_{\text{in view}}$</span><span>pixels with model depth that project inside the live image</span>
        <span>$N_{\text{rejected}}$</span><span>of those, pixels with $|r| > \tau$; each counts $\tau^2$ but adds nothing to $H$, $\mathbf g$</span>
      </div></div>
      <div class="note"><b>Worked example.</b> $\tau = 0.1$, residuals $0.05, -0.02, 0.3, 0.08$: inliers give $0.0025 + 0.0004 + 0.0064 = 0.0093$; one rejected adds $0.01$. $E = 0.0193/4 \approx 0.0048$.</div>

      <h3>Coarse-to-fine: widening the basin</h3>
      <p>The linearisation $I(i + x + \delta) \approx I(i + x) + I'(i + x)\,\delta$ is only good while $\delta$ is small compared with the width of the image's features. Blur widens every feature, so the cost valley widens too (try the blur slider in the 1D widget). Image pyramids (chapter 1) do this and also halve the pixel count:</p>
      <ul>
        <li>Solve on the coarsest (most blurred, smallest) level first; a 16-pixel motion is only 2 pixels at level 3.</li>
        <li>Use that answer as the starting guess on the next finer level (scale pixel motions ×2), refine, repeat down to level 0.</li>
        <li>Fine levels add accuracy; coarse levels add basin width. Each level needs only a few iterations.</li>
      </ul>
      <pre><code>// Coarse-to-fine (KLT in chapter 6, dense tracking in chapter 11)
p = initial guess
for level = coarsest … 0:
    express p in this level's units    // pixel motions ×2 per finer level
    run Gauss–Newton / LM for a few iterations on this level
                                     // dense tracking here: 20, 20, 15, 10 iterations
return p</code></pre>
      <div class="key"><b>Summary.</b> Linearise residuals → normal equations $J^\top J\,\boldsymbol\delta = -J^\top\mathbf r$ built as sums over pixels → step → repeat. Guard steps with LM, down-weight outliers with a robust cost, and widen the basin coarse-to-fine.</div>
    `);

    // ================================================================ quiz
    const f2 = (v) => L.fmt(v, 2), f3 = (v) => L.fmt(v, 3), f4 = (v) => L.fmt(v, 4);
    L.quiz(root, "lsq", [
      { id: "sse", type: "num",
        gen: (r) => {
          const a = r.int(1, 3), b = r.int(-2, 2);
          const xs = [0, 1, 2], ys = xs.map((x) => a * x + b + r.pick([-1, -0.5, 0, 0.5, 1]));
          const res = xs.map((x, i) => a * x + b - ys[i]);
          const S = res.reduce((s, v) => s + v * v, 0);
          return { q: String.raw`Line $y = ${a}x ${b < 0 ? "-" : "+"} ${Math.abs(b)}$. Points $(0, ${ys[0]})$, $(1, ${ys[1]})$, $(2, ${ys[2]})$. What is the sum of squared residuals $S$?`,
            answer: S, tol: 1e-6,
            explain: String.raw`Predictions ${xs.map((x) => a * x + b).join(", ")}; residuals ${res.map(f2).join(", ")}; squares add to $${f4(S)}$.` };
        } },
      { id: "linefit", type: "num",
        gen: (r) => {
          const ys = [r.int(0, 3), r.int(2, 5), r.int(4, 8)];
          const sy = ys[0] + ys[1] + ys[2], sxy = ys[1] + 2 * ys[2];
          // 5a + 3b = sxy ; 3a + 3b = sy
          const a = (sxy - sy) / 2, b = (sy - 3 * a) / 3;
          return { q: String.raw`Fit $y = ax + b$ by least squares to $(0, ${ys[0]})$, $(1, ${ys[1]})$, $(2, ${ys[2]})$. Give $a$ and $b$ (to 3 decimals).`,
            answer: [a, b], labels: ["$a$", "$b$"], tol: 2e-3,
            explain: String.raw`$\sum x^2 = 5$, $\sum x = 3$, $N = 3$, $\sum xy = ${sxy}$, $\sum y = ${sy}$. Equations $5a + 3b = ${sxy}$, $3a + 3b = ${sy}$. Subtract: $2a = ${sxy - sy}$, $a = ${f3(a)}$; $b = (${sy} - 3\cdot${f3(a)})/3 = ${f3(b)}$.` };
        } },
      { id: "taylor", type: "num",
        gen: (r) => {
          const x = r.int(1, 4), d = r.pick([0.1, 0.2, -0.1, 0.3, -0.2]);
          const approx = x ** 3 + 3 * x * x * d;
          return { q: String.raw`$f(x) = x^3$. Using the first-order Taylor approximation at $x = ${x}$, estimate $f(${x} ${d < 0 ? "-" : "+"} ${Math.abs(d)})$.`,
            answer: approx, tol: 1e-6,
            explain: String.raw`$f(${x}) = ${x ** 3}$, $f'(x) = 3x^2 = ${3 * x * x}$. Estimate $${x ** 3} + ${3 * x * x}\cdot(${d}) = ${f4(approx)}$ (true value $${f4((x + d) ** 3)}$).` };
        } },
      { id: "newton1", type: "num",
        gen: (r) => {
          const cc = r.pick([2, 3, 5, 6, 7, 10]), x0 = r.pick([1, 2, 3]);
          const res = x0 * x0 - cc, J = 2 * x0, x1 = x0 - res / J;
          return { q: String.raw`Single residual $r(x) = x^2 - ${cc}$. Start at $x = ${x0}$. What is $x$ after one Gauss–Newton step? (to 4 decimals)`,
            answer: x1, tol: 1e-4,
            explain: String.raw`$r = ${res}$, $J = r'(x) = 2x = ${J}$, $\delta = -r/J = ${f4(-res / J)}$, so $x = ${f4(x1)}$.` };
        } },
      { id: "gn1d", type: "num",
        gen: (r) => {
          const J = [r.nz(3), r.nz(3), r.nz(3)], res = [r.float(-0.5, 0.5, 1), r.float(-0.5, 0.5, 1), r.float(-0.5, 0.5, 1)];
          const g = J.reduce((s, j, i) => s + j * res[i], 0), H = J.reduce((s, j) => s + j * j, 0), d = -g / H;
          return { q: String.raw`One unknown, three residuals: $r = (${res.join(", ")})$, slopes $J = (${J.join(", ")})$. What is the Gauss–Newton step $\delta$? (to 4 decimals)`,
            answer: d, tol: 1e-4,
            explain: String.raw`$\sum J r = ${f4(g)}$, $\sum J^2 = ${H}$, $\delta = -${f4(g)}/${H} = ${f4(d)}$.` };
        } },
      { id: "accum", type: "num",
        gen: (r) => {
          const J1 = [r.nz(2), r.int(-2, 2)], J2 = [r.int(-2, 2), r.nz(2)];
          const H = [J1[0] ** 2 + J2[0] ** 2, J1[0] * J1[1] + J2[0] * J2[1], J1[1] ** 2 + J2[1] ** 2];
          return { q: String.raw`Two pixels contribute to a 2-unknown problem with Jacobian rows $J_1 = (${J1.join(", ")})$ and $J_2 = (${J2.join(", ")})$. Give the entries of $H = J^\top J$.`,
            answer: H, labels: ["$H_{11}$", "$H_{12}$", "$H_{22}$"], tol: 1e-9,
            explain: String.raw`$H = J_1^\top J_1 + J_2^\top J_2$. $H_{11} = ${J1[0]}^2 + ${J2[0]}^2 = ${H[0]}$, $H_{12} = ${J1[0]}\cdot${J1[1]} + ${J2[0]}\cdot${J2[1]} = ${H[1]}$, $H_{22} = ${J1[1]}^2 + ${J2[1]}^2 = ${H[2]}$.` };
        } },
      { id: "gn2d", type: "num",
        gen: (r) => {
          let J, res, H, det;
          do {
            J = [[r.int(-2, 2), r.int(-2, 2)], [r.int(-2, 2), r.int(-2, 2)], [r.int(-2, 2), r.int(-2, 2)]];
            res = [r.int(-3, 3), r.int(-3, 3), r.int(-3, 3)];
            H = [[0, 0], [0, 0]];
            for (const [a, b] of J) { H[0][0] += a * a; H[0][1] += a * b; H[1][1] += b * b; }
            H[1][0] = H[0][1];
            det = H[0][0] * H[1][1] - H[0][1] ** 2;
          } while (det < 3);
          const g = [0, 1].map((k) => J.reduce((s, row, i) => s + row[k] * res[i], 0));
          const d = [(-(H[1][1] * g[0]) + H[0][1] * g[1]) / det, (H[0][1] * g[0] - H[0][0] * g[1]) / det];
          return { q: String.raw`Gauss–Newton with 2 unknowns. Jacobian rows $(${J[0].join(", ")})$, $(${J[1].join(", ")})$, $(${J[2].join(", ")})$; residuals $(${res.join(", ")})$. Solve $J^\top J\,\boldsymbol\delta = -J^\top\mathbf r$ (to 3 decimals).`,
            answer: d, labels: ["$\\delta_1$", "$\\delta_2$"], tol: 2e-3,
            explain: String.raw`$H = \begin{pmatrix}${H[0][0]}&${H[0][1]}\\${H[1][0]}&${H[1][1]}\end{pmatrix}$, $\mathbf g = J^\top\mathbf r = (${g.join(", ")})$. $\det H = ${det}$. $\boldsymbol\delta = -H^{-1}\mathbf g = -\frac{1}{${det}}\begin{pmatrix}${H[1][1]}&${-H[0][1]}\\${-H[0][1]}&${H[0][0]}\end{pmatrix}\begin{pmatrix}${g[0]}\\${g[1]}\end{pmatrix} = (${f3(d[0])}, ${f3(d[1])})$.` };
        } },
      { id: "trilat", type: "num",
        gen: (r) => {
          const tri = r.pick([[3, 4, 5], [6, 8, 10], [4, 3, 5], [5, 12, 13], [8, 6, 10]]);
          const ax = r.int(0, 3), ay = r.int(0, 3), sx = r.sign(), sy = r.sign();
          const px = ax + sx * tri[0], py = ay + sy * tri[1], d = tri[2] + r.pick([-1, 1, 2, -2, 0.5]);
          const res = tri[2] - d, J = [(px - ax) / tri[2], (py - ay) / tri[2]];
          return { q: String.raw`Tower at $\mathbf a = (${ax}, ${ay})$, measured distance $d = ${d}$, current guess $\mathbf p = (${px}, ${py})$. Residual $r = \|\mathbf p - \mathbf a\| - d$. Give $r$ and the Jacobian row $J = (\partial r/\partial x, \partial r/\partial y)$.`,
            answer: [res, J[0], J[1]], labels: ["$r$", "$J_x$", "$J_y$"], tol: 1e-3,
            explain: String.raw`$\mathbf p - \mathbf a = (${px - ax}, ${py - ay})$, length $${tri[2]}$, so $r = ${tri[2]} - ${d} = ${f3(res)}$. $J = (\mathbf p - \mathbf a)/\|\mathbf p - \mathbf a\| = (${f3(J[0])}, ${f3(J[1])})$: moving $\mathbf p$ away from the tower grows the distance at rate 1.` };
        } },
      { id: "lmstep", type: "num",
        gen: (r) => {
          const H = r.int(2, 8), g = r.nz(6), mu = r.pick([0.5, 1, 2, 3, 9]);
          const d = -g / (H * (1 + mu));
          return { q: String.raw`One unknown: $H = ${H}$, $g = ${g}$. Damping $\mu = ${mu}$ (this repo's form: $H(1+\mu)\,\delta = -g$). What is the damped step $\delta$? (to 4 decimals)`,
            answer: d, tol: 1e-4,
            explain: String.raw`$\delta = -g/(H(1+\mu)) = ${-g}/(${H}\cdot${1 + mu}) = ${f4(d)}$. The undamped GN step would be $${f4(-g / H)}$.` };
        } },
      { id: "lmsched", type: "num",
        gen: (r) => {
          const n = r.int(3, 5), seq = Array.from({ length: n }, () => r.pick(["accept", "reject"]));
          let mu = 1e-4;
          for (const s of seq) mu = s === "accept" ? Math.max(mu / 3, 1e-7) : mu * 10;
          return { q: String.raw`This repo's LM rule: $\mu$ starts at $10^{-4}$; accept → $\mu \leftarrow \max(\mu/3, 10^{-7})$; reject → $\mu \leftarrow 10\mu$. The steps go: <b>${seq.join(", ")}</b>. What is $\mu$ now? (e.g. 3.3e-5)`,
            answer: mu, rtol: 0.02,
            explain: String.raw`Apply in order: ${(() => { let m = 1e-4; return seq.map((s) => { m = s === "accept" ? Math.max(m / 3, 1e-7) : m * 10; return m.toExponential(3); }).join(" → "); })()}.` };
        } },
      { id: "lmreject", type: "multi",
        q: "An LM step would raise the cost. What does the solver do? (select all)",
        choices: ["Keeps the previous parameters (discards the candidate)", "Increases the damping $\\mu$", "Solves again using the same $H$ and $\\mathbf g$ as before", "Accepts the step anyway but halves $\\mu$", "Recomputes the Jacobian at the rejected candidate and continues from there"],
        answer: [0, 1, 2],
        explain: "Reject = stay put, damp harder (×10 here), and re-solve from the accepted point's normal equations. Only accepted points get new $H$, $\\mathbf g$." },
      { id: "lmmu", type: "mc",
        q: "What happens to the LM step as $\\mu$ becomes very large?",
        choices: ["It becomes very short and points roughly downhill (like gradient descent)", "It becomes the plain Gauss–Newton step", "It becomes longer, to escape local minima", "It points uphill"],
        answer: 0,
        explain: "Each $H_{jj}$ is multiplied by $1+\\mu$, so $\\delta_j \\approx -g_j/(H_{jj}(1+\\mu))$: tiny and along $-\\mathbf g$. $\\mu \\to 0$ gives Gauss–Newton." },
      { id: "huber", type: "num",
        gen: (r) => {
          const eps = r.pick([0.1, 0.2, 0.5, 1]), x = r.pick([-1, 1]) * r.pick([0.05, 0.1, 0.3, 0.8, 1.5, 2.5]);
          const v = Math.abs(x) <= eps ? (x * x) / (2 * eps) : Math.abs(x) - eps / 2;
          return { q: String.raw`Huber norm, paper eq. (4), with $\epsilon = ${eps}$. What is $\|${x}\|_\epsilon$? (to 4 decimals)`,
            answer: v, tol: 1e-4,
            explain: Math.abs(x) <= eps ? String.raw`$|${x}| \le \epsilon$: quadratic branch $${x}^2/(2\cdot${eps}) = ${f4(v)}$.` : String.raw`$|${x}| > \epsilon$: linear branch $${Math.abs(x)} - ${eps}/2 = ${f4(v)}$.` };
        } },
      { id: "trunc", type: "num",
        gen: (r) => {
          const tau = r.pick([0.1, 0.12, 0.2]);
          const rs = Array.from({ length: 5 }, () => r.pick([-1, 1]) * r.pick([0.02, 0.05, 0.08, 0.3, 0.5, 0.15, 0.25]));
          let s = 0, nr = 0;
          for (const v of rs) { if (Math.abs(v) <= tau) s += v * v; else nr++; }
          const E = (s + nr * tau * tau) / rs.length;
          return { q: String.raw`Truncated-quadratic tracking cost with $\tau = ${tau}$, five in-view pixels with residuals $${rs.join(",\\ ")}$. Compute $E = \frac{1}{N}\big(\sum_{|r|\le\tau} r^2 + N_{\text{rejected}}\tau^2\big)$. (to 5 decimals)`,
            answer: E, tol: 1e-5,
            explain: String.raw`Inlier squares add to $${L.fmt(s, 5)}$; ${nr} rejected × $${L.fmt(tau * tau, 4)}$ = $${L.fmt(nr * tau * tau, 5)}$. $E = ${L.fmt(s + nr * tau * tau, 5)}/5 = ${L.fmt(E, 5)}$.` };
        } },
      { id: "weight", type: "num",
        gen: (r) => {
          const eps = r.pick([0.1, 0.2, 0.5]), x = r.pick([0.4, 0.8, 1, 2, 0.25]) * r.sign();
          const w = Math.abs(x) <= eps ? 1 : eps / Math.abs(x);
          return { q: String.raw`Reweighted least squares with the Huber cost, $\epsilon = ${eps}$ (inliers have weight 1). What weight does a residual $r = ${x}$ get?`,
            answer: w, tol: 1e-4,
            explain: Math.abs(x) <= eps ? String.raw`$|r| \le \epsilon$ → weight 1.` : String.raw`$|r| > \epsilon$ → $w = \epsilon/|r| = ${eps}/${Math.abs(x)} = ${f4(w)}$. Its pull ($w\cdot r$) is capped at $\pm\epsilon$.` };
        } },
      { id: "basin", type: "multi",
        q: "Gauss–Newton converged to a wrong answer with low but non-zero cost. Which of these could fix it? (select all)",
        choices: ["Start from a better initial guess", "Solve coarse-to-fine: blurred / smaller images first", "Run more iterations from the same start", "Use a smaller robust threshold $\\tau$ from the first iteration"],
        answer: [0, 1],
        explain: "It sits in a local minimum: more iterations stay there. A tighter $\\tau$ at the start shrinks the basin further. A better start or a coarse level (wider valleys) puts you in the right basin." },
      { id: "pyr", type: "num",
        gen: (r) => {
          const s = r.pick([8, 12, 16, 20, 24, 32, 40]), l = r.int(1, 3);
          return { q: String.raw`An image feature moves $${s}$ pixels at pyramid level 0. How many pixels is that motion at level $${l}$ (each level halves width and height)?`,
            answer: s / 2 ** l, tol: 1e-6,
            explain: String.raw`Halve ${l} times: $${s}/2^{${l}} = ${f3(s / 2 ** l)}$. That's why coarse levels have wider basins.` };
        } },
      { id: "outl", type: "mc",
        q: "In this repo's dense tracker, what happens to a pixel whose residual exceeds the threshold $\\tau$?",
        choices: ["It adds nothing to $H$ and $\\mathbf g$, and counts $\\tau^2$ in the cost", "Its residual is clamped to $\\tau$ and it still adds its full Jacobian to $H$", "It is removed from the image permanently", "Its weight becomes $\\tau/|r|$"],
        answer: 0,
        explain: "Truncated quadratic: weight 0 beyond $\\tau$, constant $\\tau^2$ in the cost so costs of different candidates stay comparable. $\\tau/|r|$ would be Huber." },
    ]);
  },
});
