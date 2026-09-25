// Chapter 10: solving the decoupled energy (7): discrete gradient/divergence,
// the Legendre–Fenchel dual of the Huber norm, the saddle-point energy (9)–(10),
// the primal–dual updates from (11)–(12), the point-wise auxiliary search
// (13)–(14) with the band (15)–(17), the Newton step (18), the θ schedule,
// initialisation, and a full working 1D solver.
(() => {
  "use strict";

  const huber = (x, eps) => (Math.abs(x) <= eps ? (x * x) / (2 * eps) : Math.abs(x) - eps / 2);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // <solver>  (a line-by-line port of mapping.rs solve() + the WGSL passes, in 1D)
  const SOLVER = (() => {
    const N = 64, S = 32, XMIN = 0.2, XMAX = 2.0, DX = (XMAX - XMIN) / (S - 1);
    // DtamParams::default() (mod.rs)
    const P = { alpha: 100, beta: 1.6, eps: 1e-4, sigmaQ: 0.5, sigmaD: 0.25, theta0: 0.2, thetaEnd: 1e-4,
      betaFast: 2.5e-4, betaSlow: 2.5e-5, thetaSwitch: 1e-3, maxTrough: 6, minCount: 2 };
    const BF = [-3, -2, -1, 1, 2, 3]; // disparity (pixels) per unit inverse depth for each extra frame

    function makeScene(rand) {
      const xi = new Float64Array(N), I = new Float64Array(N);
      const ph = [rand() * 6.28, rand() * 6.28, rand() * 6.28];
      const tex = (u) => 0.07 * (Math.sin(0.9 * u + ph[0]) + 0.7 * Math.sin(2.3 * u + ph[1]) + 0.5 * Math.sin(3.7 * u + ph[2]));
      for (let u = 0; u < N; u++) {
        if (u < 20) { xi[u] = 0.45 + 0.01 * u; I[u] = 0.32 + tex(u); }
        else if (u < 44) { xi[u] = 1.5; I[u] = 0.72 + (u < 26 || u >= 38 ? tex(u) : 0); }
        else { xi[u] = 0.8 - 0.008 * (u - 44); I[u] = 0.3 + tex(u); }
      }
      const lin = (arr, x) => {
        if (x < 0 || x > N - 1) return NaN;
        const i = Math.min(N - 2, Math.floor(x)), f = x - i;
        return arr[i] * (1 - f) + arr[i + 1] * f;
      };
      const gauss = () => (rand() + rand() + rand() + rand() - 2) * 1.7;
      // Render each other frame: forward-splat sub-pixel samples of the reference,
      // nearest surface (largest ξ) wins, holes take the nearest filled neighbour.
      const frames = BF.map((bf) => {
        const img = new Float64Array(N).fill(NaN), zb = new Float64Array(N).fill(-1);
        for (let s = 0; s < N * 8; s++) {
          const us = s / 8;
          if (us > N - 1) break;
          const x = us + bf * xi[Math.round(us)];
          const b = Math.round(x);
          if (b < 0 || b >= N) continue;
          const z = xi[Math.round(us)];
          if (z > zb[b] + 1e-9 || (Math.abs(z - zb[b]) < 1e-9 && Math.abs(x - b) < 0.0626)) { zb[b] = z; img[b] = lin(I, us + (b - x)); }
        }
        for (let b = 0; b < N; b++) if (isNaN(img[b])) {
          for (let o = 1; o < N; o++) {
            if (b - o >= 0 && !isNaN(img[b - o]) && zb[b - o] >= 0) { img[b] = img[b - o]; break; }
            if (b + o < N && !isNaN(img[b + o]) && zb[b + o] >= 0) { img[b] = img[b + o]; break; }
          }
        }
        for (let b = 0; b < N; b++) img[b] = clamp((isNaN(img[b]) ? 0.5 : img[b]) + 0.008 * gauss(), 0, 1);
        return { bf, img };
      });
      // Cost volume (eqs 2–3): average L1 photometric error over the frames that see the voxel.
      const sum = new Float64Array(N * S), cnt = new Uint8Array(N * S);
      for (const f of frames) {
        for (let k = 0; k < S; k++) {
          const xk = XMIN + k * DX;
          for (let u = 0; u < N; u++) {
            const v = lin(f.img, u + f.bf * xk);
            if (isNaN(v)) continue;
            sum[k * N + u] += Math.abs(I[u] - v);
            cnt[k * N + u]++;
          }
        }
      }
      // cost_access.wgsl: C(u,k) and whether it was observed by enough frames.
      const C = new Float64Array(N * S), obs = new Uint8Array(N * S);
      for (let i = 0; i < N * S; i++) {
        if (cnt[i] >= Math.max(P.minCount, 1)) { C[i] = sum[i] / cnt[i]; obs[i] = 1; }
      }
      // weights.wgsl: g = exp(-α |∇I|^β), central differences, clamped at the border.
      const g = new Float64Array(N);
      for (let u = 0; u < N; u++) {
        const gx = 0.5 * (I[Math.min(N - 1, u + 1)] - I[Math.max(0, u - 1)]);
        g[u] = Math.exp(-P.alpha * Math.pow(Math.abs(gx), P.beta));
      }
      // cost_minmax.wgsl: Cmin, Cmax, arg min, trough width.
      const cmin = new Float64Array(N), cmax = new Float64Array(N), argmin = new Float64Array(N), width = new Float64Array(N);
      for (let u = 0; u < N; u++) {
        let lo = 1e30, hi = 0, kmin = (S - 1) >> 1, valid = 0;
        for (let k = 0; k < S; k++) {
          if (!obs[k * N + u]) continue;
          valid++;
          const c = C[k * N + u];
          if (c < lo) { lo = c; kmin = k; }
          hi = Math.max(hi, c);
        }
        if (!valid) lo = 0;
        const tau = Math.max(0.01, 0.1 * (hi - lo));
        let first = S, last = 0;
        for (let k = 0; k < S; k++) if (obs[k * N + u] && C[k * N + u] <= lo + tau) { first = Math.min(first, k); last = Math.max(last, k); }
        cmin[u] = lo; cmax[u] = hi; argmin[u] = XMIN + kmin * DX;
        width[u] = valid && last >= first ? last - first + 1 : S;
      }
      return { xi, I, g, C, obs, cmin, cmax, argmin, width };
    }

    // mapping.rs fill_unconstrained(), with h = 1.
    function pushPull(val0, wt0, w, h) {
      const val = Array.from(val0);
      const wt = Array.from(wt0);
      if (wt.every((v) => v === 0)) return val;
      const levels = [[w, h, val.map((v, i) => v * wt[i]), wt.slice()]];
      while (levels[levels.length - 1][0] > 1 || levels[levels.length - 1][1] > 1) {
        const [pw, ph, pv, pwt] = levels[levels.length - 1];
        const nw = Math.ceil(pw / 2), nh = Math.ceil(ph / 2);
        const nv = new Array(nw * nh).fill(0), nwt = new Array(nw * nh).fill(0);
        for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
          const i = y * pw + x, j = (y >> 1) * nw + (x >> 1);
          nv[j] += pv[i]; nwt[j] += pwt[i];
        }
        levels.push([nw, nh, nv, nwt]);
      }
      for (let l = levels.length - 2; l >= 0; l--) {
        const [cw, , cv, cwt] = levels[l + 1];
        const [fw, fh, fv, fwt] = levels[l];
        for (let y = 0; y < fh; y++) for (let x = 0; x < fw; x++) {
          const i = y * fw + x, j = (y >> 1) * cw + (x >> 1);
          const coarse = cwt[j] > 0 ? cv[j] / cwt[j] : 0;
          const t = Math.min(fwt[i], 1);
          const mean = fwt[i] > 0 ? fv[i] / fwt[i] : coarse;
          fv[i] = t * mean + (1 - t) * coarse;
          fwt[i] = 1;
        }
      }
      return val.map((v, i) => (wt[i] === 0 ? levels[0][2][i] : v));
    }

    function init(sc, usePushPull) {
      const wt = Array.from(sc.width, (wd) => (wd <= P.maxTrough ? 1 : 0));
      const d0 = usePushPull ? pushPull(sc.argmin, wt, N, 1) : Array.from(sc.argmin);
      return { d: Float64Array.from(d0), a: Float64Array.from(d0), d0: Float64Array.from(d0), localised: wt,
        q: new Float64Array(N), theta: P.theta0, iter: 0, done: false, tested: 0 };
    }

    // aux.wgsl energy(): unobserved voxels cost Cmax.
    function eaux(sc, u, k, du, theta, lambda) {
      const xk = XMIN + k * DX;
      const data = sc.obs[k * N + u] ? sc.C[k * N + u] : sc.cmax[u];
      return ((du - xk) * (du - xk)) / (2 * theta) + lambda * data;
    }

    // One pass of the loop body in solve(): dual, primal, aux, then the θ update.
    function step(sc, st, { lambda = 1, newton = true, band = true } = {}) {
      if (st.done) return;
      const { d, a, q, theta } = st, g = sc.g;
      // dual.wgsl
      for (let i = 0; i < N; i++) {
        const grad = i + 1 < N ? d[i + 1] - d[i] : 0;
        const qn = (q[i] + P.sigmaQ * g[i] * grad) / (1 + P.sigmaQ * P.eps);
        q[i] = qn / Math.max(1, Math.abs(qn));
      }
      // primal.wgsl
      for (let i = 0; i < N; i++) {
        let div = 0;
        if (i + 1 < N) div += g[i] * q[i];
        if (i > 0) div -= g[i - 1] * q[i - 1];
        d[i] = (d[i] + P.sigmaD * (div + a[i] / theta)) / (1 + P.sigmaD / theta);
      }
      // aux.wgsl
      let tested = 0;
      for (let i = 0; i < N; i++) {
        const du = d[i];
        const r = band ? Math.sqrt(2 * theta * lambda * Math.max(sc.cmax[i] - sc.cmin[i], 0)) : 1e9;
        const k0 = clamp(Math.floor((du - r - XMIN) / DX), 0, S - 1);
        const k1 = clamp(Math.ceil((du + r - XMIN) / DX), 0, S - 1);
        let best = k0, eb = eaux(sc, i, k0, du, theta, lambda);
        for (let k = k0 + 1; k <= k1; k++) {
          const e = eaux(sc, i, k, du, theta, lambda);
          if (e < eb) { eb = e; best = k; }
        }
        tested += k1 - k0 + 1;
        let x = XMIN + best * DX;
        if (newton && best > 0 && best + 1 < S) {
          const em = eaux(sc, i, best - 1, du, theta, lambda), ep = eaux(sc, i, best + 1, du, theta, lambda);
          const grad = (ep - em) / (2 * DX), hess = (ep - 2 * eb + em) / (DX * DX);
          if (hess > 0) x -= clamp(grad / hess, -DX, DX);
        }
        a[i] = x;
      }
      st.tested = tested / N;
      // mapping.rs: θ_{n+1} = θ_n (1 − β n)
      const beta = theta >= P.thetaSwitch ? P.betaFast : P.betaSlow;
      const factor = 1 - beta * st.iter;
      st.iter++;
      if (factor <= 0 || st.iter > 5000) { st.done = true; return; }
      st.theta = theta * factor;
      if (!(st.theta > P.thetaEnd)) st.done = true;
    }

    // Energy (7) in vector form (C(a) linearly interpolated between layers).
    function energy(sc, st, lambda) {
      let e = 0;
      for (let i = 0; i < N; i++) {
        const grad = i + 1 < N ? st.d[i + 1] - st.d[i] : 0;
        e += sc.g[i] * huber(grad, P.eps) + ((st.d[i] - st.a[i]) ** 2) / (2 * st.theta);
        const f = clamp((st.a[i] - XMIN) / DX, 0, S - 1), k = Math.min(S - 2, Math.floor(f)), t = f - k;
        const c0 = sc.obs[k * N + i] ? sc.C[k * N + i] : sc.cmax[i], c1 = sc.obs[(k + 1) * N + i] ? sc.C[(k + 1) * N + i] : sc.cmax[i];
        e += lambda * (c0 * (1 - t) + c1 * t);
      }
      return e;
    }

    return { N, S, XMIN, XMAX, DX, P, makeScene, pushPull, init, step, energy };
  })();
  // </solver>

  DTAM.chapter({
    id: "primaldual",
    order: 10,
    title: "Solving it: the primal–dual algorithm",
    subtitle: "Gradients as matrices, the dual of the Huber norm, a saddle-point game, and the point-wise search",
    minutes: 75,
    render(root, L) {
      const { N, S, XMIN, XMAX, DX, P } = SOLVER;
      const fmt = L.fmt;

      root.insertAdjacentHTML("beforeend", String.raw`
<style>
#primaldual table.tbl { border-collapse: collapse; font-size: 14.5px; margin: 8px 0; width: 100%; }
#primaldual table.tbl td, #primaldual table.tbl th { border-bottom: 1px solid var(--line); padding: 5px 6px; text-align: left; vertical-align: top; }
#primaldual table.tbl th { color: var(--muted); font-weight: 600; }
#primaldual .tblwrap { overflow-x: auto; }
</style>
<p>Chapter 9 ended with the decoupled energy (7). This chapter turns it into the loop that actually runs on the GPU. From here on we use the paper's names: the <b>primal</b> variable $\mathbf d$ (the inverse depth map we want; $\xi$ in earlier chapters) and the <b>auxiliary</b> variable $\mathbf a$.</p>
<div class="eq-card"><div class="eq-label">Paper eq. (7), written as a sum over pixels</div>
$$E(\mathbf d,\mathbf a)=\sum_{\mathbf u}\Big[\,g(\mathbf u)\,\big\|\nabla d(\mathbf u)\big\|_\epsilon+\frac{1}{2\theta}\big(d_{\mathbf u}-a_{\mathbf u}\big)^2+\lambda\, C(\mathbf u,a_{\mathbf u})\Big]$$
<div class="parts">
<span>$g\|\nabla d\|_\epsilon$</span><span>edge-weighted Huber smoothness: convex in $\mathbf d$, couples neighbours</span>
<span>$\frac1{2\theta}(d-a)^2$</span><span>coupling: keeps $\mathbf d$ and $\mathbf a$ close; tighter as $\theta\to0$</span>
<span>$\lambda C(\mathbf u,a_{\mathbf u})$</span><span>data term: non-convex, but each pixel on its own</span>
</div></div>
<p>The trick is to alternate between two problems that are each easy:</p>
<div class="tblwrap"><table class="tbl">
<tr><th>step</th><th>hold fixed</th><th>solve for</th><th>how</th></tr>
<tr><td>1</td><td>$\mathbf a$</td><td>$\mathbf d$ (smooth + close to $\mathbf a$)</td><td>one primal–dual step (convex)</td></tr>
<tr><td>2</td><td>$\mathbf d$</td><td>$\mathbf a$ (close to $\mathbf d$ + low cost)</td><td>search every pixel's cost row</td></tr>
<tr><td>3</td><td>—</td><td>$\theta$</td><td>shrink it a little, repeat</td></tr>
</table></div>
<p>Step 1 needs three new tools: the gradient as a matrix, a "dual" way of writing the Huber norm, and a saddle point. We build them one at a time.</p>

<h3>1 · The gradient as a matrix $A$</h3>
<p>Stack all pixels of $\mathbf d$ into one long vector (row by row). The gradient is then a matrix $A$ times that vector. This implementation uses <b>forward differences</b>, with the difference set to <b>zero at the far border</b> (last column / last row), as in <code>dual.wgsl</code>.</p>
<div class="eq-card"><div class="eq-label">Discrete gradient $A$ (1D, $N$ pixels)</div>
$$(A\mathbf d)_i=\begin{cases}d_{i+1}-d_i & i<N-1\\ 0 & i=N-1\end{cases}$$
<div class="parts">
<span>$A$</span><span>an $N\times N$ matrix: row $i$ has $-1$ at column $i$ and $+1$ at column $i+1$; the last row is all zeros</span>
<span>$(A\mathbf d)_i$</span><span>how much $d$ rises going one pixel right</span>
</div></div>
<p>Example: $\mathbf d=(1,3,4,4,2)$ gives $A\mathbf d=(2,1,0,-2,0)$. The last entry is $0$ by the border rule.</p>
<p>The <b>transpose</b> $A^\top$ is just as important, because the $\mathbf d$-update needs it. Reading $A$ column by column gives $(A^\top\mathbf q)_i = q_{i-1}-q_i$, i.e. minus a <b>backward</b> difference. We call $-A^\top$ the <b>divergence</b>:</p>
<div class="eq-card"><div class="eq-label">Divergence $\operatorname{div}=-A^\top$ (1D)</div>
$$(\operatorname{div}\mathbf q)_i=\underbrace{q_i}_{\text{only if }i<N-1}\;-\;\underbrace{q_{i-1}}_{\text{only if }i>0}$$
<div class="parts">
<span>$q_i$ term</span><span>dropped at the last pixel (that row of $A$ is zero, so $q_{N-1}$ never matters)</span>
<span>$q_{i-1}$ term</span><span>dropped at the first pixel (there is no pixel $-1$)</span>
<span>adjoint rule</span><span>$\langle A\mathbf d,\mathbf q\rangle=\langle\mathbf d,A^\top\mathbf q\rangle=-\langle\mathbf d,\operatorname{div}\mathbf q\rangle$ for <i>every</i> $\mathbf d,\mathbf q$</span>
</div></div>
<p>Check with $\mathbf d=(1,3,4,4,2)$, $\mathbf q=(0.5,-1,0.25,1,0.7)$: $\operatorname{div}\mathbf q=(0.5,-1.5,1.25,0.75,-1)$. Then $\langle A\mathbf d,\mathbf q\rangle=2(0.5)+1(-1)+0+(-2)(1)+0=-2$ and $-\langle\mathbf d,\operatorname{div}\mathbf q\rangle=-(0.5-4.5+5+3-2)=-2$. ✓</p>
`);

      // ------------------------------------------------------------ widget 1: 1D gradient / divergence
      {
        const d = [1, 3, 4, 4, 2, 2.5];
        const q = [0.5, -1, 0.25, 1, -0.4, 0.7];
        const gw = [1, 1, 0.15, 1, 1, 1];
        let useG = false;
        const n = d.length;
        const fig = L.figure(root, "<b>Gradient and divergence.</b> Drag the blue dots (d) and the orange dots (q). Arrows show $G A\\mathbf d$; bars show $\\operatorname{div}(g\\mathbf q)$. The two inner products always agree. Note the last q changes nothing.");
        const c = L.canvas(fig.el, { aspect: 0.95 });
        fig.add(c.el);
        const ctl = L.controls(fig.el); fig.add(ctl);
        L.toggle(ctl, "edge weight $g$ (low at pixel 2)", false, (v) => { useG = v; c.redraw(); upd(); });
        const out = L.readout(fig.el); fig.add(out.el);
        const geo = () => {
          const pl = 34, pr = 10, cw = (c.w - pl - pr) / n;
          const top = { y0: 0, y1: 5, t: 22, b: c.h * 0.47 };
          const bot = { y0: -2.2, y1: 2.2, t: c.h * 0.56, b: c.h - 10 };
          const Y = (p, v) => p.b - ((v - p.y0) / (p.y1 - p.y0)) * (p.b - p.t);
          const iY = (p, y) => p.y0 + ((p.b - y) / (p.b - p.t)) * (p.y1 - p.y0);
          return { pl, cw, top, bot, X: (i) => pl + (i + 0.5) * cw, Y, iY };
        };
        const gg = () => (useG ? gw : gw.map(() => 1));
        const grad = () => d.map((v, i) => (i < n - 1 ? d[i + 1] - v : 0));
        const div = () => { const g = gg(); return q.map((_, i) => (i < n - 1 ? g[i] * q[i] : 0) - (i > 0 ? g[i - 1] * q[i - 1] : 0)); };
        const upd = () => {
          const g = gg(), Ad = grad(), dv = div();
          const lhs = Ad.reduce((s, v, i) => s + g[i] * v * q[i], 0);
          const rhs = -d.reduce((s, v, i) => s + v * dv[i], 0);
          out.html = `GAd = (${Ad.map((v, i) => fmt(g[i] * v, 2)).join(", ")})<br>div(gq) = (${dv.map((v) => fmt(v, 2)).join(", ")})<br>` +
            `⟨GAd, q⟩ = <b>${fmt(lhs, 3)}</b> &nbsp; −⟨d, div(gq)⟩ = <b>${fmt(rhs, 3)}</b>`;
        };
        c.draw = (ctx) => {
          const t = L.theme(), G = geo(), g = gg(), Ad = grad(), dv = div();
          for (const p of [G.top, G.bot]) {
            const ticks = p === G.top ? [0, 1, 2, 3, 4, 5] : [-2, -1, 0, 1, 2];
            for (const v of ticks) {
              L.draw.line(ctx, G.pl, G.Y(p, v), c.w - 10, G.Y(p, v), v === 0 ? t.faint : t.line, 1);
              L.draw.text(ctx, String(v), G.pl - 6, G.Y(p, v) + 4, t.faint, { size: 11, align: "right" });
            }
          }
          L.draw.text(ctx, "d  (arrows: g·∇d)", G.pl + 4, 14, t.muted, { size: 12 });
          L.draw.text(ctx, "q (dots)   div(gq) (bars)", G.pl + 4, G.bot.t - 8, t.muted, { size: 12 });
          // unit q band
          ctx.save(); ctx.fillStyle = t.accent2; ctx.globalAlpha = 0.07;
          ctx.fillRect(G.pl, G.Y(G.bot, 1), c.w - 10 - G.pl, G.Y(G.bot, -1) - G.Y(G.bot, 1)); ctx.restore();
          for (let i = 0; i < n; i++) {
            const x = G.X(i);
            L.draw.text(ctx, "i=" + i, x, G.top.b + 16, t.faint, { size: 11, align: "center" });
            if (useG && g[i] < 1) L.draw.text(ctx, "g=" + g[i], x, G.top.b + 30, t.bad, { size: 11, align: "center" });
            // stem + point
            L.draw.line(ctx, x, G.Y(G.top, 0), x, G.Y(G.top, d[i]), t.accent, 3);
            // gradient arrow drawn half-way to the next pixel
            if (i < n - 1) {
              const xm = x + G.cw * 0.5, v = g[i] * Ad[i];
              if (Math.abs(v) > 0.02) L.draw.arrow(ctx, xm, G.Y(G.top, d[i]), xm, G.Y(G.top, d[i] + v), t.accent3, 2, 7);
              L.draw.text(ctx, fmt(v, 2), xm, G.Y(G.top, Math.max(d[i], d[i] + v)) - 5, t.accent3, { size: 11, align: "center" });
            } else {
              L.draw.text(ctx, "∇=0", x + G.cw * 0.28, G.Y(G.top, d[i]) - 6, t.faint, { size: 11, align: "center" });
            }
            // div bars
            const bw = G.cw * 0.5;
            ctx.save(); ctx.fillStyle = t.accent4; ctx.globalAlpha = 0.55;
            const y0 = G.Y(G.bot, 0), y1 = G.Y(G.bot, clamp(dv[i], -2.2, 2.2));
            ctx.fillRect(x - bw / 2, Math.min(y0, y1), bw, Math.abs(y1 - y0)); ctx.restore();
          }
          for (let i = 0; i < n; i++) {
            L.draw.handle(ctx, G.X(i), G.Y(G.top, d[i]), t.accent);
            ctx.save(); ctx.globalAlpha = i === n - 1 ? 0.4 : 1;
            L.draw.handle(ctx, G.X(i), G.Y(G.bot, q[i]), t.accent2); ctx.restore();
          }
        };
        L.drag(c, () => { const G = geo(); return [...d.map((v, i) => ({ x: G.X(i), y: G.Y(G.top, v) })), ...q.map((v, i) => ({ x: G.X(i), y: G.Y(G.bot, v) }))]; },
          (k, p) => {
            const G = geo();
            if (k < n) d[k] = Math.round(clamp(G.iY(G.top, p.y), 0, 5) * 4) / 4;
            else q[k - n] = Math.round(clamp(G.iY(G.bot, p.y), -1, 1) * 20) / 20;
            upd();
          });
        upd();
      }

      root.insertAdjacentHTML("beforeend", String.raw`
<p><b>In 2D</b> every pixel gets a 2-vector gradient, so $A\mathbf d$ has $2N$ entries (the paper's "$2MN\times1$" vector) and $\mathbf q$ is a 2-vector per pixel:</p>
<div class="eq-card"><div class="eq-label">2D gradient and divergence (dual.wgsl, primal.wgsl)</div>
$$(A\mathbf d)(u,v)=\begin{pmatrix}d(u{+}1,v)-d(u,v)\\ d(u,v{+}1)-d(u,v)\end{pmatrix},\qquad (\operatorname{div}\mathbf q)(u,v)=q^x(u,v)-q^x(u{-}1,v)+q^y(u,v)-q^y(u,v{-}1)$$
<div class="parts">
<span>$x$ part of $A\mathbf d$</span><span>$0$ on the last column</span>
<span>$y$ part of $A\mathbf d$</span><span>$0$ on the last row</span>
<span>$q^x(u,v)$, $q^y(u,v)$</span><span>dropped on the last column / last row; $q^x(u{-}1,v)$, $q^y(u,v{-}1)$ dropped on the first column / first row</span>
</div></div>
<p>Gradient reads the pixel's <b>right and lower</b> neighbours; divergence reads the <b>left and upper</b> ones. That mirror image is exactly what "transpose" means here.</p>
`);

      // ------------------------------------------------------------ widget 2: 2D stencil
      {
        const W = 5, H = 4;
        const grid = [1, 1, 2, 3, 3, 1, 2, 2, 3, 4, 0, 1, 3, 3, 4, 0, 0, 2, 2, 3];
        let sel = 6;
        const fig = L.figure(root, "<b>2D stencils.</b> Tap a pixel to select it; use +/− to change its value. Blue outlines: what the gradient reads. Orange outlines: what the divergence reads. Arrows: $\\nabla d$ at every pixel.");
        const c = L.canvas(fig.el, { aspect: 0.66 });
        fig.add(c.el);
        const ctl = L.controls(fig.el); fig.add(ctl);
        const out = L.readout(fig.el); fig.add(out.el);
        const at = (u, v) => grid[v * W + u];
        const gradAt = (u, v) => [u + 1 < W ? at(u + 1, v) - at(u, v) : 0, v + 1 < H ? at(u, v + 1) - at(u, v) : 0];
        const upd = () => {
          const u = sel % W, v = Math.floor(sel / W);
          const [gx, gy] = gradAt(u, v);
          const qx = (uu, vv) => gradAt(uu, vv)[0], qy = (uu, vv) => gradAt(uu, vv)[1];
          const terms = [];
          let dv = 0;
          if (u + 1 < W) { terms.push(`+qx(${u},${v})=${fmt(qx(u, v))}`); dv += qx(u, v); } else terms.push("(last column: no qx(u,v))");
          if (u > 0) { terms.push(`−qx(${u - 1},${v})=${fmt(-qx(u - 1, v))}`); dv -= qx(u - 1, v); } else terms.push("(first column: no left term)");
          if (v + 1 < H) { terms.push(`+qy(${u},${v})=${fmt(qy(u, v))}`); dv += qy(u, v); } else terms.push("(last row: no qy(u,v))");
          if (v > 0) { terms.push(`−qy(${u},${v - 1})=${fmt(-qy(u, v - 1))}`); dv -= qy(u, v - 1); } else terms.push("(first row: no upper term)");
          out.html = `pixel (u,v)=(${u},${v}), d=${at(u, v)}<br>∇d = (<b>${gx}</b>, <b>${gy}</b>)` +
            `${u + 1 >= W ? " (x part 0: last column)" : ""}${v + 1 >= H ? " (y part 0: last row)" : ""}` +
            `<br>with q = ∇d: div q = ${terms.join(" ")} = <b>${fmt(dv)}</b>`;
        };
        const geo = () => {
          const cs = Math.min((c.w - 8) / W, (c.h - 8) / H);
          return { cs, ox: (c.w - cs * W) / 2, oy: (c.h - cs * H) / 2 };
        };
        c.draw = (ctx) => {
          const t = L.theme(), { cs, ox, oy } = geo();
          const su = sel % W, sv = Math.floor(sel / W);
          for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
            const x = ox + u * cs, y = oy + v * cs;
            ctx.fillStyle = L.gray(0.15 + at(u, v) * 0.17);
            ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
            L.draw.text(ctx, String(at(u, v)), x + 7, y + 17, at(u, v) >= 3 ? "#111" : "#eee", { size: 13, bold: true });
            const [gx, gy] = gradAt(u, v);
            const s = cs * 0.14;
            if (gx || gy) L.draw.arrow(ctx, x + cs / 2, y + cs / 2, x + cs / 2 + gx * s, y + cs / 2 + gy * s, t.accent3, 2, 7);
            else L.draw.dot(ctx, x + cs / 2, y + cs / 2, 2.5, t.accent3);
          }
          const box = (u, v, col, w) => {
            if (u < 0 || v < 0 || u >= W || v >= H) return;
            ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = w;
            ctx.strokeRect(ox + u * cs + w / 2 + 1, oy + v * cs + w / 2 + 1, cs - w - 2, cs - w - 2); ctx.restore();
          };
          box(su + 1, sv, t.accent, 3); box(su, sv + 1, t.accent, 3);
          box(su - 1, sv, t.accent2, 3); box(su, sv - 1, t.accent2, 3);
          box(su, sv, t.fg, 4);
        };
        c.el.addEventListener("pointerdown", (e) => {
          const p = c.pos(e), { cs, ox, oy } = geo();
          const u = Math.floor((p.x - ox) / cs), v = Math.floor((p.y - oy) / cs);
          if (u >= 0 && v >= 0 && u < W && v < H) { sel = v * W + u; upd(); c.redraw(); }
        });
        L.button(ctl, "+1", () => { grid[sel] = Math.min(5, grid[sel] + 1); upd(); c.redraw(); });
        L.button(ctl, "−1", () => { grid[sel] = Math.max(0, grid[sel] - 1); upd(); c.redraw(); });
        upd();
      }

      root.insertAdjacentHTML("beforeend", String.raw`
<p><b>Adding the edge weight.</b> Let $G=\operatorname{diag}(g)$ scale each pixel's gradient by $g(\mathbf u)$. The paper writes the weighted gradient as "$AG\mathbf d$"; what it means, and what the code computes, is <i>gradient first, then multiply by $g$</i>: $g(\mathbf u)\,(A\mathbf d)(\mathbf u)$. We write that $GA\mathbf d$. Its transpose is $A^\top G$, so the $\mathbf d$-update will need $-A^\top G\mathbf q=\operatorname{div}(g\,\mathbf q)$: multiply $\mathbf q$ by $g$ at each pixel, then take the divergence.</p>

<h3>2 · The Huber norm as a maximum (eq. 8)</h3>
<p>The Huber norm has a kink-free but awkward two-case formula. The <b>Legendre–Fenchel</b> trick rewrites it as the maximum of a family of simple expressions, each <b>linear in $x$</b>:</p>
<div class="eq-card"><div class="eq-label">Paper eq. (8) · dual form of the Huber norm (one pixel)</div>
$$\|x\|_\epsilon=\max_{\|q\|\le 1}\Big(\langle x,q\rangle-\frac{\epsilon}{2}\|q\|^2\Big)=\begin{cases}\dfrac{\|x\|^2}{2\epsilon}&\|x\|\le\epsilon\\[2mm] \|x\|-\dfrac\epsilon2&\|x\|>\epsilon\end{cases}$$
<div class="parts">
<span>$x$</span><span>the (weighted) gradient at one pixel: $g(\mathbf u)(A\mathbf d)(\mathbf u)$</span>
<span>$q$</span><span>the <b>dual variable</b> at that pixel, confined to the unit disc $\|q\|\le1$</span>
<span>$\langle x,q\rangle$</span><span>linear in $x$: easy to differentiate with respect to $\mathbf d$</span>
<span>$-\frac\epsilon2\|q\|^2$</span><span>rounds off the corner of $|x|$ (gives the quadratic part near 0)</span>
<span>paper's $\delta_q(q)$</span><span>"$0$ inside the disc, $\infty$ outside": the same thing as writing $\max_{\|q\|\le1}$</span>
</div></div>
<p><b>Why it works (1D).</b> $f(q)=qx-\frac\epsilon2q^2$ is an upside-down parabola with peak at $q=x/\epsilon$.</p>
<ul>
<li>If $|x|\le\epsilon$, the peak is inside $[-1,1]$: $q^\star=x/\epsilon$, value $\frac{x^2}{\epsilon}-\frac{x^2}{2\epsilon}=\frac{x^2}{2\epsilon}$.</li>
<li>If $|x|>\epsilon$, the peak is outside, so the best allowed $q$ is the nearest end: $q^\star=\operatorname{sign}(x)$, value $|x|-\frac\epsilon2$.</li>
</ul>
<p>Both cases are exactly Huber (4). Example: $\epsilon=0.5$, $x=0.2$: $q^\star=0.4$, value $0.2(0.4)-0.25(0.16)=0.04=\frac{0.2^2}{2(0.5)}$. With $x=2$: $q^\star=1$, value $2-0.25=1.75$. In 2D the same holds with $q^\star=x/\epsilon$ if $\|x\|\le\epsilon$, else $q^\star=x/\|x\|$.</p>
`);

      // ------------------------------------------------------------ widget 3: Legendre–Fenchel
      {
        const fig = L.figure(root, "<b>The max hides the Huber norm.</b> Move $x$ and $\\epsilon$. Top: $f(q)=qx-\\frac\\epsilon2q^2$; only $|q|\\le1$ is allowed (shaded outside). Its highest allowed point equals the Huber value (bottom).");
        const c = L.canvas(fig.el, { aspect: 0.95, scroll: true });
        fig.add(c.el);
        const ctl = L.controls(fig.el); fig.add(ctl);
        const sx = L.slider(ctl, { label: "$x$", min: -1.5, max: 1.5, step: 0.01, value: 0.3, oninput: () => { c.redraw(); upd(); } });
        const se = L.slider(ctl, { label: "$\\epsilon$", min: 0.05, max: 1, step: 0.01, value: 0.5, oninput: () => { c.redraw(); upd(); } });
        const out = L.readout(fig.el); fig.add(out.el);
        const qstar = (x, e) => clamp(x / e, -1, 1);
        const upd = () => {
          const x = sx.value, e = se.value, qs = qstar(x, e), v = qs * x - (e / 2) * qs * qs;
          out.html = `q* = ${Math.abs(x) <= e ? "x/ε" : "sign(x)"} = <b>${fmt(qs)}</b> &nbsp; max f = <b>${fmt(v, 4)}</b> &nbsp; ‖x‖ε = <b>${fmt(huber(x, e), 4)}</b>`;
        };
        c.draw = (ctx) => {
          const t = L.theme(), x = sx.value, e = se.value;
          const f = (q) => q * x - (e / 2) * q * q;
          let lo = Infinity, hi = -Infinity;
          for (let q = -1.4; q <= 1.4; q += 0.01) { lo = Math.min(lo, f(q)); hi = Math.max(hi, f(q)); }
          hi = Math.max(hi, huber(x, e)) + 0.1; lo -= 0.1;
          const h1 = c.h * 0.55;
          const top = L.plot({ w: c.w, h: h1 }, { x0: -1.4, x1: 1.4, y0: lo, y1: hi, pad: [14, 12, 22, 40] });
          top.axes(ctx, { xticks: 4, yticks: 3, xlabel: "q" });
          ctx.save(); ctx.fillStyle = t.faint; ctx.globalAlpha = 0.15;
          ctx.fillRect(top.X(-1.4), 14, top.X(-1) - top.X(-1.4), h1 - 36);
          ctx.fillRect(top.X(1), 14, top.X(1.4) - top.X(1), h1 - 36); ctx.restore();
          const pts = (a, b) => { const r = []; for (let q = a; q <= b + 1e-9; q += 0.02) r.push([top.X(q), top.Y(f(q))]); return r; };
          L.draw.path(ctx, pts(-1.4, -1), t.faint, 2, [4, 4]);
          L.draw.path(ctx, pts(1, 1.4), t.faint, 2, [4, 4]);
          L.draw.path(ctx, pts(-1, 1), t.accent2, 2.5);
          const qs = qstar(x, e), val = f(qs);
          L.draw.line(ctx, top.X(-1.4), top.Y(huber(x, e)), top.X(1.4), top.Y(huber(x, e)), t.accent, 1, [3, 3]);
          L.draw.dot(ctx, top.X(qs), top.Y(val), 6, t.accent2, t.panel);
          L.draw.text(ctx, "f(q)", 44, 26, t.accent2, { size: 12, bold: true });
          // bottom: Huber curve
          const bot = L.plot(c, { x0: -1.5, x1: 1.5, y0: 0, y1: 1.5, pad: [h1 + 12, 12, 22, 40] });
          bot.axes(ctx, { xticks: 6, yticks: 3, xlabel: "x" });
          const hp = [], ap = [];
          for (let xx = -1.5; xx <= 1.5001; xx += 0.01) { hp.push([bot.X(xx), bot.Y(huber(xx, e))]); ap.push([bot.X(xx), bot.Y(Math.abs(xx))]); }
          L.draw.path(ctx, ap, t.faint, 1.5, [4, 4]);
          L.draw.path(ctx, hp, t.accent, 2.5);
          L.draw.line(ctx, bot.X(-e), bot.Y(0), bot.X(-e), bot.Y(1.5), t.line, 1);
          L.draw.line(ctx, bot.X(e), bot.Y(0), bot.X(e), bot.Y(1.5), t.line, 1);
          L.draw.dot(ctx, bot.X(x), bot.Y(huber(x, e)), 6, t.accent, t.panel);
          L.draw.text(ctx, "‖x‖ε   (dashed: |x|)", 46, h1 + 26, t.accent, { size: 12, bold: true });
        };
        upd();
      }

      root.insertAdjacentHTML("beforeend", String.raw`
<div class="key">The regulariser $\sum g\|\nabla d\|_\epsilon$ becomes $\max_{\mathbf q}\big(\langle GA\mathbf d,\mathbf q\rangle-\frac\epsilon2\|\mathbf q\|^2\big)$, with one $q$ per pixel. Inside the max, $\mathbf d$ appears only <b>linearly</b>, so its derivative with respect to $\mathbf d$ is simply $A^\top G\mathbf q=-\operatorname{div}(g\mathbf q)$.</div>

<h3>3 · A saddle point: a two-player game (eqs. 9–10)</h3>
<p>Put the dual form into (7). The problem becomes a max over $\mathbf q$ of a min over $\mathbf d,\mathbf a$:</p>
<div class="eq-card"><div class="eq-label">Paper eqs. (9)–(10) · saddle-point problem</div>
$$\max_{\|\mathbf q_{\mathbf u}\|\le1}\;\min_{\mathbf d,\mathbf a}\;E(\mathbf d,\mathbf a,\mathbf q),\qquad E=\langle GA\mathbf d,\mathbf q\rangle+\frac1{2\theta}\|\mathbf d-\mathbf a\|^2+\lambda C(\mathbf a)-\frac\epsilon2\|\mathbf q\|^2$$
<div class="parts">
<span>$\max_{\mathbf q}$</span><span>player "q" pushes $E$ <b>up</b>, one unit-disc vector per pixel</span>
<span>$\min_{\mathbf d,\mathbf a}$</span><span>player "d" pushes $E$ <b>down</b></span>
<span>$\langle GA\mathbf d,\mathbf q\rangle-\frac\epsilon2\|\mathbf q\|^2$</span><span>at q's best reply this equals $\sum g\|\nabla d\|_\epsilon$ (eq. 8)</span>
<span>$\frac1{2\theta}\|\mathbf d-\mathbf a\|^2$</span><span>$\sum_{\mathbf u} (d_{\mathbf u}-a_{\mathbf u})^2/2\theta$</span>
<span>$\lambda C(\mathbf a)$</span><span>$\sum_{\mathbf u}\lambda C(\mathbf u,a_{\mathbf u})$</span>
</div></div>
<ul>
<li><b>q's move:</b> make $q$ point along $g\nabla d$, as far as the disc allows. Where $\mathbf d$ jumps, $q$ saturates at length 1.</li>
<li><b>d's move:</b> with $q$ fixed, the term $\langle GA\mathbf d,\mathbf q\rangle$ is a linear "tax" on jumps in the direction $q$ points, so $\mathbf d$ shrinks its jumps while staying close to $\mathbf a$.</li>
<li>The <b>saddle point</b> is where neither player can improve. There, $\mathbf d$ minimises the original energy.</li>
</ul>
<p>The smallest possible game: 2 pixels, fixed $\mathbf a=(0,1)$, $g=1$. Only one gradient $x=d_1-d_0$ and one dual value $q=q_0$ matter, and (with the average of $\mathbf d$ at its best) $E=qx+\frac{(x-1)^2}{4\theta}-\frac\epsilon2q^2$.</p>
`);

      // ------------------------------------------------------------ widget 4: saddle game
      {
        const fig = L.figure(root, "<b>The saddle-point game.</b> Colour = $E(x,q)$ (bright = high). White curve: q's best reply to each $x$. Pink line: d's best reply to each $q$. Press Step/Run to play the real updates; they spiral into the crossing (the saddle).");
        const c = L.canvas(fig.el, { aspect: 0.8, scroll: true });
        fig.add(c.el);
        const ctl = L.controls(fig.el); fig.add(ctl);
        const sth = L.slider(ctl, { label: "$\\theta$", min: 0.05, max: 0.6, step: 0.01, value: 0.2, oninput: () => reset() });
        const sep = L.slider(ctl, { label: "$\\epsilon$", min: 0.05, max: 0.6, step: 0.01, value: 0.1, oninput: () => reset() });
        const out = L.readout(fig.el); fig.add(out.el);
        const a0 = 0, a1 = 1;
        let d0, d1, q, path, n, running = false;
        const reset = () => { d0 = a0; d1 = a1; q = 0; n = 0; path = [[d1 - d0, q]]; upd(); c.redraw(); };
        const stepOnce = () => {
          const th = sth.value, eps = sep.value;
          // dual: q = Π((q + σq g ∇d)/(1 + σq ε))
          const qn = (q + P.sigmaQ * (d1 - d0)) / (1 + P.sigmaQ * eps);
          q = qn / Math.max(1, Math.abs(qn));
          // primal: div(gq) = (q, −q) for two pixels
          d0 = (d0 + P.sigmaD * (q + a0 / th)) / (1 + P.sigmaD / th);
          d1 = (d1 + P.sigmaD * (-q + a1 / th)) / (1 + P.sigmaD / th);
          n++;
          path.push([d1 - d0, q]);
          if (path.length > 400) path.shift();
        };
        const saddle = () => {
          // x = 1 − 2θq and q = clamp(x/ε): solve
          const th = sth.value, eps = sep.value;
          let qs = 1 / (eps + 2 * th);
          if (qs > 1) qs = 1;
          return [1 - 2 * th * qs, qs];
        };
        const upd = () => {
          const th = sth.value, eps = sep.value, x = d1 - d0;
          const E = q * x + ((d0 - a0) ** 2 + (d1 - a1) ** 2) / (2 * th) - (eps / 2) * q * q;
          const [xs, qs] = saddle();
          out.html = `n = ${n} &nbsp; d = (${fmt(d0)}, ${fmt(d1)}) &nbsp; x = d₁−d₀ = <b>${fmt(x)}</b> &nbsp; q = <b>${fmt(q)}</b> &nbsp; E = ${fmt(E, 4)}<br>saddle: x = ${fmt(xs)}, q = ${fmt(qs)}`;
        };
        let cache = null;
        c.draw = (ctx) => {
          const t = L.theme(), th = sth.value, eps = sep.value;
          const p = L.plot(c, { x0: -0.2, x1: 1.2, y0: -1.25, y1: 1.25, pad: [10, 10, 26, 40] });
          const E = (x, qq) => qq * x + ((x - 1) ** 2) / (4 * th) - (eps / 2) * qq * qq;
          const nx = 56, ny = 40;
          const key = `${c.w}|${c.h}|${th}|${eps}`;
          if (!cache || cache.key !== key) {
            let lo = Infinity, hi = -Infinity;
            const vals = [];
            for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
              const v = E(p.x0 + ((i + 0.5) / nx) * (p.x1 - p.x0), p.y0 + ((j + 0.5) / ny) * (p.y1 - p.y0));
              vals.push(v); lo = Math.min(lo, v); hi = Math.max(hi, v);
            }
            cache = { key, vals, lo, hi };
          }
          const cw = (p.X(p.x1) - p.X(p.x0)) / nx, ch = (p.Y(p.y0) - p.Y(p.y1)) / ny;
          for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
            const v = (cache.vals[j * nx + i] - cache.lo) / (cache.hi - cache.lo);
            ctx.fillStyle = L.viridisish(v);
            ctx.fillRect(p.X(p.x0) + i * cw, p.Y(p.y0) - (j + 1) * ch, cw + 0.6, ch + 0.6);
          }
          ctx.save(); ctx.fillStyle = t.bg; ctx.globalAlpha = 0.65;
          ctx.fillRect(p.X(p.x0), p.Y(1.25), p.X(p.x1) - p.X(p.x0), p.Y(1) - p.Y(1.25));
          ctx.fillRect(p.X(p.x0), p.Y(-1), p.X(p.x1) - p.X(p.x0), p.Y(-1.25) - p.Y(-1)); ctx.restore();
          p.axes(ctx, { xticks: 7, yticks: 5, xlabel: "x = d₁ − d₀", ylabel: "q" });
          const br = [];
          for (let x = p.x0; x <= p.x1 + 1e-9; x += 0.005) br.push([p.X(x), p.Y(clamp(x / eps, -1, 1))]);
          L.draw.path(ctx, br, "#fff", 2.5);
          L.draw.line(ctx, p.X(1 - 2 * th * -1.25), p.Y(-1.25), p.X(1 - 2 * th * 1.25), p.Y(1.25), "#ff5fa2", 2.5);
          const [xs, qs] = saddle();
          L.draw.dot(ctx, p.X(xs), p.Y(qs), 7, "rgba(0,0,0,0)", "#fff");
          L.draw.path(ctx, path.map(([x, qq]) => [p.X(x), p.Y(qq)]), t.accent2, 2);
          const [cx, cq] = path[path.length - 1];
          L.draw.dot(ctx, p.X(cx), p.Y(cq), 6, t.accent2, "#fff");
        };
        L.button(ctl, "Step", () => { stepOnce(); upd(); c.redraw(); });
        const runBtn = L.button(ctl, "Run", () => { running = !running; runBtn.textContent = running ? "Pause" : "Run"; });
        L.button(ctl, "Reset", () => reset());
        let acc = 0;
        L.loop(c, (_, dt) => {
          if (!running) return;
          acc += dt;
          if (acc > 0.12) { acc = 0; stepOnce(); upd(); c.redraw(); }
        });
        reset();
      }

      root.insertAdjacentHTML("beforeend", String.raw`
<p>With $\theta=0.2$ and small $\epsilon$ the saddle has $q=1$ and $x=1-2\theta=0.6$: the jump of $1$ in $\mathbf a$ shrinks to $0.6$ in $\mathbf d$. That shrinking is total-variation smoothing; as $\theta\to0$ it disappears and $\mathbf d$ follows $\mathbf a$.</p>

<h3>4 · The updates (from eqs. 11–12)</h3>
<p>Hold $\mathbf a$ fixed. Differentiate $E$ with respect to each player:</p>
<div class="eq-card"><div class="eq-label">Paper eqs. (11)–(12) · slopes of $E$</div>
$$\frac{\partial E}{\partial\mathbf q}=GA\mathbf d-\epsilon\mathbf q,\qquad \frac{\partial E}{\partial\mathbf d}=A^\top G\mathbf q+\frac1\theta(\mathbf d-\mathbf a)=-\operatorname{div}(g\mathbf q)+\frac1\theta(\mathbf d-\mathbf a)$$
<div class="parts">
<span>$\partial E/\partial\mathbf q$</span><span>q climbs this (it maximises)</span>
<span>$\partial E/\partial\mathbf d$</span><span>d descends this (it minimises); $A^\top G\mathbf q$ comes from $\langle GA\mathbf d,\mathbf q\rangle=\langle\mathbf d,A^\top G\mathbf q\rangle$</span>
</div></div>
<p>Each player takes a step of size $\sigma$. "Semi-implicit" means the pieces that only involve the player's own variable ($-\epsilon\mathbf q$, $\frac1\theta\mathbf d$) are evaluated at the <b>new</b> value, which keeps the step stable:</p>
$$\frac{\mathbf q^{n+1}-\mathbf q^n}{\sigma_q}=GA\mathbf d^n-\epsilon\mathbf q^{n+1},\qquad \frac{\mathbf d^{n+1}-\mathbf d^n}{\sigma_d}=\operatorname{div}(g\mathbf q^{n+1})-\frac1\theta\big(\mathbf d^{n+1}-\mathbf a^n\big)$$
<p>Solve each for the new value:</p>
<div class="eq-card"><div class="eq-label">Dual ascent (dual.wgsl) · every pixel</div>
$$\mathbf q^{n+1}=\Pi\!\left(\frac{\mathbf q^n+\sigma_q\,g\,A\mathbf d^n}{1+\sigma_q\epsilon}\right),\qquad \Pi(\mathbf x)=\frac{\mathbf x}{\max(1,\|\mathbf x\|)}$$
<div class="parts">
<span>$\sigma_q\,gA\mathbf d^n$</span><span>step towards the (weighted) gradient of the <b>old</b> $\mathbf d$</span>
<span>$1+\sigma_q\epsilon$</span><span>the implicit $-\epsilon\mathbf q$ part (tiny: $\epsilon=10^{-4}$ here)</span>
<span>$\Pi$</span><span>projection: if the 2-vector is longer than 1, rescale it to length 1 (direction kept)</span>
</div></div>
<div class="eq-card"><div class="eq-label">Primal descent (primal.wgsl) · every pixel</div>
$$\mathbf d^{n+1}=\frac{\mathbf d^n+\sigma_d\big(\operatorname{div}(g\mathbf q^{n+1})+\frac1\theta\mathbf a^n\big)}{1+\sigma_d/\theta}$$
<div class="parts">
<span>$\operatorname{div}(g\mathbf q^{n+1})$</span><span>uses the <b>new</b> $\mathbf q$; pulls $d$ to flatten jumps</span>
<span>$\frac{\sigma_d}{\theta}\mathbf a^n$ vs. $\frac{\sigma_d}\theta\mathbf d^{n+1}$</span><span>pulls $d$ towards $a$; with small $\theta$ this dominates and $d\approx a$</span>
</div></div>
<p><b>Worked example</b> (1D, 3 pixels, $g=1$, $\sigma_q=0.5$, $\sigma_d=0.25$, $\theta=0.5$; $\epsilon=0.5$ exaggerated so it is visible). Start $\mathbf d=(0,3,3)$, $\mathbf a=(0.5,2,3)$, $\mathbf q=\mathbf 0$.</p>
<ol>
<li>$A\mathbf d=(3,0,0)$. Dual: $q_0=\frac{0+0.5\cdot3}{1+0.5\cdot0.5}=\frac{1.5}{1.25}=1.2$ → projected to $1$. $q_1=q_2=0$.</li>
<li>$\operatorname{div}\mathbf q=(q_0,\;q_1-q_0,\;-q_1)=(1,-1,0)$. Divisor $1+0.25/0.5=1.5$.</li>
<li>$d_0=\frac{0+0.25(1+0.5/0.5)}{1.5}=0.333$, $d_1=\frac{3+0.25(-1+2/0.5)}{1.5}=2.5$, $d_2=\frac{3+0.25(0+3/0.5)}{1.5}=3$.</li>
</ol>
<p>The jump $d_1-d_0$ fell from $3$ to $2.17$ in one step.</p>
<div class="note"><b>Step sizes.</b> The iteration is guaranteed to converge when $\sigma_q\sigma_d\|A\|^2\le1$. The squared "stretch" $\|A\|^2$ (largest value of $\|A\mathbf d\|^2/\|\mathbf d\|^2$) of forward differences is at most $4$ per direction: $4$ in 1D, $8$ in 2D. Example: $\mathbf d=(1,-1,1,-1,\dots)$ has every difference $\pm2$, so $\|A\mathbf d\|^2\approx4\|\mathbf d\|^2$. This implementation uses $\sigma_q=0.5$, $\sigma_d=0.25$: $0.5\cdot0.25\cdot8=1$. Since $g\le1$, weighting cannot break this. The paper tunes the steps per iteration following its ref. [3]; here that acceleration exists (<code>accelerate</code>) but is <b>off</b>, so the steps stay fixed.</div>

<h3>5 · The point-wise search for $\mathbf a$ (eqs. 13–14)</h3>
<p>Now hold $\mathbf d$ fixed. In $E$ only two terms contain $\mathbf a$, and neither links neighbours, so every pixel is solved on its own:</p>
<div class="eq-card"><div class="eq-label">Paper eqs. (13)–(14) · auxiliary search (aux.wgsl)</div>
$$a_{\mathbf u}=\arg\min_{a\in\mathcal D}E_{\text{aux}}(\mathbf u,d_{\mathbf u},a),\qquad E_{\text{aux}}(\mathbf u,d_{\mathbf u},a)=\frac{1}{2\theta}(d_{\mathbf u}-a)^2+\lambda C(\mathbf u,a)$$
<div class="parts">
<span>$\mathcal D$</span><span>the $S$ sampled inverse depths $\xi_k=\xi_{\min}+k\,\Delta\xi$, $k=0..S-1$</span>
<span>$\frac1{2\theta}(d-a)^2$</span><span>a parabola centred on the current $d_{\mathbf u}$; narrower as $\theta$ shrinks</span>
<span>$\lambda C(\mathbf u,a)$</span><span>the pixel's cost row (chapter 8). Unobserved samples count as $C_{\max}$</span>
</div></div>
<p>The data term is non-convex (several dips), so we cannot follow a slope. But it is one number per sample: just <b>try every sample</b> and keep the lowest. That makes each step globally optimal (up to the sampling).</p>
<p><b>The band (eqs. 15–17).</b> We do not even need all $S$ samples. Choosing $a=d_{\mathbf u}$ costs at most $\lambda C_{\max}$ (the parabola is 0 there). Any $a$ with lower energy must satisfy</p>
<div class="eq-card"><div class="eq-label">Paper eqs. (15)–(17) · search band</div>
$$\lambda C_{\min}+\frac{1}{2\theta}(a-d_{\mathbf u})^2\;\le\;\lambda C_{\max}\quad\Longrightarrow\quad |a-d_{\mathbf u}|\le r=\sqrt{2\theta\lambda\,(C_{\max}-C_{\min})}$$
<div class="parts">
<span>$\lambda C_{\min}$</span><span>the data term can never be lower than this</span>
<span>$\lambda C_{\max}$</span><span>an upper bound on the energy of the best $a$ (achieved by $a=d$ or better)</span>
<span>$r$</span><span>half-width of the band around $d_{\mathbf u}$; shrinks like $\sqrt\theta$</span>
<span>in code</span><span>layers $k_0=\lfloor(d-r-\xi_{\min})/\Delta\xi\rfloor$ to $k_1=\lceil(d+r-\xi_{\min})/\Delta\xi\rceil$, clamped to $[0,S-1]$</span>
</div></div>
<p>The paper's printed (15) drops the $\lambda$; (17) and the code include it. Example: $\lambda=1$, $C_{\max}-C_{\min}=0.25$, $\theta=0.02$ gives $r=\sqrt{2\cdot0.02\cdot0.25}=0.1$. With $\xi_{\min}=0.2$, $\Delta\xi=0.05$ and $d=1.23$: $k_0=\lfloor 0.93/0.05\rfloor=18$, $k_1=\lceil1.13/0.05\rceil=23$, so only 6 layers are tested.</p>
`);

      // ------------------------------------------------------------ widget 5: point-wise search
      const rowPresets = {
        textured: (x) => 0.08 + 0.5 * (1 - Math.exp(-((x - 1.1) ** 2) / (2 * 0.1 * 0.1))) + 0.03 * Math.sin(9 * x),
        "two dips": (x) => 0.1 + 0.45 * Math.min(1 - Math.exp(-((x - 0.7) ** 2) / (2 * 0.07 * 0.07)), 1 - 0.85 * Math.exp(-((x - 1.5) ** 2) / (2 * 0.07 * 0.07))),
        flat: (x) => 0.3 + 0.02 * Math.sin(7 * x + 1) + 0.015 * Math.sin(23 * x),
      };
      {
        let preset = "two dips", dcur = 1.25, anim = false;
        const fig = L.figure(root, "<b>Point-wise search.</b> Drag the handle on the axis to move $d_{\\mathbf u}$. Red: $\\lambda C$. Blue: $(d-a)^2/2\\theta$. Green: their sum $E_\\text{aux}$. Only samples in the unshaded band are tested. Shrink $\\theta$ and watch the band close in.");
        const c = L.canvas(fig.el, { aspect: 0.72 });
        fig.add(c.el);
        const ctl = L.controls(fig.el); fig.add(ctl);
        for (const k of Object.keys(rowPresets)) L.button(ctl, k, () => { preset = k; c.redraw(); upd(); });
        const ctl2 = L.controls(fig.el); fig.add(ctl2);
        const sth = L.slider(ctl2, { label: "$\\theta$", min: -4, max: Math.log10(0.2), step: 0.01, value: -1.3, fmt: (v) => fmt(10 ** v, 4), oninput: () => { c.redraw(); upd(); } });
        const sl = L.slider(ctl2, { label: "$\\lambda$", min: 0.2, max: 2, step: 0.05, value: 1, oninput: () => { c.redraw(); upd(); } });
        const animBtn = L.button(ctl2, "Run θ schedule", () => { anim = true; sth.value = Math.log10(0.2); animIter = 0; });
        const out = L.readout(fig.el); fig.add(out.el);
        let animIter = 0;
        const solve = () => {
          const f = rowPresets[preset], th = 10 ** sth.value, lam = sl.value;
          const C = Array.from({ length: S }, (_, k) => f(XMIN + k * DX));
          const cmin = Math.min(...C), cmax = Math.max(...C);
          const E = C.map((cv, k) => ((dcur - (XMIN + k * DX)) ** 2) / (2 * th) + lam * cv);
          const r = Math.sqrt(2 * th * lam * (cmax - cmin));
          const k0 = clamp(Math.floor((dcur - r - XMIN) / DX), 0, S - 1), k1 = clamp(Math.ceil((dcur + r - XMIN) / DX), 0, S - 1);
          let best = k0;
          for (let k = k0; k <= k1; k++) if (E[k] < E[best]) best = k;
          let gbest = 0;
          for (let k = 0; k < S; k++) if (E[k] < E[gbest]) gbest = k;
          let ahat = XMIN + best * DX, hess = NaN;
          if (best > 0 && best < S - 1) {
            const grad = (E[best + 1] - E[best - 1]) / (2 * DX);
            hess = (E[best + 1] - 2 * E[best] + E[best - 1]) / (DX * DX);
            if (hess > 0) ahat -= clamp(grad / hess, -DX, DX);
          }
          return { C, E, cmin, cmax, r, k0, k1, best, gbest, ahat, th, lam };
        };
        const upd = () => {
          const s = solve();
          out.html = `r = √(2·${fmt(s.th, 4)}·${fmt(s.lam)}·${fmt(s.cmax - s.cmin, 3)}) = <b>${fmt(s.r, 3)}</b> &nbsp; tested layers ${s.k0}…${s.k1} (<b>${s.k1 - s.k0 + 1}</b> of ${S})<br>` +
            `best sample a = ${fmt(XMIN + s.best * DX, 3)} (layer ${s.best}${s.best === s.gbest ? ", same as full search ✓" : ", full search: " + s.gbest}) &nbsp; Newton â = <b>${fmt(s.ahat, 3)}</b>`;
        };
        c.draw = (ctx) => {
          const t = L.theme(), s = solve();
          const ymax = s.lam * s.cmax * 1.35 + 0.02;
          const p = L.plot(c, { x0: XMIN, x1: XMAX, y0: 0, y1: ymax, pad: [12, 12, 40, 40] });
          p.axes(ctx, { xticks: 6, yticks: 4, xlabel: "a (inverse depth)" });
          const bl = clamp(dcur - s.r, XMIN, XMAX), br = clamp(dcur + s.r, XMIN, XMAX);
          ctx.save(); ctx.fillStyle = t.faint; ctx.globalAlpha = 0.18;
          ctx.fillRect(p.X(XMIN), p.Y(ymax), p.X(bl) - p.X(XMIN), p.Y(0) - p.Y(ymax));
          ctx.fillRect(p.X(br), p.Y(ymax), p.X(XMAX) - p.X(br), p.Y(0) - p.Y(ymax)); ctx.restore();
          L.draw.line(ctx, p.X(XMIN), p.Y(s.lam * s.cmax), p.X(XMAX), p.Y(s.lam * s.cmax), t.bad, 1, [4, 4]);
          L.draw.text(ctx, "λCmax", p.X(XMAX) - 4, p.Y(s.lam * s.cmax) - 4, t.bad, { size: 11, align: "right" });
          L.draw.line(ctx, p.X(XMIN), p.Y(s.lam * s.cmin), p.X(XMAX), p.Y(s.lam * s.cmin), t.bad, 1, [2, 4]);
          const par = [];
          for (let x = XMIN; x <= XMAX + 1e-9; x += 0.005) {
            const v = ((dcur - x) ** 2) / (2 * s.th);
            if (v <= ymax) par.push([p.X(x), p.Y(v)]);
            else if (par.length && x > dcur) break;
            else par.length = 0;
          }
          L.draw.path(ctx, par, t.accent, 2);
          L.draw.path(ctx, s.C.map((cv, k) => [p.X(XMIN + k * DX), p.Y(s.lam * cv)]), t.bad, 1.5);
          const ep = s.E.map((e, k) => [p.X(XMIN + k * DX), p.Y(Math.min(e, ymax))]);
          L.draw.path(ctx, ep, t.accent3, 2);
          for (let k = 0; k < S; k++) {
            const inside = k >= s.k0 && k <= s.k1;
            L.draw.dot(ctx, ep[k][0], ep[k][1], inside ? 3.5 : 2, inside ? t.accent3 : t.faint);
          }
          L.draw.dot(ctx, ep[s.best][0], ep[s.best][1], 7, "rgba(0,0,0,0)", t.fg);
          L.draw.line(ctx, p.X(s.ahat), p.Y(0), p.X(s.ahat), p.Y(ymax), t.accent2, 1.5, [3, 3]);
          L.draw.text(ctx, "â", p.X(s.ahat) + 4, p.Y(ymax) + 12, t.accent2, { size: 12, bold: true });
          L.draw.line(ctx, p.X(dcur), p.Y(0), p.X(dcur), p.Y(ymax), t.accent, 1, [2, 3]);
          L.draw.handle(ctx, p.X(dcur), p.Y(0), t.accent);
          L.draw.text(ctx, "d", p.X(dcur), p.Y(0) + 28, t.accent, { size: 12, bold: true, align: "center" });
        };
        L.drag(c, () => { const p = L.plot(c, { x0: XMIN, x1: XMAX, y0: 0, y1: 1, pad: [12, 12, 40, 40] }); return [{ x: p.X(dcur), y: p.Y(0) }]; },
          (_, pt) => { const p = L.plot(c, { x0: XMIN, x1: XMAX, y0: 0, y1: 1, pad: [12, 12, 40, 40] }); dcur = clamp(p.invX(pt.x), XMIN, XMAX); upd(); });
        L.loop(c, () => {
          if (!anim) return;
          // paper-speed schedule, a few iterations per frame
          for (let k = 0; k < 3; k++) {
            const th = 10 ** sth.value;
            const beta = th >= 1e-3 ? 1e-3 : 1e-4;
            const nth = th * (1 - beta * animIter++);
            if (nth <= 1e-4) { anim = false; break; }
            sth.value = Math.log10(nth);
          }
          c.redraw(); upd();
        });
        void animBtn;
        upd();
      }

      root.insertAdjacentHTML("beforeend", String.raw`
<p>Things to notice: with large $\theta$ the parabola is wide and the search jumps to the <b>global</b> dip even far from $d$; with small $\theta$ the band shrinks to a few samples around $d$ and $a$ can only fine-tune. On the flat row the data term barely matters, so $a\approx d$: textureless pixels are filled by the regulariser.</p>

<h3>6 · Sub-sample accuracy: one Newton step (eq. 18)</h3>
<p>The best sample is only accurate to $\pm\Delta\xi/2$. Near its minimum $E_{\text{aux}}$ looks like a parabola, so fit one through the best sample and its two neighbours and jump to its bottom:</p>
<div class="eq-card"><div class="eq-label">Paper eq. (18) · Newton step on the sampled energy (aux.wgsl)</div>
$$\hat a=a-\frac{E'_{\text{aux}}}{E''_{\text{aux}}},\qquad E'\approx\frac{E_+-E_-}{2\Delta\xi},\qquad E''\approx\frac{E_+-2E_0+E_-}{\Delta\xi^2}$$
<div class="parts">
<span>$E_0$, $E_\pm$</span><span>$E_{\text{aux}}$ at the best sample $a$ and at $a\pm\Delta\xi$</span>
<span>$E'$, $E''$</span><span>slope and curvature from finite differences (central, as in chapter 1)</span>
<span>in code</span><span>skipped if $E''\le0$ or the best sample is the first/last layer; the step is clamped to $\pm\Delta\xi$</span>
</div></div>
<p>Example: $E_-=3$, $E_0=1$, $E_+=2$, $\Delta\xi=0.1$: $E'=\frac{2-3}{0.2}=-5$, $E''=\frac{2-2+3}{0.01}=300$, step $=-(-5/300)=+0.0167$. The minimum lies towards the lower neighbour, as it should. Equivalently: $\delta=\Delta\xi\frac{E_--E_+}{2(E_--2E_0+E_+)}$.</p>
`);

      // ------------------------------------------------------------ widget 6: Newton step
      {
        const E = [3, 1, 2];
        const fig = L.figure(root, "<b>Parabola through three samples.</b> Drag the three dots up and down. The orange arrow is the Newton step $-E'/E''$ (in units of $\\Delta\\xi$), clamped to $\\pm1$.");
        const c = L.canvas(fig.el, { aspect: 0.62 });
        fig.add(c.el);
        const out = L.readout(fig.el); fig.add(out.el);
        const plotOf = () => L.plot(c, { x0: -2.2, x1: 2.2, y0: 0, y1: 5, pad: [12, 12, 28, 34] });
        const upd = () => {
          const g = (E[2] - E[0]) / 2, h = E[2] - 2 * E[1] + E[0];
          const raw = h > 0 ? -g / h : NaN;
          out.html = `E′ = (E₊−E₋)/2Δ = ${fmt(g, 3)}/Δ &nbsp; E″ = (E₊−2E₀+E₋)/Δ² = ${fmt(h, 3)}/Δ²<br>` +
            (h > 0 ? `step −E′/E″ = <b>${fmt(raw, 3)} Δ</b>${Math.abs(raw) > 1 ? " → clamped to " + (raw > 0 ? "+1" : "−1") + " Δ" : ""}` : "<b>E″ ≤ 0: no parabola minimum, the code keeps the sample</b>");
        };
        c.draw = (ctx) => {
          const t = L.theme(), p = plotOf();
          p.axes(ctx, { xticks: 4, yticks: 5, fmt: (v) => +v.toFixed(1), xlabel: "offset from best sample (Δξ)" });
          const g = (E[2] - E[0]) / 2, h = E[2] - 2 * E[1] + E[0];
          const par = (x) => E[1] + g * x + 0.5 * h * x * x;
          const pts = [];
          for (let x = -2.2; x <= 2.2001; x += 0.02) { const y = par(x); if (y > -0.5 && y < 5.5) pts.push([p.X(x), p.Y(y)]); }
          L.draw.path(ctx, pts, t.accent3, 2);
          if (h > 0) {
            const raw = -g / h, st = clamp(raw, -1, 1);
            L.draw.line(ctx, p.X(raw), p.Y(0), p.X(raw), p.Y(par(raw)), t.faint, 1, [3, 3]);
            L.draw.arrow(ctx, p.X(0), p.Y(0.3), p.X(st), p.Y(0.3), t.accent2, 3);
            L.draw.text(ctx, "â", p.X(st), p.Y(0.3) - 8, t.accent2, { size: 13, bold: true, align: "center" });
          }
          ["a−Δ", "a", "a+Δ"].forEach((lab, i) => {
            L.draw.handle(ctx, p.X(i - 1), p.Y(E[i]), t.accent);
            L.draw.text(ctx, lab, p.X(i - 1), p.Y(E[i]) - 14, t.fg, { size: 12, align: "center" });
          });
        };
        L.drag(c, () => { const p = plotOf(); return E.map((v, i) => ({ x: p.X(i - 1), y: p.Y(v) })); },
          (i, pt) => { const p = plotOf(); E[i] = Math.round(clamp(p.invY(pt.y), 0.1, 4.9) * 20) / 20; upd(); });
        upd();
      }

      root.insertAdjacentHTML("beforeend", String.raw`
<p>If the best sample really is the lowest of the three, $|E_--E_+|\le(E_--E_0)+(E_+-E_0)$, so the step is at most $\Delta\xi/2$: the clamp is only a safety net. The paper stresses that this step must happen <b>inside</b> every iteration: at the end $\theta$ is tiny, the coupling parabola is a narrow spike at $d$, and a fitted parabola would just return $d$.</p>

<h3>7 · Driving θ to zero</h3>
<div class="eq-card"><div class="eq-label">θ schedule (paper §2.2.3 step 3, §2.2.6; mapping.rs)</div>
$$\theta_{n+1}=\theta_n\,(1-\beta\,n),\qquad \theta_0=0.2,\quad\text{stop when }\theta\le\theta_{\text{end}}=10^{-4}$$
<div class="parts">
<span>$n$</span><span>iteration counter, starting at 0 (so $\theta_1=\theta_0$)</span>
<span>$\beta$</span><span>$2.5\cdot10^{-4}$ while $\theta\ge10^{-3}$, then $2.5\cdot10^{-5}$ (this implementation)</span>
<span>paper's $\beta$</span><span>$10^{-3}$ / $10^{-4}$: four times larger, fewer iterations</span>
</div></div>
<p>Because the factor is $1-\beta n$, the decay speeds up with $n$: slow at first (large $\theta$, the search roams freely), fast later. Example with an exaggerated $\beta=0.05$: $\theta_1=0.2$, $\theta_2=0.2(1-0.05)=0.19$, $\theta_3=0.19(1-0.1)=0.171$.</p>
`);

      // ------------------------------------------------------------ widget 7: θ schedule
      {
        const count = (bf, bs) => {
          let th = P.theta0, it = 0;
          const pts = [[0, th]];
          while (th > P.thetaEnd) {
            const b = th >= P.thetaSwitch ? bf : bs;
            const f = 1 - b * it; it++;
            if (f <= 0 || it > 5000) break;
            th *= f; pts.push([it, th]);
          }
          return { it, pts };
        };
        const code = count(P.betaFast, P.betaSlow), paper = count(1e-3, 1e-4);
        const fig = L.figure(root, "<b>θ schedules.</b> Slide $\\beta$ (the slow phase uses $\\beta/10$). Dashed: the paper's and this implementation's values. Log scale on θ.");
        const c = L.canvas(fig.el, { aspect: 0.55, scroll: true });
        fig.add(c.el);
        const ctl = L.controls(fig.el); fig.add(ctl);
        const sb = L.slider(ctl, { label: "$\\beta$", min: -4.5, max: -2, step: 0.01, value: Math.log10(P.betaFast), fmt: (v) => (10 ** v).toExponential(1), oninput: () => { c.redraw(); upd(); } });
        const out = L.readout(fig.el); fig.add(out.el);
        const upd = () => {
          const cur = count(10 ** sb.value, 10 ** sb.value / 10);
          out.html = `this β: <b>${cur.it}</b> iterations &nbsp; paper (1e-3): ${paper.it} &nbsp; this implementation (2.5e-4): ${code.it}`;
        };
        c.draw = (ctx) => {
          const t = L.theme(), cur = count(10 ** sb.value, 10 ** sb.value / 10);
          const xmax = Math.max(cur.it, code.it) * 1.05;
          const p = L.plot(c, { x0: 0, x1: xmax, y0: -4.2, y1: -0.5, pad: [12, 12, 26, 44] });
          p.axes(ctx, { xticks: 4, yticks: 4, fmt: (v) => Math.round(v), xlabel: "iteration n" });
          L.draw.text(ctx, "log₁₀ θ", 48, 24, t.muted, { size: 11 });
          L.draw.line(ctx, p.X(0), p.Y(-3), p.X(xmax), p.Y(-3), t.faint, 1, [2, 4]);
          L.draw.text(ctx, "θ = 1e-3 (β switch)", p.X(xmax) - 4, p.Y(-3) - 4, t.faint, { size: 11, align: "right" });
          const P2 = (s) => s.pts.map(([n, th]) => [p.X(n), p.Y(Math.log10(th))]);
          L.draw.path(ctx, P2(paper), t.accent4, 1.5, [5, 4]);
          L.draw.path(ctx, P2(code), t.accent3, 1.5, [5, 4]);
          L.draw.path(ctx, P2(cur), t.accent, 2.5);
        };
        upd();
      }

      root.insertAdjacentHTML("beforeend", String.raw`
<p>The paper notes that smaller $\beta$ with more iterations gives better results. This implementation uses <b>¼ of the paper's $\beta$</b> (about 474 iterations instead of 236), which gives the regulariser time to fill textureless regions.</p>
<div class="tblwrap"><table class="tbl">
<tr><th>parameter</th><th>value (mod.rs)</th><th>role</th></tr>
<tr><td>$\theta_0$, $\theta_{\text{end}}$</td><td>0.2, $10^{-4}$</td><td>coupling schedule start / stop</td></tr>
<tr><td>$\beta$</td><td>$2.5\cdot10^{-4}$ ($\theta\ge10^{-3}$), $2.5\cdot10^{-5}$ after</td><td>schedule speed (paper: $10^{-3}$, $10^{-4}$)</td></tr>
<tr><td>$\sigma_q$, $\sigma_d$</td><td>0.5, 0.25 (fixed)</td><td>dual / primal step sizes</td></tr>
<tr><td>$\epsilon$</td><td>$10^{-4}$</td><td>Huber threshold</td></tr>
<tr><td>$\alpha$, $\beta_g$</td><td>100, 1.6</td><td>edge weight $g$ (eq. 5)</td></tr>
<tr><td>$\lambda$</td><td>$1/(1+0.5\,\bar d)$, $\bar d$ = nearest predicted depth; 1 for the first keyframe</td><td>data weight (paper §2.2.6)</td></tr>
<tr><td>$S$</td><td>64 (32 on mobile)</td><td>inverse-depth samples</td></tr>
</table></div>
<p>The $\lambda$ rule: a nearer scene gives a sharper data term, so trust it more. Example: nearest depth $\bar d=2$ m gives $\lambda=1/(1+1)=0.5$.</p>

<h3>8 · Where to start: initialisation</h3>
<p>The paper starts from $\mathbf q^0=\mathbf 0$ and $\mathbf d^0=\mathbf a^0=\arg\min C$ everywhere. In a textureless pixel the arg min is noise, and the solver must then undo it. This implementation instead:</p>
<ol>
<li>computes for each pixel $C_{\min}$, $C_{\max}$, the arg min and the <b>trough width</b>: the number of layers from the first to the last sample with $C\le C_{\min}+\tau$, $\tau=\max(0.01,\;0.1(C_{\max}-C_{\min}))$ (<code>cost_minmax.wgsl</code>);</li>
<li>uses the arg min where the trough is at most <b>6 layers</b> wide (a localised minimum);</li>
<li>fills every other pixel by <b>push–pull</b>: <i>pull</i> = repeatedly halve the resolution, summing value×weight and weight over 2×2 blocks; <i>push</i> = go back up, giving each empty pixel its coarser block's weighted mean. The result is a smooth interpolation of the confident pixels (<code>fill_unconstrained</code> in mapping.rs).</li>
</ol>
<p>Example: a 6-layer row $C=(0.40,0.30,0.12,0.10,0.11,0.35)$ with $C_{\min}=0.10$, $C_{\max}=0.40$ has $\tau=0.03$; samples $\le0.13$ are layers 2–4, width 3 → localised.</p>

<h3>9 · The whole solver</h3>
<pre><code>solve(C, g, λ):
  for each pixel u:                                   # cost_minmax.wgsl
      Cmin, Cmax, argmin, trough width
  d = a = argmin where width ≤ 6 layers,
          push–pull fill elsewhere                    # fill_unconstrained
  q = 0;  θ = 0.2;  n = 0
  while θ > 1e-4:
      for each pixel u:                               # dual.wgsl
          ∇d = (d[right] − d, d[down] − d)            # 0 on last column / row
          q  = (q + σq·g·∇d) / (1 + σq·ε)
          q  = q / max(1, |q|)
      for each pixel u:                               # primal.wgsl
          div = g·qx − g[left]·qx[left] + g·qy − g[up]·qy[up]   # missing terms = 0
          d   = (d + σd·(div + a/θ)) / (1 + σd/θ)
      for each pixel u:                               # aux.wgsl
          r  = sqrt(2·θ·λ·(Cmax − Cmin))
          k* = argmin over k in [floor((d−r−ξmin)/Δξ), ceil((d+r−ξmin)/Δξ)]
                 of (d − ξk)²/(2θ) + λ·C(u,k)        # unobserved: C = Cmax
          a  = ξk*
          if 0 < k* < S−1 and E'' > 0:
              a = a − clamp(E'/E'', −Δξ, Δξ)          # eq. 18
      β = (θ ≥ 1e-3) ? 2.5e-4 : 2.5e-5
      θ = θ·(1 − β·n);  n = n + 1
  return d</code></pre>
<p>Each <code>for each pixel</code> is one GPU compute pass; all three run back-to-back without reading anything back to the CPU. Below is exactly this loop in JavaScript on a 1D scene (64 pixels, 32 layers, 6 extra frames with a real photometric cost volume). The middle object has a <b>textureless</b> centre: its cost rows are flat.</p>
`);

      // ------------------------------------------------------------ widget 8: full solver
      {
        let seed = 5;
        const mkRand = (s) => { const r = L.rng(s); return () => r.next(); };
        let sc = SOLVER.makeScene(mkRand(seed));
        let st = null, running = false, lambda = 1, newton = true, band = true, pushpull = true, showTruth = true;
        let heat = null;
        const thetaHist = [];
        const fig = L.figure(root, "<b>The full solver.</b> Background: the cost volume (pixel → right, inverse depth ↑; dark = low cost, each column normalised). Grey dots: arg min. Orange: $\\mathbf a$. Blue: $\\mathbf d$. Dashed: truth. Strips: $I_r$ and $g$. Step, Run or Finish, then try the toggles and Reset.");
        const c = L.canvas(fig.el, { aspect: 0.85, scroll: true });
        fig.add(c.el);
        const c2 = L.canvas(fig.el, { aspect: 0.28, scroll: true });
        fig.add(c2.el);
        const ctl = L.controls(fig.el); fig.add(ctl);
        L.button(ctl, "Step", () => { doSteps(1); });
        const runBtn = L.button(ctl, "Run", () => { if (st.done) return; running = !running; runBtn.textContent = running ? "Pause" : "Run"; });
        L.button(ctl, "Finish", () => { doSteps(10000); });
        L.button(ctl, "Reset", () => reset());
        L.button(ctl, "New scene", () => { seed++; sc = SOLVER.makeScene(mkRand(seed)); heat = null; reset(); });
        const ctl2 = L.controls(fig.el); fig.add(ctl2);
        L.slider(ctl2, { label: "$\\lambda$", min: 0.1, max: 3, step: 0.05, value: 1, oninput: (v) => { lambda = v; reset(); } });
        L.toggle(ctl2, "push–pull init", true, (v) => { pushpull = v; reset(); });
        L.toggle(ctl2, "Newton step", true, (v) => { newton = v; });
        L.toggle(ctl2, "band search", true, (v) => { band = v; });
        L.toggle(ctl2, "truth", true, (v) => { showTruth = v; c.redraw(); });
        const out = L.readout(fig.el); fig.add(out.el);
        const rms = (arr) => Math.sqrt(arr.reduce((s, v, i) => s + (v - sc.xi[i]) ** 2, 0) / N);
        const upd = () => {
          const E = SOLVER.energy(sc, st, lambda);
          out.html = `n = <b>${st.iter}</b> &nbsp; θ = <b>${st.theta.toExponential(2)}</b> &nbsp; β = ${st.theta >= P.thetaSwitch ? "2.5e-4" : "2.5e-5"}${st.done ? " &nbsp; <b>done</b>" : ""}<br>` +
            `samples tested / pixel: ${fmt(st.tested, 1)} of ${S} &nbsp; mean |d−a| = ${fmt(st.d.reduce((s, v, i) => s + Math.abs(v - st.a[i]), 0) / N, 4)}<br>` +
            `energy (7) = ${fmt(E, 3)} &nbsp; RMS error: arg min ${fmt(rms(sc.argmin), 3)}, d ${fmt(rms(st.d), 3)}`;
        };
        const reset = () => {
          running = false; runBtn.textContent = "Run";
          st = SOLVER.init(sc, pushpull);
          thetaHist.length = 0; thetaHist.push(st.theta);
          upd(); c.redraw(); c2.redraw();
        };
        const doSteps = (k) => {
          for (let i = 0; i < k && !st.done; i++) { SOLVER.step(sc, st, { lambda, newton, band }); thetaHist.push(st.theta); }
          if (st.done) { running = false; runBtn.textContent = "Run"; }
          upd(); c.redraw(); c2.redraw();
        };
        const buildHeat = () => {
          const cv = document.createElement("canvas");
          cv.width = N; cv.height = S;
          const cx = cv.getContext("2d");
          const id = cx.createImageData(N, S);
          for (let k = 0; k < S; k++) for (let u = 0; u < N; u++) {
            const range = Math.max(1e-6, sc.cmax[u] - sc.cmin[u]);
            const v = sc.obs[k * N + u] ? (sc.C[k * N + u] - sc.cmin[u]) / range : 1;
            const [r, g, b] = L.viridisish(0.1 + 0.8 * v).match(/\d+/g).map(Number);
            const o = 4 * ((S - 1 - k) * N + u);
            id.data[o] = r; id.data[o + 1] = g; id.data[o + 2] = b; id.data[o + 3] = 255;
          }
          cx.putImageData(id, 0, 0);
          return cv;
        };
        c.draw = (ctx) => {
          const t = L.theme();
          if (!heat) heat = buildHeat();
          const pl = 34, pr = 6, top = 6, stripH = 12, hb = c.h - 2 * stripH - 30;
          const X = (u) => pl + ((u + 0.5) / N) * (c.w - pl - pr);
          const Y = (x) => top + (1 - (x - XMIN + DX / 2) / (XMAX - XMIN + DX)) * (hb - top);
          ctx.save(); ctx.imageSmoothingEnabled = false; ctx.globalAlpha = 0.8;
          ctx.drawImage(heat, pl, top, c.w - pl - pr, hb - top); ctx.restore();
          for (const v of [0.5, 1, 1.5, 2]) L.draw.text(ctx, String(v), pl - 4, Y(v) + 4, t.faint, { size: 11, align: "right" });
          L.draw.text(ctx, "ξ", 4, top + 12, t.muted, { size: 12 });
          if (showTruth) L.draw.path(ctx, Array.from(sc.xi, (v, u) => [X(u), Y(v)]), t.fg, 1.5, [4, 3]);
          for (let u = 0; u < N; u++) L.draw.dot(ctx, X(u), Y(sc.argmin[u]), 2.2, "rgba(220,220,220,0.9)");
          for (let u = 0; u < N; u++) L.draw.dot(ctx, X(u), Y(st.a[u]), 2.6, t.accent2);
          L.draw.path(ctx, Array.from(st.d, (v, u) => [X(u), Y(v)]), t.accent, 2.5);
          // strips: I_r and g
          const cw = (c.w - pl - pr) / N;
          for (let u = 0; u < N; u++) {
            ctx.fillStyle = L.gray(sc.I[u]);
            ctx.fillRect(pl + u * cw, hb + 6, cw + 0.5, stripH);
            ctx.fillStyle = L.gray(sc.g[u]);
            ctx.fillRect(pl + u * cw, hb + 10 + stripH, cw + 0.5, stripH);
          }
          L.draw.text(ctx, "Iᵣ", pl - 4, hb + 6 + stripH - 2, t.muted, { size: 11, align: "right" });
          L.draw.text(ctx, "g", pl - 4, hb + 10 + 2 * stripH - 2, t.muted, { size: 11, align: "right" });
          L.draw.text(ctx, "textureless", X(31.5), hb + 10 + 2 * stripH + 13, t.muted, { size: 11, align: "center" });
          L.draw.line(ctx, X(25.5), hb + 10 + 2 * stripH + 3, X(37.5), hb + 10 + 2 * stripH + 3, t.muted, 1.5);
        };
        c2.draw = (ctx) => {
          const t = L.theme();
          const xmax = 480;
          const p = L.plot(c2, { x0: 0, x1: xmax, y0: -4.2, y1: -0.5, pad: [8, 8, 22, 34] });
          p.axes(ctx, { xticks: 4, yticks: 2, fmt: (v) => Math.round(v), xlabel: "n" });
          L.draw.text(ctx, "log₁₀θ", 38, 20, t.muted, { size: 11 });
          L.draw.path(ctx, thetaHist.map((th, i) => [p.X(Math.min(i, xmax)), p.Y(Math.log10(th))]), t.accent, 2);
          const i = thetaHist.length - 1;
          L.draw.dot(ctx, p.X(Math.min(i, xmax)), p.Y(Math.log10(thetaHist[i])), 4.5, t.accent2);
        };
        L.loop(c, () => { if (running) doSteps(4); });
        reset();
      }

      root.insertAdjacentHTML("beforeend", String.raw`
<p>What to try:</p>
<ul>
<li><b>Band search off/on</b>: the result is identical, but with the band only about 7 of the 32 samples are tested per pixel at the start and 2–3 at the end. The bound is exact, not an approximation.</li>
<li><b>Push–pull off</b>: the textureless centre starts from noisy arg mins; the regulariser must pull it flat from its edges.</li>
<li><b>λ</b> (then Finish): smaller λ trusts the regulariser more (flatter, more staircase-like pieces); larger λ follows the data more closely, including its mistakes.</li>
<li>The jumps at the object edges survive because $g$ is nearly $0$ there (dark in the $g$ strip): TV is cheap across image edges.</li>
</ul>
<div class="note"><b>Paper vs. this implementation.</b> Same algorithm, with these choices: $\beta$ at ¼ of the paper's value; fixed step sizes (no acceleration); push–pull initialisation in non-localised pixels; unobserved cost samples count as $C_{\max}$; the Newton step is clamped to $\pm\Delta\xi$ and skipped at the end layers or when $E''\le0$; $\mathbf q$ is projected onto the Euclidean unit disc; the gradient is forward differences with $0$ on the far border and the divergence is its exact negative transpose.</div>
`);

      // ------------------------------------------------------------ quiz
      const vec = (a) => "(" + a.map((v) => fmt(v)).join(",\\ ") + ")";
      L.quiz(root, "primaldual", [
        { id: "grad1d", type: "num",
          gen: (r) => {
            const d = Array.from({ length: 5 }, () => r.int(0, 9) / 2);
            const g = d.map((v, i) => (i < 4 ? d[i + 1] - v : 0));
            return { q: String.raw`1D, forward differences with the zero border rule. $\mathbf d=${vec(d)}$ (pixels 0–4). Give $(A\mathbf d)_1$, $(A\mathbf d)_3$ and $(A\mathbf d)_4$.`,
              answer: [g[1], g[3], g[4]], labels: ["$(A\\mathbf d)_1$", "$(A\\mathbf d)_3$", "$(A\\mathbf d)_4$"], tol: 1e-6,
              explain: String.raw`$(A\mathbf d)_1=d_2-d_1=${fmt(d[2])}-${fmt(d[1])}=${fmt(g[1])}$, $(A\mathbf d)_3=d_4-d_3=${fmt(d[4])}-${fmt(d[3])}=${fmt(g[3])}$, and $(A\mathbf d)_4=0$: the last pixel has no right neighbour, so its row of $A$ is zero.` };
          } },
        { id: "div1d", type: "num",
          gen: (r) => {
            const q = Array.from({ length: 5 }, () => r.int(-10, 10) / 10);
            const dv = q.map((_, i) => (i < 4 ? q[i] : 0) - (i > 0 ? q[i - 1] : 0));
            return { q: String.raw`1D, 5 pixels, $g=1$. $\mathbf q=${vec(q)}$. Using the divergence that is exactly $-A^\top$, give $(\operatorname{div}\mathbf q)_0$, $(\operatorname{div}\mathbf q)_2$ and $(\operatorname{div}\mathbf q)_4$.`,
              answer: [dv[0], dv[2], dv[4]], labels: ["pixel 0", "pixel 2", "pixel 4"], tol: 1e-6,
              explain: String.raw`$(\operatorname{div}\mathbf q)_i=q_i-q_{i-1}$, dropping $q_{i-1}$ at the first pixel and $q_i$ at the last. Pixel 0: $q_0=${fmt(q[0])}$. Pixel 2: $${fmt(q[2])}-(${fmt(q[1])})=${fmt(dv[2])}$. Pixel 4: $-q_3=${fmt(dv[4])}$ ($q_4$ is ignored).` };
          } },
        { id: "adjoint", type: "num",
          gen: (r) => {
            const d = Array.from({ length: 4 }, () => r.int(0, 6));
            const q = Array.from({ length: 4 }, () => r.int(-4, 4) / 4);
            const Ad = d.map((v, i) => (i < 3 ? d[i + 1] - v : 0));
            const ans = Ad.reduce((s, v, i) => s + v * q[i], 0);
            return { q: String.raw`1D, 4 pixels. $\mathbf d=${vec(d)}$, $\mathbf q=${vec(q)}$. Compute $\langle A\mathbf d,\mathbf q\rangle$ (you can check it equals $-\langle\mathbf d,\operatorname{div}\mathbf q\rangle$).`,
              answer: ans, tol: 1e-6,
              explain: String.raw`$A\mathbf d=${vec(Ad)}$, so $\langle A\mathbf d,\mathbf q\rangle=${Ad.map((v, i) => `(${fmt(v)})(${fmt(q[i])})`).join("+")}=${fmt(ans)}$.` };
          } },
        { id: "grad2dg", type: "num",
          gen: (r) => {
            const d = r.int(0, 8) / 4, dr = r.int(0, 8) / 4, dd = r.int(0, 8) / 4, g = r.pick([0.2, 0.5, 0.8, 1]);
            const lastCol = r.next() < 0.3;
            const gx = lastCol ? 0 : g * (dr - d), gy = g * (dd - d);
            return { q: String.raw`2D. At pixel $(u,v)$: $d=${fmt(d)}$, right neighbour $${fmt(dr)}$, lower neighbour $${fmt(dd)}$, $g(\mathbf u)=${g}$.${lastCol ? " The pixel is on the <b>last column</b> (the “right neighbour” value is outside the image)." : ""} Give both components of $g(\mathbf u)(A\mathbf d)(u,v)$.`,
              answer: [gx, gy], labels: ["$x$", "$y$"], tol: 1e-6,
              explain: String.raw`$x$: ${lastCol ? "0 (last column)" : `$${g}(${fmt(dr)}-${fmt(d)})=${fmt(gx)}$`}; $y$: $${g}(${fmt(dd)}-${fmt(d)})=${fmt(gy)}$.` };
          } },
        { id: "lf", type: "num",
          gen: (r) => {
            const eps = r.pick([0.2, 0.4, 0.5, 1]);
            const x = r.sign() * r.pick([0.1, 0.2, 0.3, 0.6, 0.8, 1.5, 2]);
            const qs = clamp(x / eps, -1, 1), v = qs * x - (eps / 2) * qs * qs;
            return { q: String.raw`Eq. (8) in 1D: maximise $f(q)=qx-\frac\epsilon2q^2$ over $|q|\le1$ with $x=${x}$, $\epsilon=${eps}$. Give the maximiser $q^\star$ and the maximum (= $\|x\|_\epsilon$).`,
              answer: [qs, v], labels: ["$q^\\star$", "max"], tol: 1e-4,
              explain: String.raw`Unconstrained peak at $x/\epsilon=${fmt(x / eps)}$. ${Math.abs(x) <= eps ? String.raw`Inside $[-1,1]$, so $q^\star=${fmt(qs)}$ and the value is $x^2/2\epsilon=${fmt(v, 4)}$.` : String.raw`Outside $[-1,1]$, so $q^\star=\operatorname{sign}(x)=${fmt(qs)}$ and the value is $|x|-\epsilon/2=${fmt(v, 4)}$.`}` };
          } },
        { id: "whydual", type: "mc",
          q: "Why rewrite the Huber regulariser as a maximum over $\\mathbf q$ (eq. 8)?",
          choices: [
            "Inside the max, $\\mathbf d$ appears only linearly ($\\langle GA\\mathbf d,\\mathbf q\\rangle$), so the $\\mathbf d$-update needs only $A^\\top G\\mathbf q$ and every pixel can be updated in parallel",
            "Because the maximum is always smaller than the Huber norm, giving a lower bound on the energy",
            "Because it makes the data term $C$ convex",
            "Because $\\mathbf q$ stores the final inverse depth at sub-sample accuracy",
          ],
          answer: 0,
          explain: "The max equals the Huber norm exactly (not a bound). It leaves the data term untouched (that is handled by the search for $\\mathbf a$). What it buys is a smooth, per-pixel update: the derivative of $\\langle GA\\mathbf d,\\mathbf q\\rangle$ w.r.t. $\\mathbf d$ is $A^\\top G\\mathbf q=-\\operatorname{div}(g\\mathbf q)$." },
        { id: "game", type: "multi",
          q: "In the saddle-point problem (9)–(10), which statements are true? (select all)",
          choices: [
            "$\\mathbf q$ maximises $E$, subject to $\\|\\mathbf q_{\\mathbf u}\\|\\le1$ at every pixel",
            "$\\mathbf d$ and $\\mathbf a$ minimise $E$",
            "For fixed $\\mathbf d$, the best $\\mathbf q$ makes $\\langle GA\\mathbf d,\\mathbf q\\rangle-\\frac\\epsilon2\\|\\mathbf q\\|^2$ equal to $\\sum g\\|\\nabla d\\|_\\epsilon$",
            "$\\mathbf q$ also has a coupling term $\\frac1{2\\theta}\\|\\mathbf q-\\mathbf a\\|^2$",
          ],
          answer: [0, 1, 2],
          explain: "q is the maximising player on the unit disc, d and a minimise, and q's best reply recovers the Huber term (eq. 8). The coupling is between $\\mathbf d$ and $\\mathbf a$, not $\\mathbf q$." },
        { id: "dual1d", type: "num",
          gen: (r) => {
            const q = r.int(-6, 6) / 10, g = r.pick([0.5, 1]), di = r.int(0, 8) / 4, dn = r.int(0, 8) / 4;
            const sq = 0.5, eps = r.pick([0.2, 0.5, 1]);
            const raw = (q + sq * g * (dn - di)) / (1 + sq * eps), ans = raw / Math.max(1, Math.abs(raw));
            return { q: String.raw`1D dual step at pixel $i$ (not the last). $q_i=${q}$, $g_i=${g}$, $d_i=${fmt(di)}$, $d_{i+1}=${fmt(dn)}$, $\sigma_q=0.5$, $\epsilon=${eps}$ (exaggerated). Give $q_i^{n+1}$ to 3 decimals.`,
              answer: ans, tol: 1.5e-3,
              explain: String.raw`$\frac{${q}+0.5\cdot${g}\cdot(${fmt(dn)}-${fmt(di)})}{1+0.5\cdot${eps}}=\frac{${fmt(q + sq * g * (dn - di), 4)}}{${fmt(1 + sq * eps)}}=${fmt(raw, 4)}$${Math.abs(raw) > 1 ? String.raw`, longer than 1, so projected to $${fmt(ans)}$` : ", already inside $[-1,1]$"}.` };
          } },
        { id: "proj2d", type: "num",
          gen: (r) => {
            const [qx, qy] = r.pick([[0.3, 0.4], [0, 0.6], [0.6, 0], [-0.3, 0.4], [0.3, -0.4]]);
            const [gx, gy] = r.pick([[2, 2], [2, 0], [4, 2], [2, 4], [-2, 2], [0, 4]]);
            const g = r.pick([0.5, 1]);
            const vx = qx + 0.5 * g * gx, vy = qy + 0.5 * g * gy; // ε ignored (1e-4)
            const n = Math.hypot(vx, vy), s = Math.max(1, n);
            return { q: String.raw`2D dual step, $\epsilon\approx0$ (ignore the $1+\sigma_q\epsilon$ divisor). $\mathbf q=(${qx},${qy})$, $(A\mathbf d)(\mathbf u)=(${gx},${gy})$, $g=${g}$, $\sigma_q=0.5$. Give the new $\mathbf q$ after projection, to 3 decimals.`,
              answer: [vx / s, vy / s], labels: ["$q^x$", "$q^y$"], tol: 2e-3,
              explain: String.raw`Before projection: $(${qx}+0.5\cdot${g}\cdot${gx},\ ${qy}+0.5\cdot${g}\cdot${gy})=(${fmt(vx)},${fmt(vy)})$ with length $${fmt(n, 4)}$. ${n > 1 ? `Divide by it: $(${fmt(vx / s, 4)},${fmt(vy / s, 4)})$.` : "Length ≤ 1, so it stays."}` };
          } },
        { id: "primal1d", type: "num",
          gen: (r) => {
            const d = r.int(2, 8) / 4, a = r.int(0, 10) / 4, qi = r.pick([-1, -0.5, 0, 0.5, 1]), ql = r.pick([-1, -0.5, 0, 0.5, 1]);
            const g = r.pick([0.5, 1]), gl = 1, th = r.pick([0.25, 0.5, 1]), sd = 0.25;
            const div = g * qi - gl * ql, ans = (d + sd * (div + a / th)) / (1 + sd / th);
            return { q: String.raw`1D primal step at an interior pixel $i$. $d_i=${fmt(d)}$, $a_i=${fmt(a)}$, $q_i=${qi}$, $g_i=${g}$, $q_{i-1}=${ql}$, $g_{i-1}=1$, $\theta=${th}$, $\sigma_d=0.25$. Give $d_i^{n+1}$ to 3 decimals.`,
              answer: ans, tol: 1.5e-3,
              explain: String.raw`$\operatorname{div}(g\mathbf q)_i=g_iq_i-g_{i-1}q_{i-1}=${fmt(g * qi)}-(${fmt(ql)})=${fmt(div)}$. Then $d=\frac{${fmt(d)}+0.25(${fmt(div)}+${fmt(a)}/${th})}{1+0.25/${th}}=\frac{${fmt(d + sd * (div + a / th), 4)}}{${fmt(1 + sd / th, 4)}}=${fmt(ans, 4)}$.` };
          } },
        { id: "order", type: "mc",
          q: "Within one iteration of this implementation, which values does each pass read?",
          choices: [
            "Dual reads the old $\\mathbf d^n$; primal reads the new $\\mathbf q^{n+1}$ and the old $\\mathbf a^n$; the search reads the new $\\mathbf d^{n+1}$",
            "Dual reads $\\mathbf d^{n+1}$; primal reads $\\mathbf q^n$ and $\\mathbf a^{n+1}$; the search reads $\\mathbf d^n$",
            "All three passes read only values from the previous iteration",
            "The search runs first, then primal, then dual",
          ],
          answer: 0,
          explain: "Order is dual → primal → aux, each GPU pass reading what the previous one just wrote (paper steps 1–2)." },
        { id: "steps", type: "num",
          gen: (r) => {
            const sq = r.pick([0.25, 0.5, 1, 2]), dim = r.pick([1, 2]);
            const L2 = dim === 1 ? 4 : 8, ans = 1 / (sq * L2);
            return { q: String.raw`Convergence needs $\sigma_q\sigma_d\|A\|^2\le1$ with $\|A\|^2\le4$ in 1D and $\le8$ in 2D. For a ${dim}D image with $\sigma_q=${sq}$, what is the largest safe $\sigma_d$?`,
              answer: ans, tol: 1e-6,
              explain: String.raw`$\sigma_d=1/(\sigma_q\|A\|^2)=1/(${sq}\cdot${L2})=${fmt(ans, 4)}$. (This implementation: 2D, $\sigma_q=0.5$, $\sigma_d=0.25$.)` };
          } },
        { id: "eaux", type: "num",
          gen: (r) => {
            const xs = [0.6, 0.8, 1.0, 1.2];
            let th, lam, C, d, E, b;
            for (;;) { // redraw until the minimum is clear (no near-ties)
              th = r.pick([0.05, 0.1, 0.2]); lam = r.pick([0.5, 1]);
              C = xs.map(() => r.int(1, 9) / 10);
              d = r.pick([0.7, 0.8, 0.9, 1.0, 1.1]);
              E = xs.map((x, k) => ((d - x) ** 2) / (2 * th) + lam * C[k]);
              b = 0;
              for (let k = 1; k < 4; k++) if (E[k] < E[b]) b = k;
              if (E.every((e, k) => k === b || e - E[b] > 0.02)) break;
            }
            return { q: String.raw`Point-wise search (14) at one pixel: samples $a\in\{0.6,0.8,1.0,1.2\}$ with costs $C=${vec(C)}$, $d_{\mathbf u}=${d}$, $\theta=${th}$, $\lambda=${lam}$. Which $a$ minimises $E_{\text{aux}}$?`,
              answer: xs[b], tol: 1e-6,
              explain: String.raw`$E_{\text{aux}}=\frac{(d-a)^2}{2\theta}+\lambda C$: ${xs.map((x, k) => `$a=${x}$: ${fmt(E[k], 4)}`).join(", ")}. Smallest at $a=${xs[b]}$.` };
          } },
        { id: "radius", type: "num",
          gen: (r) => {
            const th = r.pick([0.2, 0.05, 0.02, 0.01, 0.005, 0.001]), lam = r.pick([0.5, 1, 2]), dc = r.pick([0.1, 0.2, 0.25, 0.4, 0.5]);
            const ans = Math.sqrt(2 * th * lam * dc);
            return { q: String.raw`Band half-width (17): $\theta=${th}$, $\lambda=${lam}$, $C_{\max}-C_{\min}=${dc}$. Give $r$ to 4 decimals.`,
              answer: ans, rtol: 0.002,
              explain: String.raw`$r=\sqrt{2\theta\lambda(C_{\max}-C_{\min})}=\sqrt{2\cdot${th}\cdot${lam}\cdot${dc}}=\sqrt{${fmt(2 * th * lam * dc, 6)}}=${fmt(ans, 4)}$.` };
          } },
        { id: "layers", type: "num",
          gen: (r) => {
            // ξmin = 0.2, Δξ = 0.05, S = 32; keep (d ± r − ξmin)/Δξ away from integers.
            let d, rr, k0, k1;
            for (;;) {
              d = r.int(10, 36) * 0.05 + r.pick([0.01, 0.02, 0.03, 0.04]);
              rr = r.pick([0.06, 0.08, 0.12, 0.16, 0.22]);
              const f0 = (d - rr - 0.2) / 0.05, f1 = (d + rr - 0.2) / 0.05;
              if (Math.abs(f0 - Math.round(f0)) > 0.05 && Math.abs(f1 - Math.round(f1)) > 0.05) {
                k0 = clamp(Math.floor(f0), 0, 31); k1 = clamp(Math.ceil(f1), 0, 31); break;
              }
            }
            return { q: String.raw`$\xi_{\min}=0.2$, $\Delta\xi=0.05$, $S=32$. The current $d_{\mathbf u}=${fmt(d)}$ and $r=${rr}$. Using $k_0=\lfloor(d-r-\xi_{\min})/\Delta\xi\rfloor$, $k_1=\lceil(d+r-\xi_{\min})/\Delta\xi\rceil$ (clamped to $[0,31]$), how many layers are tested?`,
              answer: k1 - k0 + 1, tol: 1e-6,
              explain: String.raw`$(d-r-0.2)/0.05=${fmt((d - rr - 0.2) / 0.05, 3)}$ → $k_0=${k0}$; $(d+r-0.2)/0.05=${fmt((d + rr - 0.2) / 0.05, 3)}$ → $k_1=${k1}$. Count $=${k1}-${k0}+1=${k1 - k0 + 1}$.` };
          } },
        { id: "bandwhy", type: "mc",
          q: "Why can a sample $a$ with $|a-d_{\\mathbf u}|>r$ never be the minimiser of $E_{\\text{aux}}$?",
          choices: [
            "Its coupling term alone exceeds $\\lambda(C_{\\max}-C_{\\min})$, so its energy is above $\\lambda C_{\\max}$, while $a=d_{\\mathbf u}$ achieves at most $\\lambda C_{\\max}$",
            "Because the cost volume is only valid within $r$ of the arg min",
            "Because the Newton step can move $a$ by at most $r$",
            "Because $r$ is the distance the primal step moves $d$ per iteration",
          ],
          answer: 0,
          explain: "Outside the band: $E_{\\text{aux}}>\\lambda C_{\\min}+\\lambda(C_{\\max}-C_{\\min})=\\lambda C_{\\max}$. Near $d$ the parabola is ~0, so some sample there costs at most about $\\lambda C_{\\max}$. (The code rounds the band outwards to whole layers, so the nearest samples to $d$ are always tested.)" },
        { id: "newton", type: "num",
          gen: (r) => {
            const e0 = r.int(2, 10) / 10, em = e0 + r.int(1, 8) / 10, ep = e0 + r.int(1, 8) / 10;
            const a = r.pick([0.8, 1.0, 1.25, 1.5]), dx = r.pick([0.05, 0.1]);
            const grad = (ep - em) / (2 * dx), hess = (ep - 2 * e0 + em) / (dx * dx);
            const ans = a - clamp(grad / hess, -dx, dx);
            return { q: String.raw`Newton step (18). Best sample $a=${a}$, $\Delta\xi=${dx}$, $E_-=${fmt(em)}$, $E_0=${fmt(e0)}$, $E_+=${fmt(ep)}$. Give $\hat a$ to 4 decimals.`,
              answer: ans, tol: 2e-4,
              explain: String.raw`$E'=\frac{${fmt(ep)}-${fmt(em)}}{2\cdot${dx}}=${fmt(grad, 4)}$, $E''=\frac{${fmt(ep)}-2\cdot${fmt(e0)}+${fmt(em)}}{${dx}^2}=${fmt(hess, 4)}$. $\hat a=${a}-(${fmt(grad, 4)}/${fmt(hess, 4)})=${fmt(ans, 4)}$.` };
          } },
        { id: "newtonwhy", type: "multi",
          q: "About the sub-sample Newton step in this implementation (select all true):",
          choices: [
            "It runs inside every iteration, right after the discrete search",
            "It is skipped when the best sample is the first or last layer, or when $E''\\le0$",
            "It would work just as well once, after the loop ends",
            "When the best sample is lower than both neighbours, the step is at most $\\Delta\\xi/2$",
          ],
          answer: [0, 1, 3],
          explain: "After the loop $\\theta$ is tiny, so $E_{\\text{aux}}$ is dominated by a spike-like parabola at $d$ and the fit returns $\\approx d$: it must be embedded in the iterations (paper §2.2.5). The $\\Delta\\xi/2$ bound follows from $|E_--E_+|\\le(E_--E_0)+(E_+-E_0)$." },
        { id: "theta", type: "num",
          gen: (r) => {
            const beta = r.pick([0.01, 0.02, 0.05, 0.1]), th0 = r.pick([0.2, 0.1]);
            const t1 = th0, t2 = t1 * (1 - beta), t3 = t2 * (1 - 2 * beta), t4 = t3 * (1 - 3 * beta);
            return { q: String.raw`Schedule $\theta_{n+1}=\theta_n(1-\beta n)$, $n$ starting at 0, with an exaggerated $\beta=${beta}$ and $\theta_0=${th0}$. Give $\theta_4$ to 4 significant figures.`,
              answer: t4, rtol: 5e-4,
              explain: String.raw`$\theta_1=\theta_0(1-0)=${th0}$, $\theta_2=${th0}(1-${beta})=${fmt(t2, 5)}$, $\theta_3=\theta_2(1-${fmt(2 * beta)})=${fmt(t3, 5)}$, $\theta_4=\theta_3(1-${fmt(3 * beta)})=${fmt(t4, 5)}$.` };
          } },
        { id: "sched", type: "multi",
          q: "Which statements about this implementation's θ schedule are true? (select all)",
          choices: [
            "$\\theta_0=0.2$ and the loop stops once $\\theta\\le10^{-4}$",
            "$\\beta=2.5\\cdot10^{-4}$ while $\\theta\\ge10^{-3}$, then $2.5\\cdot10^{-5}$: a quarter of the paper's values",
            "It runs roughly 470 iterations, about twice the paper's schedule",
            "$\\theta$ is multiplied by a constant factor each iteration",
          ],
          answer: [0, 1, 2],
          explain: "The factor is $1-\\beta n$, which changes every iteration (decay accelerates). With the quarter-size β the loop takes 474 iterations vs 236 with the paper's β." },
        { id: "lambda", type: "num",
          gen: (r) => {
            const dm = r.pick([0.5, 1, 1.5, 2, 3, 4, 6]);
            return { q: String.raw`The nearest predicted scene depth is $\bar d=${dm}$ m (not the first keyframe). What $\lambda$ does the paper's rule give?`,
              answer: 1 / (1 + 0.5 * dm), tol: 1e-3,
              explain: String.raw`$\lambda=1/(1+0.5\bar d)=1/(1+${fmt(0.5 * dm)})=${fmt(1 / (1 + 0.5 * dm), 4)}$. Farther scenes → weaker data term → more regularisation.` };
          } },
        { id: "trough", type: "num",
          gen: (r) => {
            const n = 10, k = r.int(2, 7), sl = r.pick([0.01, 0.02, 0.03, 0.05, 0.1]);
            const C = Array.from({ length: n }, (_, i) => +(0.1 + Math.min(0.5, Math.abs(i - k) * sl)).toFixed(2));
            C[0] = Math.max(C[0], 0.5);
            const lo = Math.min(...C), hi = Math.max(...C), tau = Math.max(0.01, 0.1 * (hi - lo));
            let first = n, last = 0;
            C.forEach((v, i) => { if (v <= lo + tau + 1e-9) { first = Math.min(first, i); last = Math.max(last, i); } });
            return { q: String.raw`A cost row over 10 layers: $C=${vec(C)}$. With $\tau=\max(0.01,\ 0.1(C_{\max}-C_{\min}))$, how many layers wide is the trough (first to last layer with $C\le C_{\min}+\tau$)?`,
              answer: last - first + 1, tol: 1e-6,
              explain: String.raw`$C_{\min}=${fmt(lo)}$, $C_{\max}=${fmt(hi)}$, $\tau=${fmt(tau, 4)}$, threshold $${fmt(lo + tau, 4)}$. Layers ${first}…${last} → width $${last - first + 1}$ (${last - first + 1 <= 6 ? "≤ 6: localised, start from the arg min" : "> 6: not localised, push–pull fill"}).` };
          } },
        { id: "init", type: "mc",
          q: "How does this implementation initialise $\\mathbf d$ and $\\mathbf a$ before the loop?",
          choices: [
            "Arg min of the cost row where the trough is ≤ 6 layers wide; elsewhere a push–pull interpolation of those pixels; $\\mathbf q=\\mathbf 0$",
            "Arg min of the cost row at every pixel, as in the paper; $\\mathbf q=\\mathbf 0$",
            "The middle of the depth range everywhere; $\\mathbf q$ = the image gradient",
            "The previous keyframe's depth map, reprojected",
          ],
          answer: 0,
          explain: "The paper uses the arg min everywhere; in textureless pixels that is noise, so this implementation fills them from localised neighbours (fill_unconstrained) and lets the regulariser refine. The dual starts at zero." },
        { id: "thetazero", type: "multi",
          q: "What happens as $\\theta\\to0$ during the solve? (select all)",
          choices: [
            "The coupling $\\frac1{2\\theta}(d-a)^2$ forces $\\mathbf d$ and $\\mathbf a$ together, so the final $\\mathbf d$ minimises the original energy (6)",
            "The search band $r=\\sqrt{2\\theta\\lambda(C_{\\max}-C_{\\min})}$ shrinks, so fewer samples are tested",
            "The primal step pulls $\\mathbf d$ more strongly towards $\\mathbf a$ (the $\\sigma_d/\\theta$ terms dominate)",
            "The data term $C$ is re-computed from the images at finer resolution",
          ],
          answer: [0, 1, 2],
          explain: "The cost volume is fixed during the solve. θ only controls the coupling: tighter coupling, a smaller band, and a primal step dominated by $\\mathbf a/\\theta$." },
      ]);
    },
  });
})();
