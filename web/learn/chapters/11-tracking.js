// Chapter 11: Dense tracking (paper §2.3, eqs 19–21): model prediction,
// whole-image Gauss–Newton alignment, coarse-to-fine, robust rejection,
// rotation pre-alignment, gain/bias, failure detection.
(() => {
  "use strict";
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const DEG = Math.PI / 180;

  /** "#rrggbb" / "#rgb" / "rgb(r,g,b)" -> [r, g, b]. */
  function rgbOf(s) {
    s = String(s || "").trim();
    let m = s.match(/^#([0-9a-f]{3})$/i);
    if (m) return m[1].split("").map((h) => parseInt(h + h, 16));
    m = s.match(/^#([0-9a-f]{6})/i);
    if (m) return [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
    m = s.match(/(\d+)\D+(\d+)\D+(\d+)/);
    if (m) return [+m[1], +m[2], +m[3]];
    return [128, 128, 128];
  }

  /** Draws a w×h pixel image (pix(i, out) fills out=[r,g,b]) into a canvas rect, nearest-neighbour. */
  const blitCache = new Map();
  function blit(ctx, w, h, x, y, dw, dh, pix) {
    const key = w + "x" + h;
    let e = blitCache.get(key);
    if (!e) {
      const cv = document.createElement("canvas");
      cv.width = w; cv.height = h;
      const cx = cv.getContext("2d");
      e = { cv, cx, id: cx.createImageData(w, h) };
      blitCache.set(key, e);
    }
    const d = e.id.data, o = [0, 0, 0];
    for (let i = 0; i < w * h; i++) {
      pix(i, o);
      d[4 * i] = o[0]; d[4 * i + 1] = o[1]; d[4 * i + 2] = o[2]; d[4 * i + 3] = 255;
    }
    e.cx.putImageData(e.id, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(e.cv, x, y, dw, dh);
    ctx.restore();
  }
  const g8 = (v) => Math.round(clamp(v, 0, 1) * 255);

  // ------------------------------------------------------------ image alignment engine
  // A 2D stand-in for DTAM's 6DOF alignment: template I_v (64×64) is the
  // centre of a scene; the live image I_l (128×128) is that scene moved by an
  // unknown (t_x, t_y, θ) about the image centre. Same machinery as
  // track6.wgsl + gn.wgsl: bilinear sampling, central-difference gradients,
  // truncated-quadratic cost, LM-safeguarded Gauss–Newton, pixel-centred
  // pyramid, closed-form gain/bias.
  const LW = 128, TW = 64, OFF = 32, CEN = 63.5;

  function pyramid(data, w, h, n) {
    const lv = [{ w, h, data }];
    for (let l = 1; l < n; l++) {
      const s = lv[l - 1], nw = Math.ceil(s.w / 2), nh = Math.ceil(s.h / 2), d = new Float32Array(nw * nh);
      for (let y = 0; y < nh; y++) {
        for (let x = 0; x < nw; x++) {
          let sum = 0;
          for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) sum += s.data[Math.min(2 * y + j, s.h - 1) * s.w + Math.min(2 * x + i, s.w - 1)];
          d[y * nw + x] = sum / 4;
        }
      }
      lv.push({ w: nw, h: nh, data: d });
    }
    return lv;
  }
  function sample(im, x, y) {
    const x0 = Math.floor(x), y0 = Math.floor(y), tx = x - x0, ty = y - y0;
    const X0 = clamp(x0, 0, im.w - 1), X1 = clamp(x0 + 1, 0, im.w - 1), Y0 = clamp(y0, 0, im.h - 1), Y1 = clamp(y0 + 1, 0, im.h - 1);
    const d = im.data, w = im.w;
    const a = d[Y0 * w + X0] + (d[Y0 * w + X1] - d[Y0 * w + X0]) * tx;
    const b = d[Y1 * w + X0] + (d[Y1 * w + X1] - d[Y1 * w + X0]) * tx;
    return a + (b - a) * ty;
  }
  /** Template pixel (level-0 coords) -> live coords under params p = [tx, ty, θ]. */
  function warp0(px, py, p) {
    const sx = px + OFF - CEN, sy = py + OFF - CEN, c = Math.cos(p[2]), s = Math.sin(p[2]);
    return [c * sx - s * sy + CEN + p[0], s * sx + c * sy + CEN + p[1]];
  }
  /** One residual pass at pyramid level `lvl`: normal equations + counters. */
  function evaluate(A, lvl, p, tau, gain, bias, mask) {
    const Lv = A.live[lvl], T = A.tmpl[lvl], S = 2 ** lvl;
    const H = new Float64Array(9), g = new Float64Array(3);
    let r2 = 0, nRej = 0, nView = 0, nUsed = 0, sl = 0, sv = 0, sll = 0, slv = 0;
    const c = Math.cos(p[2]), s = Math.sin(p[2]);
    for (let j = 0; j < T.h; j++) {
      for (let i = 0; i < T.w; i++) {
        const sx = (i + 0.5) * S - 0.5 + OFF - CEN, sy = (j + 0.5) * S - 0.5 + OFF - CEN;
        const wx = c * sx - s * sy + CEN + p[0], wy = s * sx + c * sy + CEN + p[1];
        const qx = (wx + 0.5) / S - 0.5, qy = (wy + 0.5) / S - 0.5;
        const k = j * T.w + i;
        if (qx < 1 || qy < 1 || qx > Lv.w - 2 || qy > Lv.h - 2) { if (mask) mask[k] = 3; continue; }
        nView++;
        const il = sample(Lv, qx, qy), iv = T.data[k];
        const r = gain * il + bias - iv;
        if (Math.abs(r) > tau) { nRej++; if (mask) mask[k] = 2; continue; }
        if (mask) mask[k] = 1;
        const gx = 0.5 * (sample(Lv, qx + 1, qy) - sample(Lv, qx - 1, qy));
        const gy = 0.5 * (sample(Lv, qx, qy + 1) - sample(Lv, qx, qy - 1));
        const J0 = gain * gx / S, J1 = gain * gy / S, J2 = gain * (gx * (-s * sx - c * sy) + gy * (c * sx - s * sy)) / S;
        H[0] += J0 * J0; H[1] += J0 * J1; H[2] += J0 * J2; H[4] += J1 * J1; H[5] += J1 * J2; H[8] += J2 * J2;
        g[0] += J0 * r; g[1] += J1 * r; g[2] += J2 * r;
        r2 += r * r; nUsed++; sl += il; sv += iv; sll += il * il; slv += il * iv;
      }
    }
    H[3] = H[1]; H[6] = H[2]; H[7] = H[5];
    return { H, g, r2, nRej, nView, nUsed, sl, sv, sll, slv };
  }

  function render(root, L) {
    const la = L.la;

    function makeScene(seed) {
      const r = L.rng(seed);
      const blobs = [];
      for (let k = 0; k < 80; k++) blobs.push([-24 + r.next() * 176, -24 + r.next() * 176, 4 + r.next() * 6, (r.next() - 0.5) * 0.8]);
      const w = (2 * Math.PI) / 6;
      return (x, y) => {
        let v = 0.5 + 0.07 * Math.sin(w * x) * Math.sin(w * y);
        for (const [bx, by, s, a] of blobs) {
          const d2 = (x - bx) ** 2 + (y - by) ** 2;
          if (d2 < 16 * s * s) v += a * Math.exp(-d2 / (2 * s * s));
        }
        return clamp(v, 0, 1);
      };
    }
    const scene = makeScene(7);
    /** Live + template pyramids for a true motion, optional occluder [x, y, w, h] and exposure gain. */
    function build(truth, occ, expo) {
      const [tx, ty, th] = truth, c = Math.cos(th), s = Math.sin(th);
      const live = new Float32Array(LW * LW), tmpl = new Float32Array(TW * TW);
      for (let y = 0; y < LW; y++) {
        for (let x = 0; x < LW; x++) {
          const dx = x - CEN - tx, dy = y - CEN - ty;
          let v = clamp(expo * scene(c * dx + s * dy + CEN, -s * dx + c * dy + CEN), 0, 1);
          if (occ && x >= occ[0] && x < occ[0] + occ[2] && y >= occ[1] && y < occ[1] + occ[3]) v = 0.08 + 0.04 * Math.sin(x * 0.5);
          live[y * LW + x] = v;
        }
      }
      for (let y = 0; y < TW; y++) for (let x = 0; x < TW; x++) tmpl[y * TW + x] = scene(x + OFF, y + OFF);
      return { live: pyramid(live, LW, LW, 3), tmpl: pyramid(tmpl, TW, TW, 3) };
    }
    /**
     * LM-safeguarded Gauss–Newton over a schedule [{lvl, iters, tau}], mirroring
     * gn.wgsl: accept if the robust cost did not rise (damping /3, gain/bias
     * refit), else revert and damping ×10; step from the accepted normal equations.
     */
    function makeTracker(A, schedule, { gainBias = false, free = [1, 1, 1], start = [0, 0, 0] } = {}) {
      const st = { pose: start.slice(), cand: start.slice(), gain: 1, bias: 0, li: 0, it: 0, hist: [], finished: false, best: NaN };
      const begin = () => { st.cand = st.pose.slice(); st.hasAcc = false; st.damping = 1e-4; st.done = false; st.it = 0; };
      begin();
      st.step = () => {
        if (st.finished) return;
        const lev = schedule[st.li];
        if (!st.done) {
          const S = evaluate(A, lev.lvl, st.cand, lev.tau, st.gain, st.bias);
          if (S.nView < 20 || S.nUsed < 20) st.done = true;
          else {
            const cost = (S.r2 + S.nRej * lev.tau * lev.tau) / S.nView;
            let ok = true;
            if (st.hasAcc && cost > st.best) {
              st.damping *= 10;
              if (st.damping > 1e4) { st.done = true; ok = false; }
            } else {
              st.damping = Math.max(st.damping / 3, 1e-7);
              if (gainBias) {
                const n = S.nUsed, det = n * S.sll - S.sl * S.sl;
                if (n > 100 && Math.abs(det) > 1e-9) {
                  const a = clamp((n * S.slv - S.sl * S.sv) / det, 0.5, 2);
                  st.bias = (S.sv - a * S.sl) / n;
                  st.gain = a;
                }
              }
              st.hasAcc = true; st.best = cost; st.pose = st.cand.slice(); st.acc = S;
            }
            if (ok) {
              const H = [0, 1, 2].map((a) => [0, 1, 2].map((b) => {
                if (!free[a] || !free[b]) return a === b ? 1 : 0;
                return st.acc.H[a * 3 + b] * (a === b ? 1 + st.damping : 1);
              }));
              const d = la.solve(H, [0, 1, 2].map((a) => (free[a] ? -st.acc.g[a] : 0)));
              if (!d) st.done = true;
              else {
                st.cand = st.pose.map((v, a) => v + d[a]);
                if (Math.hypot(...d) < 1e-6) st.done = true;
              }
            }
            st.hist.push({ cost: st.best, lvl: lev.lvl, pose: st.pose.slice() });
          }
        }
        st.it++;
        if (st.done || st.it >= lev.iters) {
          st.li++;
          if (st.li >= schedule.length) st.finished = true;
          else begin();
        }
      };
      return st;
    }

    // ================================================================ intro
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>Mapping (chapters 8–10) needs a pose for every frame. <b>Dense tracking</b> supplies it: render the model from a guessed pose, then nudge the pose until the rendering matches the live image <i>at every pixel</i>. No corners, no matching.</p>
      <div class="key">Tracking is image alignment. The unknown is a 6-number pose update $\psi$; the data is every pixel with a model prediction; the method is Gauss–Newton (chapter 5).</div>
      <p>Per live frame $I_l$, this implementation runs:</p>
      <ol>
        <li><b>Rotation pre-alignment</b> against the previous frame (coarse pyramid levels only).</li>
        <li><b>Initial guess</b> $\hat T_{wl}$: that rotation plus constant-velocity translation.</li>
        <li><b>Model prediction</b>: render all keyframes into a virtual camera $v$ at $T_{wv}=\hat T_{wl}$, giving $I_v$ and $\xi_v$.</li>
        <li><b>6DOF alignment</b> of $I_l$ to $I_v$, coarse to fine, ignoring outlier pixels.</li>
      </ol>
      <p>We go in the order that teaches best: prediction, the cost, its Jacobian, the solver, then the extras.</p>

      <h3>Step 1: predict what the camera should see</h3>
      <p>A keyframe $r$ holds an image $I_r$ and an inverse depth map $\xi_r$ (chapters 8–10). Every pixel back-projects to a 3D point, so the map <i>is</i> a surface:</p>
      <ul>
        <li><b>Vertices:</b> one per keyframe pixel, $\mathbf x_r = \pi^{-1}(\mathbf u, \xi_r(\mathbf u))$, coloured $I_r(\mathbf u)$.</li>
        <li><b>Triangles:</b> two per 2×2 block of neighbouring pixels (indices $i, i{+}1, i{+}w$ and $i{+}1, i{+}w{+}1, i{+}w$).</li>
        <li><b>Render</b> into the virtual camera with $T_{vr}=T_{wv}^{-1}T_{wr}$ and $K$, keeping the nearest surface per pixel (a z-buffer). All keyframes are drawn into the same image.</li>
        <li><b>Output:</b> predicted image $I_v(\mathbf u)$ and predicted inverse depth $\xi_v(\mathbf u)$; $\xi_v=0$ where no surface landed.</li>
      </ul>
      <p>The GPU does this with its ordinary triangle rasteriser (<code>predict.wgsl</code>). One catch: at a depth jump, the triangles joining the near and far surface are a fake "rubber sheet". They are almost parallel to the keyframe's viewing ray, so they are easy to detect and drop.</p>
      <div class="eq-card"><div class="eq-label">Oblique-triangle culling (paper §2.2.6, predict.wgsl)</div>
      $$\text{keep triangle} \iff \frac{|\mathbf n\cdot \mathbf x_r|}{\|\mathbf n\|\,\|\mathbf x_r\|} \ge \cos 85^\circ \approx 0.087$$
      <div class="parts">
        <span>$\mathbf n$</span><span>triangle normal in the keyframe's frame, $(\mathbf x_1-\mathbf x_0)\times(\mathbf x_2-\mathbf x_0)$</span>
        <span>$\mathbf x_r$</span><span>a point on the triangle = the direction of the keyframe ray that saw it</span>
        <span>ratio</span><span>$|\cos|$ of the angle between normal and ray: 1 = facing the keyframe, 0 = seen exactly edge-on</span>
        <span>$85^\circ$</span><span><code>max_oblique_deg</code>; steeper triangles are "bridges" across a depth discontinuity</span>
      </div></div>
    `);

    // ================================================================ widget 1: model prediction
    {
      const KW = 40, KH = 30, KF = 36, KCX = 19.5, KCY = 14.5;
      const VW = 80, VH = 60, VF = 72, VCX = 39.5, VCY = 29.5;
      const N = KW * KH;
      const P = new Float32Array(3 * N), Lr = new Float32Array(N), Xr = new Float32Array(N);
      const tex = (X, Y, Z, kind) => {
        if (kind === 2) return 0.8 - 0.35 * ((Math.floor(X / 0.14) + Math.floor(Y / 0.14)) & 1);
        if (kind === 1) return 0.3 + 0.14 * Math.sin(9 * X) + 0.1 * Math.sin(5 * Z);
        return 0.55 + 0.2 * Math.sin(5 * X) * Math.sin(5 * Y) + 0.12 * Math.sin(13 * X + 4 * Y);
      };
      for (let v = 0; v < KH; v++) {
        for (let u = 0; u < KW; u++) {
          const rx = (u - KCX) / KF, ry = (v - KCY) / KF;
          let z = 3, kind = 0;
          if (ry > 0 && 0.75 / ry < z) { z = 0.75 / ry; kind = 1; }
          const bx = 1.6 * rx, by = 1.6 * ry;
          if (bx >= -0.25 && bx <= 0.3 && by >= -0.45 && by <= 0.75 && 1.6 < z) { z = 1.6; kind = 2; }
          const i = v * KW + u;
          P[3 * i] = rx * z; P[3 * i + 1] = ry * z; P[3 * i + 2] = z;
          Xr[i] = 1 / z;
          Lr[i] = clamp(tex(rx * z, ry * z, z, kind), 0, 1);
        }
      }
      const tris = [], triCos = [];
      for (let y = 0; y < KH - 1; y++) {
        for (let x = 0; x < KW - 1; x++) {
          const i = y * KW + x;
          for (const t of [[i, i + 1, i + KW], [i + 1, i + KW + 1, i + KW]]) {
            const p = t.map((k) => [P[3 * k], P[3 * k + 1], P[3 * k + 2]]);
            const n = la.cross(la.sub(p[1], p[0]), la.sub(p[2], p[0]));
            const cen = la.scale(la.add(la.add(p[0], p[1]), p[2]), 1 / 3);
            tris.push(t);
            triCos.push(Math.abs(la.dot(n, cen)) / (la.norm(n) * la.norm(cen) || 1));
          }
        }
      }
      const S = { tx: 0.35, tz: 0, yaw: -4, cull: true, maxDeg: 85 };
      let out = null;
      const predict = () => {
        const Iv = new Float32Array(VW * VH), Xv = new Float32Array(VW * VH);
        const cy = Math.cos(S.yaw * DEG), sy = Math.sin(S.yaw * DEG), minCos = Math.cos(S.maxDeg * DEG);
        const sx = new Float32Array(N), sv = new Float32Array(N), q = new Float32Array(N), ok = new Uint8Array(N);
        for (let i = 0; i < N; i++) {
          const d0 = P[3 * i] - S.tx, d1 = P[3 * i + 1], d2 = P[3 * i + 2] - S.tz;
          const xv = cy * d0 - sy * d2, yv = d1, zv = sy * d0 + cy * d2;
          ok[i] = zv > 0.05 ? 1 : 0;
          sx[i] = (VF * xv) / zv + VCX; sv[i] = (VF * yv) / zv + VCY; q[i] = 1 / zv;
        }
        let culled = 0;
        for (let t = 0; t < tris.length; t++) {
          if (S.cull && triCos[t] < minCos) { culled++; continue; }
          const [a, b, c] = tris[t];
          if (!ok[a] || !ok[b] || !ok[c]) continue;
          const x0 = sx[a], y0 = sv[a], x1 = sx[b], y1 = sv[b], x2 = sx[c], y2 = sv[c];
          const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
          if (Math.abs(area) < 1e-9) continue;
          const minx = Math.max(0, Math.ceil(Math.min(x0, x1, x2))), maxx = Math.min(VW - 1, Math.floor(Math.max(x0, x1, x2)));
          const miny = Math.max(0, Math.ceil(Math.min(y0, y1, y2))), maxy = Math.min(VH - 1, Math.floor(Math.max(y0, y1, y2)));
          for (let py = miny; py <= maxy; py++) {
            for (let px = minx; px <= maxx; px++) {
              const w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area;
              const w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area;
              const w2 = 1 - w0 - w1;
              if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;
              const qq = w0 * q[a] + w1 * q[b] + w2 * q[c];
              const k = py * VW + px;
              if (qq > Xv[k]) {
                Xv[k] = qq;
                Iv[k] = (w0 * Lr[a] * q[a] + w1 * Lr[b] * q[b] + w2 * Lr[c] * q[c]) / qq;
              }
            }
          }
        }
        let cov = 0;
        for (let k = 0; k < VW * VH; k++) if (Xv[k] > 0) cov++;
        out = { Iv, Xv, culled, cov: cov / (VW * VH) };
      };
      const fig = L.figure(root, "<b>Model prediction.</b> Move the virtual camera sideways: the box uncovers wall the keyframe never saw. With culling on, that region is honestly empty (blue: no model). Turn culling off to see the rubber-sheet triangles smear the box across it. Lower the angle limit to see the floor, seen at a grazing angle, start disappearing too.");
      const c = L.canvas(fig.el, { aspect: 0.84, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const upd = () => { predict(); c.redraw(); info(); };
      L.slider(ctl, { label: "sideways $t_x$", min: -0.6, max: 0.6, step: 0.01, value: S.tx, oninput: (v) => { S.tx = v; upd(); } });
      L.slider(ctl, { label: "forward $t_z$", min: -0.4, max: 0.8, step: 0.01, value: S.tz, oninput: (v) => { S.tz = v; upd(); } });
      L.slider(ctl, { label: "turn (yaw)", min: -15, max: 15, step: 0.5, value: S.yaw, fmt: (v) => v.toFixed(1) + "°", oninput: (v) => { S.yaw = v; upd(); } });
      L.slider(ctl, { label: "max oblique", min: 60, max: 89.5, step: 0.5, value: S.maxDeg, fmt: (v) => v.toFixed(1) + "°", oninput: (v) => { S.maxDeg = v; upd(); } });
      L.toggle(ctl, "cull oblique triangles", S.cull, (v) => { S.cull = v; upd(); });
      const ro = L.readout(fig.el); fig.add(ro.el);
      const info = () => {
        if (!out) return;
        ro.html = `coverage (pixels with surface) <b>${(out.cov * 100).toFixed(1)}%</b> · triangles culled <b>${out.culled}</b> of ${tris.length}`;
      };
      let lut = null;
      c.draw = (ctx) => {
        const t = L.theme();
        if (!out) { predict(); info(); }
        if (!lut) lut = Array.from({ length: 256 }, (_, i) => rgbOf(L.viridisish(i / 255)));
        const hole = rgbOf(t.accent);
        const gap = 10, lab = 17;
        const pw = Math.min((c.w - gap) / 2, (c.h - 2 * lab - gap) / 1.5), ph = pw * 0.75;
        const x0 = (c.w - 2 * pw - gap) / 2;
        const xmin = 1 / 3, xmax = 1 / 1.45;
        const xiCol = (xi, o) => { const l = lut[Math.round(clamp((xi - xmin) / (xmax - xmin), 0, 1) * 255)]; o[0] = l[0]; o[1] = l[1]; o[2] = l[2]; };
        const panels = [
          ["keyframe image I_r", x0, lab, KW, KH, (i, o) => { o[0] = o[1] = o[2] = g8(Lr[i]); }],
          ["keyframe inverse depth ξ_r", x0 + pw + gap, lab, KW, KH, (i, o) => xiCol(Xr[i], o)],
          ["predicted I_v", x0, 2 * lab + ph + gap, VW, VH, (i, o) => {
            if (out.Xv[i] > 0) { o[0] = o[1] = o[2] = g8(out.Iv[i]); } else { o[0] = hole[0]; o[1] = hole[1]; o[2] = hole[2]; }
          }],
          ["predicted ξ_v", x0 + pw + gap, 2 * lab + ph + gap, VW, VH, (i, o) => {
            if (out.Xv[i] > 0) xiCol(out.Xv[i], o); else { o[0] = hole[0]; o[1] = hole[1]; o[2] = hole[2]; }
          }],
        ];
        for (const [name, x, y, w, h, pix] of panels) {
          blit(ctx, w, h, x, y, pw, ph, pix);
          L.draw.text(ctx, name, x, y - 5, t.muted, { size: 12 });
        }
      };
    }

    // ================================================================ the cost
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>This keeps the model <b>fully predictive</b>: occlusions and back faces come out right automatically. Two numbers from the prediction matter later: the fraction of pixels with surface (<b>coverage</b>, chapter 12 uses it to add keyframes) and the percentiles of $\xi_v$ (used to pick a new keyframe's depth range).</p>

      <h3>Step 2: the tracking cost (eqs 19–21)</h3>
      <p>The virtual camera sits at our guess, so the live camera is a small motion $T_{lv}$ away from it. Take a pixel $\mathbf u$ of the prediction, lift it to 3D with its predicted inverse depth, move it into the live camera, project, and compare brightness:</p>
      <div class="eq-card"><div class="eq-label">Paper eq. (20) · photometric residual of pixel u</div>
      $$f_\mathbf u(\psi) = I_l\Big(\pi\big(K\,T_{lv}(\psi)\,\pi^{-1}(\mathbf u, \xi_v(\mathbf u))\big)\Big) - I_v(\mathbf u)$$
      <div class="parts">
        <span>$\pi^{-1}(\mathbf u, \xi_v(\mathbf u))$</span><span>3D point $\mathbf x_v$ in the virtual camera's frame (chapter 3)</span>
        <span>$T_{lv}(\psi)$</span><span>current estimate of virtual→live motion, updated by $\psi$ (eq. 21)</span>
        <span>$\pi(K\,\cdot)$</span><span>project into the live image: pixel $\mathbf u_l$ (not an integer: sample bilinearly)</span>
        <span>$I_l(\mathbf u_l) - I_v(\mathbf u)$</span><span>live brightness minus predicted brightness; 0 if the pose is right</span>
      </div></div>
      <div class="eq-card"><div class="eq-label">Paper eq. (19) · total cost</div>
      $$F(\psi) = \frac12\sum_{\mathbf u\in\Omega} f_\mathbf u(\psi)^2 = \frac12\|\mathbf f(\psi)\|_2^2$$
      <div class="parts">
        <span>$\Omega$</span><span>the pixels: in practice those with a prediction ($\xi_v>0$) that land inside $I_l$</span>
        <span>$\mathbf f(\psi)$</span><span>all residuals stacked in one long vector</span>
        <span>$\frac12\|\cdot\|^2$</span><span>least squares; the ½ just cancels the 2 when differentiating</span>
      </div></div>
      <div class="eq-card"><div class="eq-label">Paper eq. (21) · the parametrisation</div>
      $$T_{lv}(\psi) = \hat T_{lv}\,\exp\Big(\sum_{i=1}^6 \psi_i\,\mathrm{gen}^i_{SE(3)}\Big),\qquad \psi = (v_1,v_2,v_3,\ \omega_1,\omega_2,\omega_3)$$
      <div class="parts">
        <span>$\hat T_{lv}$</span><span>the current estimate (starts at identity: the virtual camera <i>is</i> the guess)</span>
        <span>$\exp(\sum\psi_i\,\mathrm{gen}^i)$</span><span>a small rigid motion $T(\psi)$ built from the 6 generators (chapter 4)</span>
        <span>$\mathbf v$, $\boldsymbol\omega$</span><span>translation part (first, in this implementation) and rotation part</span>
      </div></div>
      <p>The paper writes eq. (21) as just $\exp(\sum\psi_i\mathrm{gen}^i)$ and then applies it with $\hat T_{lv}\leftarrow\hat T_{lv}\,T(\hat\psi)$: the update is multiplied on the <b>right</b>, in the virtual camera's frame. That is the <b>forward-compositional</b> scheme. At every iteration we solve for a fresh small $\psi$ around $\psi=0$. Six free numbers, always a valid rigid motion, and the derivative is only ever needed at $\psi=0$, where it is simplest.</p>
      <p>When it has converged, the live pose is $T_{wl}=T_{wv}\,T_{vl}=T_{wv}\,\hat T_{lv}^{-1}$.</p>
      <div class="note"><b>Worked example.</b> $f_x=f_y=100$, $c=(50,40)$, pixel $\mathbf u=(70,30)$ with $\xi_v=0.5$ (depth 2).
      <ul>
        <li>$\mathbf x_v = \frac1{0.5}\big(\frac{70-50}{100}, \frac{30-40}{100}, 1\big) = (0.4, -0.2, 2)$.</li>
        <li>Let $\hat T_{lv}$ be a pure translation by $(0.1, 0, 0)$: $\mathbf x_l = (0.5, -0.2, 2)$.</li>
        <li>$\mathbf u_l = \big(100\cdot\frac{0.5}{2}+50,\ 100\cdot\frac{-0.2}{2}+40\big) = (75, 30)$.</li>
        <li>If $I_l(75,30)=0.62$ and $I_v(70,30)=0.55$: $f_\mathbf u = 0.07$.</li>
      </ul></div>

      <h3>Step 3: Gauss–Newton on ψ</h3>
      <p>Recall chapter 5: linearise every residual around $\psi=0$, $f_\mathbf u(\psi)\approx f_\mathbf u(0)+J_\mathbf u\,\psi$, where $J_\mathbf u$ is a 1×6 row. Minimising the linearised cost gives the normal equations:</p>
      <div class="eq-card"><div class="eq-label">Normal equations, summed over pixels</div>
      $$\underbrace{\Big(\sum_\mathbf u J_\mathbf u^\top J_\mathbf u\Big)}_{H\ (6\times6)}\ \psi = -\underbrace{\sum_\mathbf u J_\mathbf u^\top f_\mathbf u(0)}_{\mathbf g\ (6)}$$
      <div class="parts">
        <span>$J_\mathbf u^\top J_\mathbf u$</span><span>each pixel adds a 6×6 outer product; symmetric, so 21 distinct numbers</span>
        <span>$J_\mathbf u^\top f_\mathbf u$</span><span>each pixel adds 6 numbers</span>
        <span>solve</span><span>one 6×6 system per iteration, however many pixels (Cholesky on the GPU)</span>
      </div></div>
      <p>This is the paper's "solve $\nabla\mathbf f(0)\,\hat\psi=-\mathbf f(0)$ or its normal equations". Then $\hat T_{lv}\leftarrow\hat T_{lv}T(\hat\psi)$, and repeat until $\hat\psi\approx 0$.</p>

      <h3>Step 4: the Jacobian, piece by piece</h3>
      <p>$f_\mathbf u$ is a chain of three simple maps: $\psi\mapsto$ 3D point $\mathbf x_l$ $\mapsto$ pixel $\mathbf u_l$ $\mapsto$ brightness. The chain rule multiplies their derivatives:</p>
      <div class="eq-card"><div class="eq-label">Chain rule for one pixel (at ψ = 0)</div>
      $$J_\mathbf u = \frac{\partial f_\mathbf u}{\partial\psi} = \underbrace{\nabla I_l(\mathbf u_l)}_{1\times2}\ \underbrace{\frac{\partial\,\pi(K\mathbf x)}{\partial\mathbf x}\Big|_{\mathbf x_l}}_{2\times3}\ \underbrace{R_{lv}\,\big[\,I_3\ \big|\ -[\mathbf x_v]_\times\big]}_{3\times6}$$
      <div class="parts">
        <span>$\nabla I_l$</span><span>image gradient at the warped position (central differences, chapter 1): brightness change per pixel moved</span>
        <span>middle</span><span>how the pixel moves when the 3D point moves</span>
        <span>right</span><span>how the 3D point moves when $\psi$ changes</span>
        <span>$I_v(\mathbf u)$</span><span>does not depend on $\psi$: its derivative is 0</span>
      </div></div>
      <div class="eq-card"><div class="eq-label">Projection derivative (2×3)</div>
      $$u = f_x\frac{x}{z}+c_x,\ \ v = f_y\frac{y}{z}+c_y\ \ \Rightarrow\ \ \frac{\partial(u,v)}{\partial(x,y,z)} = \begin{pmatrix} f_x/z & 0 & -f_x x/z^2\\ 0 & f_y/z & -f_y y/z^2\end{pmatrix}$$
      <div class="parts">
        <span>$f_x/z$</span><span>moving the point sideways by 1 moves the pixel by $f_x/z$: far points move less</span>
        <span>$-f_x x/z^2$</span><span>moving the point away pulls its pixel toward the image centre</span>
      </div></div>
      <div class="eq-card"><div class="eq-label">Point derivative (3×6)</div>
      $$T(\psi)\,\mathbf x_v \approx \mathbf x_v + \mathbf v + \boldsymbol\omega\times\mathbf x_v = \mathbf x_v + \mathbf v - \mathbf x_v\times\boldsymbol\omega = \mathbf x_v + \big[\,I_3\ \big|\ -[\mathbf x_v]_\times\big]\,\psi$$
      <div class="parts">
        <span>$\mathbf v$</span><span>a small translation adds directly: the $I_3$ block</span>
        <span>$\boldsymbol\omega\times\mathbf x_v$</span><span>a small rotation, $\exp([\boldsymbol\omega]_\times)\approx I+[\boldsymbol\omega]_\times$ (chapter 4)</span>
        <span>$-[\mathbf x_v]_\times$</span><span>swap the cross product's order to put $\boldsymbol\omega$ on the right</span>
        <span>$R_{lv}$</span><span>then $\hat T_{lv}$ is applied, which rotates any small displacement by $R_{lv}$ (translation does not affect a displacement)</span>
      </div></div>
      <div class="note"><b>How the shader does it</b> (<code>track6.wgsl</code>): first the 1×3 row $\mathbf a = \nabla I_l\cdot\frac{\partial\pi}{\partial\mathbf x}$, then $\mathbf b = \mathbf a R_{lv}$. Because $\mathbf b\cdot(\boldsymbol\omega\times\mathbf x_v)=\boldsymbol\omega\cdot(\mathbf x_v\times\mathbf b)$,
      $$J_\mathbf u = \big(\ \mathbf b\ ,\ \ \mathbf x_v\times\mathbf b\ \big).$$</div>
      <div class="note"><b>Worked example</b> (same pixel as above, $R_{lv}=I$). Gradient $\nabla I_l(75,30) = (0.04, -0.02)$, $\mathbf x_l=(0.5,-0.2,2)$, $\mathbf x_v=(0.4,-0.2,2)$.
      <ul>
        <li>$\frac{\partial\pi}{\partial\mathbf x} = \begin{pmatrix}50&0&-12.5\\0&50&5\end{pmatrix}$ (e.g. $-100\cdot0.5/2^2=-12.5$).</li>
        <li>$\mathbf a = (0.04\cdot50,\ -0.02\cdot50,\ 0.04\cdot(-12.5)+(-0.02)\cdot5) = (2, -1, -0.6) = \mathbf b$.</li>
        <li>$\mathbf x_v\times\mathbf b = \big((-0.2)(-0.6)-2(-1),\ 2\cdot2-0.4(-0.6),\ 0.4(-1)-(-0.2)2\big) = (2.12, 4.24, 0)$.</li>
        <li>$J_\mathbf u = (2, -1, -0.6, 2.12, 4.24, 0)$. E.g. turning by $\omega_2=0.01$ rad changes this residual by about $0.0424$.</li>
      </ul></div>
    `);

    // ================================================================ widget 2: Jacobian explorer
    {
      const IW = 160, IH = 120, F = 100, CX = 80, CY = 60, PS = 0.1;
      const names = ["v₁ (x)", "v₂ (y)", "v₃ (z)", "ω₁", "ω₂", "ω₃"];
      const S = { gen: 0, z: 2, phi: 30, sel: { u: 120, v: 40 } };
      const pointAt = (u, v, z) => [((u - CX) / F) * z, ((v - CY) / F) * z, z];
      const Pi = (x) => [[F / x[2], 0, (-F * x[0]) / (x[2] * x[2])], [0, F / x[2], (-F * x[1]) / (x[2] * x[2])]];
      const pointJac = (x) => {
        const K = la.skew(x);
        return [0, 1, 2].map((i) => [...[0, 1, 2].map((j) => +(i === j)), ...K[i].map((v) => -v)]);
      };
      const flow = (u, v, i) => {
        const x = pointAt(u, v, S.z), P3 = pointJac(x), Pm = Pi(x);
        const dx = [P3[0][i] * PS, P3[1][i] * PS, P3[2][i] * PS];
        return la.matVec(Pm, dx);
      };
      const fig = L.figure(root, "<b>What each ψ component does to the image.</b> Pick a component: arrows show how pixels move for $\\psi_i=0.1$ with a flat wall at depth $z$. Change $z$: translations shrink with depth, rotations do not. Drag the orange pixel and turn its gradient to see its row $J_\\mathbf u$.");
      const c = L.canvas(fig.el, { aspect: 0.75 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const btns = names.map((n, i) => L.button(ctl, "ψ" + "₁₂₃₄₅₆"[i] + " = " + n, () => { S.gen = i; sync(); }));
      const ctl2 = L.controls(fig.el); fig.add(ctl2);
      L.slider(ctl2, { label: "wall depth $z$", min: 0.5, max: 4, step: 0.1, value: S.z, oninput: (v) => { S.z = v; sync(); } });
      L.slider(ctl2, { label: "gradient direction", min: 0, max: 360, step: 5, value: S.phi, fmt: (v) => v + "°", oninput: (v) => { S.phi = v; sync(); } });
      const ro = L.readout(fig.el); fig.add(ro.el);
      const sync = () => {
        btns.forEach((b, i) => { b.className = i === S.gen ? "btn primary" : "btn"; });
        const { u, v } = S.sel;
        const x = pointAt(u, v, S.z), Pm = Pi(x), P3 = pointJac(x);
        const g = [0.05 * Math.cos(S.phi * DEG), 0.05 * Math.sin(S.phi * DEG)];
        const a = [0, 1, 2].map((j) => g[0] * Pm[0][j] + g[1] * Pm[1][j]);
        const J = [0, 1, 2, 3, 4, 5].map((k) => a[0] * P3[0][k] + a[1] * P3[1][k] + a[2] * P3[2][k]);
        const f2 = (w) => L.fmt(w, 2), f3 = (w) => L.fmt(w, 3);
        const mat = (M, f) => "\\begin{pmatrix}" + M.map((r) => r.map(f).join("&")).join("\\\\") + "\\end{pmatrix}";
        const Jt = J.map((w, k) => (k === S.gen ? "\\mathbf{" + f3(w) + "}" : f3(w)));
        ro.html = `pixel (${Math.round(u)}, ${Math.round(v)}), x = (${x.map(f2).join(", ")})<br>` +
          L.tex(`J_\\mathbf u=${mat([g], f3)}${mat(Pm, f2)}${mat(P3, f2)}`) + "<br>" +
          L.tex(`J_\\mathbf u = (${Jt.join(",\\ ")})`);
        c.redraw();
      };
      L.drag(c, () => {
        const sc = c.w / IW;
        return [{ x: (S.sel.u + 0.5) * sc, y: (S.sel.v + 0.5) * sc }];
      }, (_, p) => {
        const sc = c.w / IW;
        S.sel.u = clamp(p.x / sc - 0.5, 2, IW - 3);
        S.sel.v = clamp(p.y / sc - 0.5, 2, IH - 3);
        sync();
      });
      c.draw = (ctx) => {
        const t = L.theme();
        const sc = c.w / IW;
        ctx.fillStyle = t.panel2;
        ctx.fillRect(0, 0, IW * sc, IH * sc);
        for (let v = 10; v < IH; v += 20) {
          for (let u = 10; u < IW; u += 20) {
            const d = flow(u, v, S.gen);
            const x0 = (u + 0.5) * sc, y0 = (v + 0.5) * sc;
            if (Math.hypot(d[0], d[1]) * sc < 2) { L.draw.dot(ctx, x0, y0, 2, t.accent); continue; }
            L.draw.arrow(ctx, x0, y0, x0 + d[0] * sc, y0 + d[1] * sc, t.accent, 1.8, 7);
          }
        }
        L.draw.dot(ctx, (CX + 0.5) * sc, (CY + 0.5) * sc, 3, t.muted);
        const { u, v } = S.sel;
        const px = (u + 0.5) * sc, py = (v + 0.5) * sc;
        const L0 = 26;
        L.draw.arrow(ctx, px, py, px + Math.cos(S.phi * DEG) * L0, py + Math.sin(S.phi * DEG) * L0, t.accent2, 2.5, 8);
        L.draw.handle(ctx, px, py, t.accent2);
        L.draw.text(ctx, "∇I", px + Math.cos(S.phi * DEG) * (L0 + 10) - 6, py + Math.sin(S.phi * DEG) * (L0 + 10) + 4, t.accent2, { size: 12, bold: true });
        L.draw.text(ctx, `ψ${"₁₂₃₄₅₆"[S.gen]} = 0.1, z = ${S.z.toFixed(1)}`, 8, IH * sc - 8, t.muted, { size: 12 });
      };
      sync();
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <p>Two things to notice. A pixel whose gradient is <b>perpendicular</b> to a component's motion has $J_i\approx0$: it cannot see that motion. A pixel in a <b>flat</b> region has $\nabla I_l=0$, so $J_\mathbf u=0$ and it adds nothing to $H$ or $\mathbf g$. Every pixel with texture votes; the sums combine the votes.</p>

      <h3>Watch it converge</h3>
      <p>The toy below is the same algorithm in 2D with 3 parameters instead of 6: the template $I_v$ (64×64) is the middle of a scene, the live image $I_l$ (128×128) is that scene shifted by $(t_x,t_y)$ and turned by $\theta$. Per pixel, $J = \nabla I_l\cdot[\,I_2\ |\ \partial\mathbf w/\partial\theta\,]$, summed into a 3×3 system. It uses this implementation's safeguard:</p>
      <div class="eq-card"><div class="eq-label">LM safeguard (gn.wgsl)</div>
      $$\big(H + \mu\,\mathrm{diag}(H)\big)\,\psi = -\mathbf g$$
      <div class="parts">
        <span>$\mu$</span><span>damping, starts at $10^{-4}$ on each pyramid level</span>
        <span>cost went down</span><span>accept the step; $\mu\leftarrow\max(\mu/3,\ 10^{-7})$</span>
        <span>cost went up</span><span>reject: go back to the last accepted pose; $\mu\leftarrow10\mu$ (a shorter, more gradient-like step); give up on the level if $\mu>10^4$</span>
        <span>stop</span><span>when $\|\psi\|<10^{-6}$ or the level's iteration budget is spent</span>
      </div></div>
    `);

    // ================================================================ widgets 3 & 5: alignment toys
    function alignFigure(cfg) {
      const fig = L.figure(root, cfg.caption);
      const c = L.canvas(fig.el, { aspect: 0.86 });
      fig.add(c.el);
      const S = { truth: cfg.truth.slice(), occ: cfg.occ ? cfg.occ.slice() : null, pyr: true, robust: !!cfg.robust, tau: 0.1, gb: false, expo: 1, tr: null, running: false, acc: 0, mask: null };
      let A = null;
      const schedule = () => {
        const tau = S.robust ? S.tau : 1e3;
        return S.pyr
          ? [{ lvl: 2, iters: 12, tau: tau * 2 }, { lvl: 1, iters: 12, tau: tau * 1.33 }, { lvl: 0, iters: 10, tau }]
          : [{ lvl: 0, iters: 34, tau }];
      };
      const reset = (redraw = true) => {
        S.tr = makeTracker(A, schedule(), { gainBias: S.gb });
        S.running = false; S.mask = null;
        if (redraw) { c.redraw(); info(); }
      };
      const rebuild = (redraw = true) => { A = build(S.truth, S.occ, S.expo); reset(redraw); };
      const ctl = L.controls(fig.el); fig.add(ctl);
      L.button(ctl, "Step", () => { if (!A) rebuild(false); S.running = false; S.tr.step(); c.redraw(); info(); });
      L.button(ctl, "Run", () => { if (!A) rebuild(false); if (S.tr.finished) reset(false); S.running = true; }, "btn primary");
      L.button(ctl, "Reset", () => { if (!A) rebuild(false); reset(); });
      const ctl2 = L.controls(fig.el); fig.add(ctl2);
      cfg.controls(ctl2, S, rebuild, reset);
      const ro = L.readout(fig.el); fig.add(ro.el);
      const info = () => {
        const tr = S.tr;
        if (!tr) return;
        const p = tr.pose;
        const lvl = tr.finished ? "done" : `level ${schedule()[tr.li].lvl}`;
        const e = Math.hypot(p[0] - S.truth[0], p[1] - S.truth[1]), eth = Math.abs(p[2] - S.truth[2]) / DEG;
        let s = `${lvl} · iteration ${tr.hist.length} · estimate (${p[0].toFixed(2)}, ${p[1].toFixed(2)}, ${(p[2] / DEG).toFixed(2)}°) · error <b>${e.toFixed(2)} px, ${eth.toFixed(2)}°</b>`;
        if (tr.hist.length) s += ` · cost ${tr.best.toExponential(2)}`;
        if (cfg.extraInfo) s += cfg.extraInfo(S);
        ro.html = s;
      };
      L.loop(c, (_, dt) => {
        if (!S.running || !S.tr) return;
        S.acc += dt;
        let n = 0;
        while (S.acc > 0.08 && n < 3) {
          S.acc -= 0.08; n++;
          S.tr.step();
          if (S.tr.finished) { S.running = false; S.acc = 0; break; }
        }
        if (n) { S.mask = null; c.redraw(); info(); }
      });
      const outline = (p) => [[0, 0], [TW - 1, 0], [TW - 1, TW - 1], [0, TW - 1]].map(([x, y]) => warp0(x, y, p));
      const geo = () => {
        const lab = 15, gap = 8;
        const a = Math.min(c.w * 0.54, c.h * 0.6);
        const b = Math.min(c.w - a - gap, (a - lab - gap) / 2);
        const x0 = Math.max(0, (c.w - a - gap - b) / 2);
        return { lab, gap, a, b, x0, sc: a / LW, bx: x0 + a + gap };
      };
      if (cfg.occ) {
        L.drag(c, () => {
          const g = geo();
          return [{ x: g.x0 + (S.occ[0] + S.occ[2] / 2) * g.sc, y: g.lab + (S.occ[1] + S.occ[3] / 2) * g.sc }];
        }, (_, p) => {
          const g = geo();
          S.occ[0] = Math.round(clamp((p.x - g.x0) / g.sc - S.occ[2] / 2, 0, LW - S.occ[2]));
          S.occ[1] = Math.round(clamp((p.y - g.lab) / g.sc - S.occ[3] / 2, 0, LW - S.occ[3]));
          rebuild();
        }, { radius: 26 });
      }
      c.draw = (ctx) => {
        const t = L.theme();
        if (!A) { rebuild(false); info(); }
        const tr = S.tr, p = tr.pose;
        const { lab, gap, a, b, x0, sc, bx } = geo();
        // live image + outlines
        const Lv = A.live[0];
        blit(ctx, LW, LW, x0, lab, a, a, (i, o) => { o[0] = o[1] = o[2] = g8(Lv.data[i]); });
        const toC = (q) => [x0 + (q[0] + 0.5) * sc, lab + (q[1] + 0.5) * sc];
        const poly = (pts, col, dash, w) => { const cp = pts.map(toC); cp.push(cp[0]); L.draw.path(ctx, cp, col, w, dash); };
        poly(outline(S.truth), t.accent, [5, 4], 2);
        poly(outline(p), t.accent2, null, 2.5);
        if (S.occ) {
          const ox = x0 + (S.occ[0] + S.occ[2] / 2) * sc, oy = lab + (S.occ[1] + S.occ[3] / 2) * sc;
          L.draw.handle(ctx, ox, oy, t.accent);
        }
        L.draw.text(ctx, "live I_l", x0, lab - 4, t.muted, { size: 11 });
        // template (+ mask)
        const T = A.tmpl[0];
        let mask = null;
        if (cfg.showMask && tr.hist.length) {
          if (!S.mask) {
            S.mask = new Uint8Array(TW * TW);
            const tau = S.robust ? S.tau : 1e3;
            evaluate(A, 0, p, tau, tr.gain, tr.bias, S.mask);
          }
          mask = S.mask;
        }
        const cols = [null, rgbOf(t.good), rgbOf(t.accent2), rgbOf(t.accent)];
        blit(ctx, TW, TW, bx, lab, b, b, (i, o) => {
          const v = g8(T.data[i]);
          if (mask && mask[i]) {
            const cc = cols[mask[i]];
            o[0] = (v + cc[0]) >> 1; o[1] = (v + cc[1]) >> 1; o[2] = (v + cc[2]) >> 1;
          } else { o[0] = o[1] = o[2] = v; }
        });
        L.draw.text(ctx, mask ? "I_v · used / rejected / out" : "template I_v", bx, lab - 4, t.muted, { size: 11 });
        // residual image at the current estimate
        const res = new Float32Array(TW * TW);
        for (let j = 0; j < TW; j++) {
          for (let i = 0; i < TW; i++) {
            const w = warp0(i, j, p);
            res[j * TW + i] = (w[0] < 0 || w[1] < 0 || w[0] > LW - 1 || w[1] > LW - 1) ? -1 : Math.abs(tr.gain * sample(Lv, w[0], w[1]) + tr.bias - T.data[j * TW + i]);
          }
        }
        const ry = 2 * lab + b + gap;  // second panel of the right column
        blit(ctx, TW, TW, bx, ry, b, b, (i, o) => { const v = res[i] < 0 ? 0 : g8(res[i] * 3); o[0] = o[1] = o[2] = v; });
        L.draw.text(ctx, "|residual| ×3", bx, ry - 4, t.muted, { size: 11 });
        // cost history
        const py0 = lab + a + 26, py1 = c.h - 18, px0 = 34, px1 = c.w - 8;
        if (py1 - py0 > 30) {
          const hist = tr.hist;
          const n = Math.max(schedule().reduce((s, l) => s + l.iters, 0), hist.length);
          const costs = hist.map((h) => Math.log10(Math.max(h.cost, 1e-9)));
          const lo = Math.min(-6, ...costs), hi = Math.max(-1, ...costs);
          const X = (k) => px0 + (k / Math.max(1, n - 1)) * (px1 - px0);
          const Y = (v) => py1 - ((v - lo) / (hi - lo)) * (py1 - py0);
          L.draw.line(ctx, px0, py1, px1, py1, t.line, 1);
          L.draw.line(ctx, px0, py0, px0, py1, t.line, 1);
          for (let e = Math.ceil(lo); e <= Math.floor(hi); e += 2) L.draw.text(ctx, "1e" + e, px0 - 4, Y(e) + 4, t.faint, { size: 11, align: "right" });
          const lvCol = [t.accent, t.accent4, t.accent3];
          for (let k = 1; k < hist.length; k++) L.draw.line(ctx, X(k - 1), Y(costs[k - 1]), X(k), Y(costs[k]), lvCol[hist[k].lvl], 2);
          hist.forEach((h, k) => L.draw.dot(ctx, X(k), Y(costs[k]), 2.5, lvCol[h.lvl]));
          L.draw.text(ctx, "cost (log) · level 2 green, 1 purple, 0 blue", px0 + 4, py0 - 6, t.muted, { size: 11 });
          L.draw.text(ctx, "iteration →", px1, c.h - 4, t.faint, { size: 11, align: "right" });
        }
      };
      return { fig, S, rebuild };
    }

    alignFigure({
      caption: "<b>Gauss–Newton image alignment.</b> Press Run. Orange: current estimate of where the template sits in the live image; blue dashed: the truth. The plot shows the robust cost per iteration. The default motion (8, 5) px and −4° is too far for the full-resolution image alone: switch the pyramid off, Reset and Run to see it stall in a false minimum. Then try small motions: both work.",
      truth: [8, 5, -4 * DEG],
      controls: (ctl, S, rebuild, reset) => {
        L.slider(ctl, { label: "true $t_x$", min: -16, max: 16, step: 0.5, value: S.truth[0], oninput: (v) => { S.truth[0] = v; rebuild(); } });
        L.slider(ctl, { label: "true $t_y$", min: -16, max: 16, step: 0.5, value: S.truth[1], oninput: (v) => { S.truth[1] = v; rebuild(); } });
        L.slider(ctl, { label: "true $\\theta$", min: -10, max: 10, step: 0.5, value: S.truth[2] / DEG, fmt: (v) => v.toFixed(1) + "°", oninput: (v) => { S.truth[2] = v * DEG; rebuild(); } });
        L.toggle(ctl, "coarse-to-fine pyramid (3 levels)", S.pyr, (v) => { S.pyr = v; reset(); });
      },
    });

    // ================================================================ coarse to fine
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Coarse to fine</h3>
      <p>Gauss–Newton only converges from inside the <b>basin</b> around the true answer: the region where the linearisation still points the right way. Fine texture makes the basin narrow, because a few pixels of shift lines the texture up with the <i>wrong</i> neighbour. On a half-resolution image the same motion is half as many pixels and fine texture is averaged away, so the basin, measured in full-resolution pixels, roughly doubles per level.</p>
      <p>So: solve on the coarsest level first, then use its answer to start the next finer level. $\psi$ is a 3D motion, the same on every level; only the pixels change. Level $L$ uses intrinsics $f/2^L$ and $(c+0.5)/2^L-0.5$ (chapter 3).</p>
      <table class="mat">
        <tr><td>level (coarse → fine)</td><td>3</td><td>2</td><td>1</td><td>0</td></tr>
        <tr><td>6DOF iterations (max)</td><td>20</td><td>20</td><td>15</td><td>10</td></tr>
        <tr><td>outlier threshold τ</td><td>0.25</td><td>0.18</td><td>0.12</td><td>0.09</td></tr>
        <tr><td>rotation pre-alignment</td><td>10 it.</td><td>10 it.</td><td>–</td><td>–</td></tr>
      </table>
      <p>(<code>track_iterations</code>, <code>track_thresholds</code>, <code>rotation_levels</code> in <code>dtam/mod.rs</code>; level 0 is the mapping resolution, at most 512 px on the long side.)</p>
    `);

    // ================================================================ widget 4: basin
    {
      const fig = L.figure(root, "<b>The basin at each level.</b> Only $t_x$ is unknown here (truth 0). Curves: cost versus the guess on each level (scaled to the same height). Bars: which starting guesses converge to the truth. Drag the handle to pick a start and watch each level's Gauss–Newton iterates.");
      const c = L.canvas(fig.el, { aspect: 0.72 });
      fig.add(c.el);
      const ro = L.readout(fig.el); fig.add(ro.el);
      const B = { A: null, curves: null, starts: [], ok: null, jobs: [], start: 9, trails: null };
      const R = 24;
      const modes = [
        { name: "level 0 alone", sched: [{ lvl: 0, iters: 25, tau: 1e3 }] },
        { name: "level 1 alone", sched: [{ lvl: 1, iters: 25, tau: 1e3 }] },
        { name: "level 2 alone", sched: [{ lvl: 2, iters: 25, tau: 1e3 }] },
        { name: "coarse → fine (2, 1, 0)", sched: [{ lvl: 2, iters: 10, tau: 1e3 }, { lvl: 1, iters: 10, tau: 1e3 }, { lvl: 0, iters: 10, tau: 1e3 }] },
      ];
      const runFrom = (m, s) => {
        const tr = makeTracker(B.A, modes[m].sched, { free: [1, 0, 0], start: [s, 0, 0] });
        const path = [s];
        while (!tr.finished) { tr.step(); path.push(tr.pose[0]); }
        return path;
      };
      const init = () => {
        B.A = build([0, 0, 0], null, 1);
        B.curves = [0, 1, 2].map((l) => {
          const pts = [];
          for (let k = 0; k <= 96; k++) {
            const S = evaluate(B.A, l, [-R + k * 0.5, 0, 0], 1e3, 1, 0);
            pts.push(S.r2 / Math.max(1, S.nView));
          }
          const mx = Math.max(...pts);
          return pts.map((v) => v / mx);
        });
        for (let s = -R; s <= R + 1e-9; s += 1.5) B.starts.push(s);
        B.ok = modes.map(() => B.starts.map(() => null));
        modes.forEach((_, m) => B.starts.forEach((_, k) => B.jobs.push([m, k])));
        trails();
      };
      const trails = () => {
        B.trails = modes.map((_, m) => runFrom(m, B.start));
        ro.html = modes.map((md, m) => {
          const e = B.trails[m][B.trails[m].length - 1];
          return `${md.name}: ends at <b>${e.toFixed(2)}</b> ${Math.abs(e) < 0.3 ? "✓" : "✗"}`;
        }).join(" · ");
      };
      L.loop(c, () => {
        if (!B.A || !B.jobs.length) return;
        const t0 = performance.now();
        while (B.jobs.length && performance.now() - t0 < 8) {
          const [m, k] = B.jobs.shift();
          const path = runFrom(m, B.starts[k]);
          B.ok[m][k] = Math.abs(path[path.length - 1]) < 0.3;
        }
        c.redraw();
      });
      const geo = () => {
        const px0 = 10, px1 = c.w - 10, top = 8, rows = 4, rowH = 30;
        const cb = c.h - rows * rowH - 26;
        return { px0, px1, top, cb, rowH, X: (x) => px0 + ((x + R) / (2 * R)) * (px1 - px0), inv: (p) => -R + ((p - px0) / (px1 - px0)) * 2 * R };
      };
      L.drag(c, () => { const g = geo(); return [{ x: g.X(B.start), y: g.cb + 10 }]; }, (_, p) => {
        const g = geo();
        B.start = Math.round(clamp(g.inv(p.x), -R, R) * 2) / 2;
        if (B.A) trails();
      });
      c.draw = (ctx) => {
        const t = L.theme();
        if (!B.A) init();
        const g = geo();
        const cols = [t.accent, t.accent4, t.accent3, t.accent2];
        const Y = (v) => g.cb - v * (g.cb - g.top - 14);
        for (let x = -R; x <= R; x += 8) {
          L.draw.line(ctx, g.X(x), g.top, g.X(x), g.cb, t.line, 1);
          L.draw.text(ctx, String(x), g.X(x), g.cb + 26, t.faint, { size: 11, align: "center" });
        }
        L.draw.line(ctx, g.px0, g.cb, g.px1, g.cb, t.muted, 1);
        B.curves.forEach((pts, l) => {
          L.draw.path(ctx, pts.map((v, k) => [g.X(-R + k * 0.5), Y(v)]), cols[l], 2);
        });
        // iterate trails on the single-level curves
        for (let l = 0; l < 3; l++) {
          const pts = B.curves[l];
          const at = (x) => { const k = clamp((x + R) * 2, 0, 96); const i = Math.floor(k), f = k - i; return pts[i] + ((pts[Math.min(96, i + 1)] || pts[i]) - pts[i]) * f; };
          for (const x of B.trails[l]) if (Math.abs(x) <= R) L.draw.dot(ctx, g.X(x), Y(at(x)), 3, cols[l]);
        }
        L.draw.text(ctx, "cost vs guess tₓ (px)", g.px0 + 2, g.top + 10, t.muted, { size: 11 });
        L.draw.handle(ctx, g.X(B.start), g.cb + 10, t.fg);
        // basin bars
        modes.forEach((md, m) => {
          const y = g.cb + 36 + m * g.rowH;
          L.draw.text(ctx, md.name, g.px0, y, cols[m], { size: 11, bold: true });
          const w = ((g.px1 - g.px0) / (2 * R)) * 1.5;
          B.starts.forEach((s, k) => {
            const v = B.ok[m][k];
            ctx.fillStyle = v === null ? t.panel2 : v ? cols[m] : t.line;
            ctx.fillRect(g.X(s) - w / 2, y + 4, w + 0.5, 8);
          });
          const e = B.trails[m][B.trails[m].length - 1];
          if (Math.abs(e) <= R) L.draw.dot(ctx, g.X(e), y + 8, 4, t.fg);
        });
      };
    }

    // ================================================================ robust
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Ignoring what the model does not explain (§2.3.2)</h3>
      <p>A hand in front of the camera, a moving person, a reflection: these pixels have huge residuals, and least squares weighs a residual by its <i>square</i>. A few of them can drag the whole pose. The fix is blunt: a pixel with $|f_\mathbf u|>\tau$ is left out of $H$ and $\mathbf g$ for this iteration.</p>
      <div class="eq-card"><div class="eq-label">Truncated-quadratic cost used to accept/reject steps (gn.wgsl)</div>
      $$\text{cost} = \frac{1}{n_\text{view}}\Big(\sum_{|f_\mathbf u|\le\tau} f_\mathbf u^2 \;+\; n_\text{rejected}\,\tau^2\Big)$$
      <div class="parts">
        <span>$n_\text{view}$</span><span>pixels with a prediction that land inside the live image</span>
        <span>$n_\text{rejected}\,\tau^2$</span><span>an outlier costs a fixed $\tau^2$, however bad: it cannot pull</span>
        <span>$\tau$</span><span>ramps down per level: 0.25, 0.18, 0.12, 0.09 (brightness in [0, 1])</span>
      </div></div>
      <p>Why ramp down? Early, the pose is rough, so even good pixels have sizeable residuals; a tight $\tau$ would throw them away. Late, the pose is close, so a tight $\tau$ removes more of the unmodelled stuff.</p>
      <h4>Gain and bias (this implementation's addition)</h4>
      <p>Auto-exposure changes the brightness of the whole live image, which breaks brightness constancy everywhere at once. The residual is computed as $a\,I_l(\mathbf u_l)+b-I_v(\mathbf u)$, and after every accepted step $a,b$ are refit by 1D least squares over the used pixels:</p>
      <div class="eq-card"><div class="eq-label">Closed-form gain/bias (gn.wgsl)</div>
      $$a = \frac{n\sum I_lI_v - \sum I_l\sum I_v}{n\sum I_l^2-\big(\sum I_l\big)^2},\qquad b = \frac{\sum I_v - a\sum I_l}{n}$$
      <div class="parts">
        <span>$n$</span><span>number of used pixels (sums over the same pixels)</span>
        <span>$a$</span><span>gain, clamped to $[0.5, 2]$; $J_\mathbf u$ is multiplied by $a$ too</span>
        <span>$b$</span><span>bias; the pair carries over to the next frame if tracking used &gt; 50% of pixels</span>
      </div></div>
      <p>This is the line fit of chapter 5 with $x=I_l$, $y=I_v$. It gives some of the lighting invariance the paper suggests getting from normalised cross-correlation (§3.2).</p>
    `);

    alignFigure({
      caption: "<b>Robust rejection with an occluder.</b> A dark “hand” (drag the blue handle) covers part of the live image. Run with rejection off: plain least squares is pulled far off. Turn rejection on: rejected pixels are shown orange on the template, used ones green, blue fell outside the live image. Then darken the live exposure (as auto-exposure would) and compare gain/bias estimation off and on.",
      truth: [6, -4, 4 * DEG],
      occ: [44, 50, 40, 34],
      robust: true,
      showMask: true,
      controls: (ctl, S, rebuild, reset) => {
        L.toggle(ctl, "reject outliers", S.robust, (v) => { S.robust = v; reset(); });
        L.slider(ctl, { label: "final $\\tau$", min: 0.03, max: 0.3, step: 0.01, value: S.tau, oninput: (v) => { S.tau = v; reset(); } });
        L.slider(ctl, { label: "live exposure ×", min: 0.65, max: 1, step: 0.05, value: S.expo, oninput: (v) => { S.expo = v; rebuild(); } });
        L.toggle(ctl, "estimate gain & bias", S.gb, (v) => { S.gb = v; reset(); });
      },
      extraInfo: (S) => ` · gain ${S.tr.gain.toFixed(2)}, bias ${S.tr.bias.toFixed(3)}`,
    });

    // ================================================================ rotation, motion model, failure, pseudocode
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Rotation first (§2.3.1)</h3>
      <p>Before touching the model, the new frame is aligned to the <b>previous live frame</b> with a rotation-only motion, on levels 3 and 2. A pure rotation moves pixels without any depth dependence (look at $\omega_1,\omega_2,\omega_3$ in the widget above), so no model and no depth is needed:</p>
      <div class="eq-card"><div class="eq-label">Inter-frame rotation (track_rot.wgsl)</div>
      $$f_\mathbf u(\boldsymbol\omega) = I_k\big(\pi(K\,R\,K^{-1}\dot{\mathbf u})\big) - I_{k-1}(\mathbf u),\qquad R\leftarrow\exp([\boldsymbol\omega]_\times)\,R$$
      <div class="parts">
        <span>$K^{-1}\dot{\mathbf u}$</span><span>the ray of pixel $\mathbf u$ in the previous frame (any length works)</span>
        <span>$R$</span><span>rotation previous → current; 3 unknowns</span>
        <span>$J_\mathbf u$</span><span>$\mathbf x\times\mathbf a$ with $\mathbf x = RK^{-1}\dot{\mathbf u}$: the rotation half of the 6DOF Jacobian</span>
      </div></div>
      <ul>
        <li><b>Large motions:</b> fast camera shake is mostly rotation, and a small turn is a big image shift: $\Delta u\approx f\tan\Delta\theta$.</li>
        <li><b>Stable with few pixels:</b> 3 unknowns on a tiny image are well determined; 6 are not.</li>
        <li><b>Blur-tolerant:</b> consecutive frames are blurred alike, while the model's rendering is sharp.</li>
      </ul>
      <div class="eq-card"><div class="eq-label">Initial guess: rotation + constant velocity (slam.rs)</div>
      $$\hat T_{wl} = T_{w,k-1}\begin{pmatrix}R^\top & \mathbf t_\text{vel}\\ 0&1\end{pmatrix},\qquad \mathbf t_\text{vel} = \text{translation of } T_{w,k-2}^{-1}T_{w,k-1}$$
      <div class="parts">
        <span>$R^\top$</span><span>the rotation just estimated, as current-in-previous</span>
        <span>$\mathbf t_\text{vel}$</span><span>assume the camera moves as much as last frame (constant velocity)</span>
      </div></div>

      <h3>When tracking fails</h3>
      <p>After alignment, the frame counts as tracked only if (<code>tracking_ok</code>):</p>
      <ul>
        <li>the prediction covered at least 2% of the image,</li>
        <li>more than 50% of the in-view predicted pixels were used (not rejected),</li>
        <li>the RMS residual of the used pixels is below 0.08.</li>
      </ul>
      <p>Otherwise it retries once from the last well-tracked pose. If that fails too, the frame gets the motion-model guess, is not used for mapping, and counts toward the lost state (chapter 12). There is no global relocaliser, as in the paper's evaluation.</p>

      <h3>The whole tracker</h3>
<pre><code>track(I_l):                                  # one live frame
  build pyramid of I_l (4 levels, 2×2 box average)
  R ← rotation_align(I_prev, I_l)            # levels 3, 2; 10 GN steps each
  T̂_wl ← T_w,prev · (Rᵀ, t_vel)              # motion model
  for guess in [T̂_wl, T_last_good]:
    I_v, ξ_v ← render all keyframe meshes at T_wv = guess   # z-buffer, cull oblique
    T_lv ← I;  (a, b) ← last good gain/bias
    for level L in 3, 2, 1, 0:
      μ ← 1e-4;  have_accepted ← false
      repeat iters[L] times:
        H, g, cost ← Σ over pixels u with ξ_v(u) > 0:
            x_v ← π⁻¹(u, ξ_v(u));  x_l ← T_lv x_v;  u_l ← π(K_L x_l)
            skip if u_l outside image
            r ← a·I_l(u_l) + b − I_v(u)
            if |r| > τ[L]: count as rejected; continue
            J ← a · ∇I_l(u_l) · ∂π/∂x(x_l) · R_lv [I | −[x_v]×]
            H += JᵀJ;  g += Jᵀr
        if have_accepted and cost > best: T_lv ← T_accepted; μ ← 10μ
        else: accept (T_accepted ← T_lv, best ← cost, refit a, b); μ ← μ/3
        solve (H + μ·diag H) ψ = −g   (from the accepted H, g)
        T_lv ← T_accepted · exp(ψ);  stop level if |ψ| tiny
    T_wl ← T_wv · T_lv⁻¹
    if tracking_ok: return T_wl
  return T̂_wl   (failed frame)
</code></pre>
      <p>On the GPU the whole loop, every level and iteration, is queued at once and runs without the CPU; chapter 12 explains why that matters.</p>
    `);

    // ================================================================ quiz
    const f3 = (v) => L.fmt(v, 3), f4 = (v) => L.fmt(v, 4);
    L.quiz(root, "tracking", [
      {
        id: "warp", type: "num",
        gen: (r) => {
          const f = r.pick([100, 200, 250, 400]), cx = 160, cy = 120;
          const u = cx + r.int(-60, 60), v = cy + r.int(-40, 40);
          const xi = r.pick([0.25, 0.5, 1, 2]), z = 1 / xi;
          const tx = r.pick([-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]), tz = r.pick([0, -0.1, 0.1, 0.2]);
          const beta = r.pick([-0.002, -0.001, 0.001, 0.002]), gam = r.pick([-0.002, -0.001, 0.001, 0.002]);
          const Iv = r.float(0.3, 0.7, 2);
          const x = ((u - cx) / f) * z, y = ((v - cy) / f) * z;
          const xl = x + tx, yl = y, zl = z + tz;
          const ul = (f * xl) / zl + cx, vl = (f * yl) / zl + cy;
          const Il = 0.5 + beta * (ul - cx) + gam * (vl - cy);
          const res = Il - Iv;
          return {
            q: String.raw`Camera: $f_x=f_y=${f}$, $c=(${cx}, ${cy})$. Predicted pixel $\mathbf u=(${u}, ${v})$ has $\xi_v=${xi}$ and $I_v(\mathbf u)=${Iv}$. The current $\hat T_{lv}$ is a pure translation by $(${tx}, 0, ${tz})$. Near there the live image is the ramp $I_l(u,v)=0.5+(${beta})(u-${cx})+(${gam})(v-${cy})$. Find the live pixel $\mathbf u_l=(u_l,v_l)$ and the residual $f_\mathbf u$ of eq. (20), all to 3 decimals.`,
            answer: [ul, vl, res], labels: ["$u_l$", "$v_l$", "$f_\\mathbf u$"], tol: 0.004,
            explain: String.raw`$\mathbf x_v=\frac1{${xi}}(\frac{${u}-${cx}}{${f}},\frac{${v}-${cy}}{${f}},1)=(${f4(x)}, ${f4(y)}, ${f4(z)})$. Add the translation: $\mathbf x_l=(${f4(xl)}, ${f4(yl)}, ${f4(zl)})$. Project: $u_l=${f}\cdot${f4(xl)}/${f4(zl)}+${cx}=${f3(ul)}$, $v_l=${f3(vl)}$. $I_l(\mathbf u_l)=${f4(Il)}$, so $f_\mathbf u=${f4(Il)}-${Iv}=${f4(res)}$.`,
          };
        },
      },
      {
        id: "projd", type: "num",
        gen: (r) => {
          const f = r.pick([100, 200, 500]), z = r.pick([0.5, 1, 2, 4]), x = r.pick([-0.6, -0.4, -0.2, 0.2, 0.3, 0.5]), y = r.pick([-0.3, 0.1, 0.4]);
          return {
            q: String.raw`A point in the live camera's frame is $\mathbf x_l=(${x}, ${y}, ${z})$ and $f_x=f_y=${f}$. Give the two entries of the first row of $\partial\pi(K\mathbf x)/\partial\mathbf x$ that are not zero: $\partial u/\partial x$ and $\partial u/\partial z$.`,
            answer: [f / z, (-f * x) / (z * z)], labels: ["$\\partial u/\\partial x$", "$\\partial u/\\partial z$"], rtol: 0.01, tol: 0.01,
            explain: String.raw`$u=f_x x/z+c_x$. $\partial u/\partial x=f_x/z=${f}/${z}=${f3(f / z)}$. $\partial u/\partial z=-f_x x/z^2=-${f}\cdot(${x})/${z}^2=${f3((-f * x) / (z * z))}$.`,
          };
        },
      },
      {
        id: "avec", type: "num",
        gen: (r) => {
          const f = r.pick([100, 200]), z = r.pick([1, 2, 4]), x = r.pick([-0.4, -0.2, 0.2, 0.4]), y = r.pick([-0.2, 0.2, 0.6]);
          const gx = r.pick([-0.04, -0.02, 0.01, 0.03, 0.05]), gy = r.pick([-0.03, -0.01, 0.02, 0.04]);
          const a = [(gx * f) / z, (gy * f) / z, -(gx * f * x + gy * f * y) / (z * z)];
          return {
            q: String.raw`At the warped pixel the live gradient is $\nabla I_l=(${gx}, ${gy})$. The point is $\mathbf x_l=(${x}, ${y}, ${z})$, $f_x=f_y=${f}$. Compute the 1×3 row $\mathbf a=\nabla I_l\cdot\partial\pi/\partial\mathbf x$.`,
            answer: a, labels: ["$a_1$", "$a_2$", "$a_3$"], tol: 0.005, rtol: 0.01,
            explain: String.raw`$\partial\pi/\partial\mathbf x=\begin{pmatrix}${f3(f / z)}&0&${f3((-f * x) / (z * z))}\\0&${f3(f / z)}&${f3((-f * y) / (z * z))}\end{pmatrix}$. $a_1=${gx}\cdot${f3(f / z)}=${f3(a[0])}$, $a_2=${gy}\cdot${f3(f / z)}=${f3(a[1])}$, $a_3=${gx}\cdot(${f3((-f * x) / (z * z))})+(${gy})\cdot(${f3((-f * y) / (z * z))})=${f3(a[2])}$.`,
          };
        },
      },
      {
        id: "jrot", type: "num",
        gen: (r) => {
          const xv = [r.pick([-0.4, -0.2, 0.2, 0.5]), r.pick([-0.3, 0.1, 0.3]), r.pick([1, 2, 3])];
          const b = [r.nz(4), r.nz(3), r.pick([-1.5, -0.5, 0.5, 1])];
          const j = la.cross(xv, b);
          return {
            q: String.raw`With $R_{lv}=I$, a pixel has $\mathbf b=\nabla I_l\,\partial\pi/\partial\mathbf x=(${b.join(", ")})$ and $\mathbf x_v=(${xv.join(", ")})$. Its Jacobian is $J_\mathbf u=(\mathbf b,\ \mathbf x_v\times\mathbf b)$. Give the rotation part $(J_4, J_5, J_6)$.`,
            answer: j, labels: ["$J_4$", "$J_5$", "$J_6$"], tol: 0.002,
            explain: String.raw`$\mathbf x_v\times\mathbf b=(y b_3-z b_2,\ z b_1-x b_3,\ x b_2-y b_1)=(${f3(j[0])}, ${f3(j[1])}, ${f3(j[2])})$. This equals $\mathbf b\,(-[\mathbf x_v]_\times)$, the last three columns of $\mathbf b[I\,|\,-[\mathbf x_v]_\times]$.`,
          };
        },
      },
      {
        id: "gn1", type: "num",
        gen: (r) => {
          const J = [r.nz(4), r.nz(4), r.nz(4)], res = [r.pick([-0.2, -0.1, 0.05, 0.1, 0.3]), r.pick([-0.3, -0.05, 0.1, 0.2]), r.pick([-0.1, 0.05, 0.15])];
          const num = J.reduce((s, j, i) => s + j * res[i], 0), den = J.reduce((s, j) => s + j * j, 0);
          const psi = -num / den;
          return {
            q: String.raw`Only one parameter is unknown. Three pixels have Jacobians $J=(${J.join(", ")})$ and residuals $f=(${res.join(", ")})$ at $\psi=0$. What Gauss–Newton step $\psi$ do the normal equations give (4 decimals)?`,
            answer: psi, tol: 2e-4, rtol: 0.005,
            explain: String.raw`$H=\sum J^2=${den}$, $g=\sum J f=${f4(num)}$, $\psi=-g/H=${f4(psi)}$.`,
          };
        },
      },
      {
        id: "lm", type: "num",
        gen: (r) => {
          const mu0 = r.pick([1e-4, 1e-3, 0.01, 0.03]);
          const seq = [0, 1, 2, 3].map(() => r.pick(["up", "down"]));
          let mu = mu0;
          for (const s of seq) mu = s === "up" ? mu * 10 : Math.max(mu / 3, 1e-7);
          return {
            q: String.raw`Mid-level, the solver's damping is $\mu=${mu0}$. The next four candidate steps make the robust cost go: <b>${seq.join(", ")}</b>. Using this implementation's rule, what is $\mu$ afterwards?`,
            answer: mu, rtol: 0.01,
            explain: String.raw`Up (rejected): $\mu\times10$. Down (accepted): $\mu/3$. Order does not matter for a product: $${mu0}\cdot10^{${seq.filter((s) => s === "up").length}}/3^{${seq.filter((s) => s === "down").length}}=${L.fmt(mu, 6)}$.`,
          };
        },
      },
      {
        id: "trunc", type: "num",
        gen: (r) => {
          const tau = r.pick([0.09, 0.12, 0.18]);
          const res = [0, 1, 2, 3, 4, 5].map(() => r.sign() * r.pick([0.02, 0.05, 0.08, 0.1, 0.15, 0.3, 0.5]));
          const used = res.filter((x) => Math.abs(x) <= tau), rej = res.length - used.length;
          const cost = (used.reduce((s, x) => s + x * x, 0) + rej * tau * tau) / res.length;
          return {
            q: String.raw`Level threshold $\tau=${tau}$. Six in-view pixels have residuals $${res.join(",\\ ")}$ (two more pixels fell outside the live image). What is the truncated-quadratic cost used to accept or reject the step (5 decimals)?`,
            answer: cost, rtol: 0.01, tol: 2e-5,
            explain: String.raw`Used: $|f|\le${tau}$: $${used.join(", ") || "none"}$, sum of squares $${f4(used.reduce((s, x) => s + x * x, 0))}$. Rejected: ${rej}, each costs $\tau^2=${f4(tau * tau)}$. Out-of-view pixels are not counted. Cost $=(${f4(used.reduce((s, x) => s + x * x, 0))}+${rej}\cdot${f4(tau * tau)})/6=${L.fmt(cost, 5)}$.`,
          };
        },
      },
      {
        id: "pyrshift", type: "num",
        gen: (r) => {
          const f = r.pick([200, 250, 300, 400]), th = r.pick([1, 2, 3, 4, 5]), lv = r.pick([2, 3]);
          const s = (f * Math.tan(th * DEG)) / 2 ** lv;
          return {
            q: String.raw`Between two frames the camera pans by $${th}^\circ$. At level 0 the focal length is $${f}$ px. By about how many pixels does the image centre shift on pyramid level ${lv}, where the rotation pre-alignment runs? (2 decimals)`,
            answer: s, rtol: 0.02,
            explain: String.raw`At level 0 the shift is $f\tan\theta=${f}\tan${th}^\circ=${f3(f * Math.tan(th * DEG))}$ px. Level ${lv} has focal length $f/2^{${lv}}$, so the shift is $${f3(f * Math.tan(th * DEG))}/${2 ** lv}=${f3(s)}$ px: small enough for Gauss–Newton.`,
          };
        },
      },
      {
        id: "gainbias", type: "num",
        gen: (r) => {
          const a = r.pick([0.8, 0.9, 1.1, 1.2, 1.25]), b = r.pick([-0.06, -0.04, 0.02, 0.05]);
          const Il = r.pick([[0.2, 0.4, 0.6], [0.1, 0.3, 0.8], [0.3, 0.5, 0.7], [0.2, 0.5, 0.6]]);
          const Iv = Il.map((x) => +(a * x + b).toFixed(4));
          return {
            q: String.raw`Three used pixels have live values $I_l=(${Il.join(", ")})$ and predicted values $I_v=(${Iv.join(", ")})$. Compute the closed-form gain $a$ and bias $b$ that best map $I_l$ onto $I_v$.`,
            answer: [a, b], labels: ["$a$", "$b$"], tol: 0.005,
            explain: String.raw`$n=3$, $\sum I_l=${f4(Il.reduce((s, x) => s + x, 0))}$, $\sum I_v=${f4(Iv.reduce((s, x) => s + x, 0))}$, $\sum I_l^2=${f4(Il.reduce((s, x) => s + x * x, 0))}$, $\sum I_lI_v=${f4(Il.reduce((s, x, i) => s + x * Iv[i], 0))}$. Plug in: $a=${a}$, then $b=(\sum I_v-a\sum I_l)/n=${b}$. (The points lie exactly on a line, so the fit is exact.)`,
          };
        },
      },
      {
        id: "oblique", type: "num",
        gen: (r) => {
          const f = 100, u1 = r.int(10, 40), u2 = u1 + 1, z1 = r.pick([1, 1.5, 2]), dz = r.pick([0, 0.02, 0.5, 1]), z2 = z1 + dz;
          const P1 = [(z1 * u1) / f, z1], P2 = [(z2 * u2) / f, z2];
          const d = [P2[0] - P1[0], P2[1] - P1[1]], m = [(P1[0] + P2[0]) / 2, (P1[1] + P2[1]) / 2];
          const alpha = Math.acos(Math.abs(d[0] * m[0] + d[1] * m[1]) / (Math.hypot(...d) * Math.hypot(...m))) / DEG;
          const ang = 90 - alpha;
          return {
            q: String.raw`Seen from above (x sideways, z forward), a keyframe with $f=100$, $c_x=0$ has neighbouring pixels $u=${u1}$ and $u=${u2}$ at depths $${z1}$ and $${z2}$. Their 3D points are joined by a mesh edge. What is the angle (degrees, 1 decimal) between the edge's normal and the keyframe ray to the edge's midpoint? The mesh drops edges steeper than $85^\circ$.`,
            answer: ang, tol: 0.3,
            explain: String.raw`Points: $\mathbf x=z\,(u/f, 1)$: $P_1=(${f4(P1[0])}, ${z1})$, $P_2=(${f4(P2[0])}, ${z2})$. Edge $\mathbf d=(${f4(d[0])}, ${f4(d[1])})$, midpoint ray $\mathbf m=(${f4(m[0])}, ${f4(m[1])})$. The angle between edge and ray is $\arccos\frac{|\mathbf d\cdot\mathbf m|}{\|\mathbf d\|\|\mathbf m\|}=${L.fmt(alpha, 2)}^\circ$, so normal-to-ray is $90^\circ-${L.fmt(alpha, 2)}^\circ=${L.fmt(ang, 2)}^\circ$: ${ang > 85 ? "culled (a bridge across a depth jump)" : "kept"}.`,
          };
        },
      },
      {
        id: "finalpose", type: "num",
        gen: (r) => {
          const cwv = [r.nz(3), r.nz(2), r.int(0, 4)].map((v) => v * 0.5), t = [r.pick([-0.08, -0.04, 0.03, 0.06]), r.pick([-0.02, 0.01, 0.05]), r.pick([-0.1, 0.04, 0.07])];
          const cwl = cwv.map((v, i) => v - t[i]);
          return {
            q: String.raw`The virtual camera was rendered at $T_{wv}$ with $R_{wv}=I$ and centre $\mathbf c_w=(${cwv.join(", ")})$. Alignment converged to $\hat T_{lv}$ with $R_{lv}=I$ and translation $(${t.join(", ")})$. Where is the live camera's centre in the world?`,
            answer: cwl, labels: ["$x$", "$y$", "$z$"], tol: 1e-3,
            explain: String.raw`$T_{wl}=T_{wv}\hat T_{lv}^{-1}$. With identity rotations, $\hat T_{lv}^{-1}$ translates by $-(${t.join(", ")})$, so $\mathbf c=(${cwl.map(f4).join(", ")})$.`,
          };
        },
      },
      {
        id: "motion", type: "num",
        gen: (r) => {
          const a = [r.int(-5, 5) / 10, r.int(-3, 3) / 10, r.int(0, 10) / 10];
          const v = [r.pick([-0.03, -0.02, 0.01, 0.04]), r.pick([-0.01, 0, 0.02]), r.pick([-0.02, 0.01, 0.03])];
          const b = a.map((x, i) => +(x + v[i]).toFixed(3)), p = b.map((x, i) => x + (b[i] - a[i]));
          return {
            q: String.raw`Frames $k{-}2$ and $k{-}1$ were tracked at camera centres $(${a.join(", ")})$ and $(${b.join(", ")})$, all rotations identity, and the rotation pre-alignment finds no turn. Where does the motion model place frame $k$'s first guess?`,
            answer: p, labels: ["$x$", "$y$", "$z$"], tol: 1e-3,
            explain: String.raw`Velocity = last frame's motion $=(${b.map((x, i) => f4(x - a[i])).join(", ")})$. Constant velocity: $(${p.map(f4).join(", ")})$.`,
          };
        },
      },
      {
        id: "fc", type: "mc",
        q: "Why does each Gauss–Newton iteration solve for a small twist $\\psi$ applied as $\\hat T_{lv}\\leftarrow\\hat T_{lv}\\,T(\\psi)$, rather than adjusting the 12 entries of the pose matrix directly?",
        choices: [
          "Six unconstrained numbers always give a valid rigid motion, and the Jacobian is only needed at $\\psi=0$, where it has the simple form $[I\\,|\\,-[\\mathbf x]_\\times]$",
          "Matrix entries cannot be differentiated",
          "The twist makes the cost convex, so one step always reaches the minimum",
          "It avoids having to compute image gradients",
        ],
        answer: 0,
        explain: "Editing matrix entries would break $R^\\top R=I$. The exp map of a twist is always a rotation plus translation, and re-linearising at the current estimate keeps $\\psi$ small. It is still non-convex: that is why the pyramid is needed.",
      },
      {
        id: "contrib", type: "multi",
        q: "Which statements about one pixel's contribution to $H=\\sum J_\\mathbf u^\\top J_\\mathbf u$ and $\\mathbf g=\\sum J_\\mathbf u^\\top f_\\mathbf u$ are true? (select all)",
        choices: [
          "A pixel with no predicted surface ($\\xi_v=0$) is skipped",
          "A pixel whose residual exceeds the level threshold $\\tau$ is skipped this iteration",
          "A pixel in a perfectly uniform region adds nothing even when used, because $\\nabla I_l=0$ makes $J_\\mathbf u=0$",
          "A pixel that projects outside the live image is skipped",
          "Each pixel solves its own 6×6 system and the poses are averaged",
          "Corner pixels are given extra weight, as in KLT",
        ],
        answer: [0, 1, 2, 3],
        explain: "There is one 6×6 system for the whole image, built by summing every usable pixel. No pixel is special; texture decides how much it contributes.",
      },
      {
        id: "cull", type: "mc",
        q: "What goes wrong in the prediction if oblique triangles are not culled?",
        choices: [
          "Triangles joining a foreground object to the background behind it form a fake sheet that paints wrong texture and wrong depth over regions the keyframe never saw",
          "The z-buffer stops working, so far surfaces are drawn over near ones",
          "Coverage drops, because fewer pixels receive a surface",
          "Nothing, culling is only done to save GPU time",
        ],
        answer: 0,
        explain: "At a depth jump, neighbouring keyframe pixels lie on different surfaces. The triangle between them is seen almost edge-on from the keyframe. Keeping it hides the true 'no model' region under a smeared sheet, which feeds wrong residuals to tracking (and inflates coverage).",
      },
      {
        id: "rot", type: "multi",
        q: "Why does the tracker first estimate a rotation-only motion between consecutive live frames on coarse levels? (select all)",
        choices: [
          "A pure rotation moves pixels independently of depth, so no model is needed",
          "Three unknowns are well determined even from the few pixels of a coarse level",
          "Consecutive frames are similarly motion-blurred, unlike the sharp model rendering",
          "It gives a better starting point for the 6DOF alignment when the camera turns quickly",
          "It measures the camera's translation, which the 6DOF step cannot",
          "It replaces the model prediction when coverage is high",
        ],
        answer: [0, 1, 2, 3],
        explain: "The rotation feeds the initial guess $\\hat T_{wl}$ (with constant-velocity translation). Translation needs depth, so it comes from the 6DOF alignment against the model.",
      },
      {
        id: "ramp", type: "mc",
        q: "The outlier threshold goes 0.25 → 0.18 → 0.12 → 0.09 from coarse to fine. Why not use 0.09 everywhere?",
        choices: [
          "On coarse levels the pose is still rough, so even good pixels have large residuals; a tight threshold would reject them and leave too little to align with",
          "Coarse images are darker, so residuals are smaller there",
          "A loose threshold at the end gives a more accurate final pose",
          "The threshold must shrink by a factor of 2 per level, like the image",
        ],
        answer: 0,
        explain: "Tighten as you converge (§2.3.2): loose while the pose is uncertain, tight when it is accurate enough that only unmodelled things have large residuals.",
      },
      {
        id: "fail", type: "multi",
        q: "After alignment, which conditions make this implementation count a frame as a tracking failure? (select all)",
        choices: [
          "Fewer than 2% of pixels had a model prediction",
          "At most 50% of in-view predicted pixels were used (the rest rejected)",
          "The RMS residual of the used pixels is 0.08 or more",
          "The pose moved more than 1 cm from the motion-model guess",
          "The gain came out different from 1",
        ],
        answer: [0, 1, 2],
        explain: "These are <code>tracking_ok</code> plus the coverage check in <code>slam.rs</code>. A failed attempt is retried once from the last good pose.",
      },
    ]);
  }

  DTAM.chapter({
    id: "tracking",
    order: 11,
    title: "Dense tracking",
    subtitle: "Aligning every pixel of the live image to the rendered model (§2.3, eqs 19–21)",
    minutes: 60,
    render,
  });
})();
