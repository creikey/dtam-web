// Chapter 7: the feature-based bootstrap. Epipolar geometry, E and F, the
// 8-point algorithm, RANSAC, focal self-calibration, decomposing E,
// triangulation, scale, PnP and bundle adjustment (calib.rs, sfm.rs, slam.rs).
DTAM.chapter({
  id: "bootstrap",
  order: 7,
  title: "Getting started: two-view geometry",
  subtitle: "From corner tracks to camera poses, 3D points and the focal length",
  minutes: 75,
  render(root, L) {
    const la = L.la;
    const H = (s) => root.insertAdjacentHTML("beforeend", s);
    const PI = Math.PI, D2R = PI / 180;

    // ------------------------------------------------------------ small maths kit (local)
    const mm = la.matMul, T = la.T, mv = la.matVec, cross = la.cross;
    const mm3 = (A, B, C) => mm(mm(A, B), C);
    const eye = la.eye;
    const det3 = (M) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    const Kmat = (f, cx, cy) => [[f, 0, cx], [0, f, cy], [0, 0, 1]];
    const Kinv = (f, cx, cy) => [[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1]];
    const proj = (K, x) => [K[0][0] * x[0] / x[2] + K[0][2], K[1][1] * x[1] / x[2] + K[1][2]];
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const gauss = (r) => Math.sqrt(-2 * Math.log(1 - r.next() + 1e-12)) * Math.cos(2 * PI * r.next());
    const median = (arr) => { const s = arr.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };

    /** Symmetric eigen-decomposition by cyclic Jacobi. vecs[i][k] = component i of eigenvector k. */
    function jacobi(A0) {
      const n = A0.length, A = A0.map((r) => r.slice()), V = eye(n);
      for (let sw = 0; sw < 60; sw++) {
        let off = 0, tot = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { if (i !== j) off += A[i][j] ** 2; tot += A[i][j] ** 2; }
        if (!(off > 1e-30 * tot)) break;
        for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
          const apq = A[p][q];
          if (Math.abs(apq) < 1e-300) continue;
          const th = (A[q][q] - A[p][p]) / (2 * apq);
          const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
          for (let k = 0; k < n; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
          for (let k = 0; k < n; k++) { const a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
        }
      }
      return { vals: A.map((r, i) => r[i]), vecs: V };
    }
    /** Unit vector minimising |A x| given the accumulated AᵀA (smallest eigenvector). */
    function nullVec(ata) {
      const e = jacobi(ata);
      let k = 0;
      for (let i = 1; i < e.vals.length; i++) if (e.vals[i] < e.vals[k]) k = i;
      return e.vecs.map((r) => r[k]);
    }
    /** 3×3 SVD: returns U, V (columns = singular vectors) and S sorted descending. */
    function svd3(A) {
      const e = jacobi(mm(T(A), A));
      const idx = [0, 1, 2].sort((a, b) => e.vals[b] - e.vals[a]);
      const Vc = idx.map((k) => e.vecs.map((r) => r[k]));
      const S = idx.map((k) => Math.sqrt(Math.max(0, e.vals[k])));
      const Uc = [];
      for (let i = 0; i < 2; i++) { const a = mv(A, Vc[i]); const s = S[i] > 1e-300 ? S[i] : 1; Uc.push(a.map((x) => x / s)); }
      const d = la.dot(Uc[0], Uc[1]);
      Uc[1] = Uc[1].map((x, i) => x - d * Uc[0][i]);
      const n = Math.hypot(...Uc[1]) || 1;
      Uc[1] = Uc[1].map((x) => x / n);
      Uc.push(cross(Uc[0], Uc[1]));
      return { U: T(Uc), S, V: T(Vc) };
    }
    function rank2(F) {
      const { U, S, V } = svd3(F);
      return mm3(U, [[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]], T(V));
    }
    /** Hartley normalisation: centroid at 0, mean distance √2. */
    function normalize(p) {
      const n = p.length;
      let mx = 0, my = 0;
      for (const q of p) { mx += q[0]; my += q[1]; }
      mx /= n; my /= n;
      let md = 0;
      for (const q of p) md += Math.hypot(q[0] - mx, q[1] - my);
      md /= n;
      const s = md > 1e-12 ? Math.SQRT2 / md : 1;
      return { n: p.map((q) => [s * (q[0] - mx), s * (q[1] - my)]), T: [[s, 0, -s * mx], [0, s, -s * my], [0, 0, 1]] };
    }
    /** The (normalised) 8-point algorithm exactly as calib.rs does it. */
    function eightPoint(x1, x2, { norm = true, r2 = true } = {}) {
      let n1 = x1, n2 = x2, T1 = eye(3), T2 = eye(3);
      if (norm) { const a = normalize(x1), b = normalize(x2); n1 = a.n; n2 = b.n; T1 = a.T; T2 = b.T; }
      const M = Array.from({ length: 9 }, () => new Array(9).fill(0));
      for (let i = 0; i < n1.length; i++) {
        const [a, b] = n1[i], [c, d] = n2[i];
        const r = [c * a, c * b, c, d * a, d * b, d, a, b, 1];
        for (let p = 0; p < 9; p++) for (let q = 0; q < 9; q++) M[p][q] += r[p] * r[q];
      }
      const f = nullVec(M);
      let F = [[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]];
      if (r2) F = rank2(F);
      return mm3(T(T2), F, T1);
    }
    /** Distance (px) of pixel p to the line l = (a, b, c). */
    const lineDist = (l, p) => Math.abs(l[0] * p[0] + l[1] * p[1] + l[2]) / Math.hypot(l[0], l[1]);
    /** DLT triangulation from views [R, t, ray] (x_c = R x + t). */
    function triangulate(views) {
      const M = Array.from({ length: 4 }, () => new Array(4).fill(0));
      for (const [R, t, ray] of views) {
        const x = ray[0] / ray[2], y = ray[1] / ray[2];
        const p = (r) => [R[r][0], R[r][1], R[r][2], t[r]];
        const p0 = p(0), p1 = p(1), p2 = p(2);
        for (const row of [p2.map((v, i) => v * x - p0[i]), p2.map((v, i) => v * y - p1[i])]) {
          for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) M[a][b] += row[a] * row[b];
        }
      }
      const v = nullVec(M);
      if (Math.abs(v[3]) < 1e-12) return null;
      return [v[0] / v[3], v[1] / v[3], v[2] / v[3]];
    }
    /** The four (R, t) candidates of an essential matrix, as sfm.rs builds them. */
    function decompose(E) {
      let { U, V } = svd3(E);
      if (det3(U) < 0) U = U.map((r) => [r[0], r[1], -r[2]]);
      if (det3(V) < 0) V = V.map((r) => [r[0], r[1], -r[2]]);
      const W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
      const t = [U[0][2], U[1][2], U[2][2]], tn = t.map((x) => -x);
      const Ra = mm3(U, W, T(V)), Rb = mm3(U, T(W), T(V));
      return [[Ra, t], [Ra, tn], [Rb, t], [Rb, tn]];
    }
    /** Clip the image line a u + b v + c = 0 to [0,W]×[0,H]. */
    function clipLine(l, W, Hh) {
      const [a, b, c] = l, out = [];
      const add = (u, v) => { if (u >= -1e-6 && u <= W + 1e-6 && v >= -1e-6 && v <= Hh + 1e-6 && !out.some((p) => Math.hypot(p[0] - u, p[1] - v) < 1e-6)) out.push([u, v]); };
      if (Math.abs(b) > 1e-12) { add(0, -c / b); add(W, -(c + a * W) / b); }
      if (Math.abs(a) > 1e-12) { add(-c / a, 0); add(-(c + b * Hh) / a, Hh); }
      return out.length >= 2 ? [out[0], out[1]] : null;
    }
    /** Fit a W×H image into a canvas, keeping aspect. */
    function fitRect(c, W, Hh, pad = 0) {
      const k = Math.min((c.w - 2 * pad) / W, (c.h - 2 * pad) / Hh);
      const ox = (c.w - W * k) / 2, oy = (c.h - Hh * k) / 2;
      return { k, ox, oy, w: W * k, h: Hh * k, X: (u) => ox + u * k, Y: (v) => oy + v * k, iu: (x) => (x - ox) / k, iv: (y) => (y - oy) / k };
    }
    function imageFrame(ctx, P, t, label) {
      ctx.save();
      ctx.fillStyle = t.panel2; ctx.strokeStyle = t.line; ctx.lineWidth = 1;
      ctx.fillRect(P.ox, P.oy, P.w, P.h); ctx.strokeRect(P.ox, P.oy, P.w, P.h);
      ctx.restore();
      if (label) L.draw.text(ctx, label, P.ox + 6, P.oy + 15, t.muted, { size: 12, bold: true });
    }
    const clipTo = (ctx, P) => { ctx.save(); ctx.beginPath(); ctx.rect(P.ox, P.oy, P.w, P.h); ctx.clip(); };
    function drawCross(ctx, x, y, col, r = 6) {
      L.draw.line(ctx, x - r, y - r, x + r, y + r, col, 2);
      L.draw.line(ctx, x - r, y + r, x + r, y - r, col, 2);
    }
    const f2 = (v) => L.fmt(v, 2), f3 = (v) => L.fmt(v, 3);

    // ================================================================ intro
    H(String.raw`
      <style>
        #bootstrap .mini { font-size: 14px; color: var(--muted); }
        #bootstrap table.plain { border-collapse: collapse; font-size: 14.5px; margin: 8px 0; width: 100%; }
        #bootstrap table.plain td, #bootstrap table.plain th { border-bottom: 1px solid var(--line); padding: 4px 6px; text-align: left; vertical-align: top; }
      </style>
      <p>Dense tracking (chapter 11) needs a 3D model. Dense mapping (chapters 8–10) needs camera poses. At the very start we have neither. The paper breaks this deadlock with "standard point feature based stereo": use a few hundred tracked corners (chapter 6) to recover the first camera poses, then hand over to the dense machinery.</p>
      <p>The corners give us <b>matches</b>: the same 3D point seen at pixel $\mathbf u_1$ in one frame and $\mathbf u_2$ in another. From matches alone, this chapter recovers:</p>
      <ul>
        <li>the <b>focal length</b> $f$ (we don't even assume the camera is calibrated),</li>
        <li>the <b>rotation and translation</b> between frames,</li>
        <li>the <b>3D positions</b> of the corners, up to one unknown overall scale.</li>
      </ul>
      <p>This is the densest chapter. Take it one widget at a time.</p>
      <h3>The plan (what <code>slam.rs</code> does)</h3>
      <pre><code>collect KLT tracks for 60 frames                 (chapter 6)
f0 = estimate_focal(tracks)                      calib.rs  (§ focal)
b  = pick second frame of the initial pair       sfm.rs    (§ initial pair)
F  = RANSAC 8-point on matches(frame 0, b)       (§ 8-point, RANSAC)
E  = Kᵀ F K;  (R, t) = decompose(E)              (§ E → motion)
triangulate the matches                          (§ triangulation)
for each frame 1..59:
    pose = PnP from already-known 3D points      (§ PnP)
    triangulate newly visible tracks
bundle adjust all poses + points + f             (§ bundle adjustment)
scale so the median point depth = 1              (§ scale)
if anything failed: drop the oldest 15 frames, wait, retry</code></pre>
    `);

    // ================================================================ normalised coords
    H(String.raw`
      <h3>1 · Normalised coordinates</h3>
      <p>Geometry is simplest if we first undo the intrinsics. Recall from chapter 3: $K^{-1}\dot{\mathbf u}$ is the ray through pixel $\mathbf u$, scaled to depth 1.</p>
      <div class="eq-card"><div class="eq-label">Normalised image point</div>
      $$\hat{\mathbf x} = K^{-1}\dot{\mathbf u} = \begin{pmatrix}(u-c_x)/f_x\\ (v-c_y)/f_y\\ 1\end{pmatrix}$$
      <div class="parts">
        <span>$\dot{\mathbf u}=(u,v,1)^\top$</span><span>the pixel, homogeneous</span>
        <span>$\hat{\mathbf x}$</span><span>the same point as if the camera had $f=1$ and centre $(0,0)$: a direction in the camera frame</span>
        <span>$(c_x, c_y)$</span><span>principal point; this implementation assumes the image centre, $((w-1)/2, (h-1)/2)$</span>
      </div></div>
      <p class="mini">Example: $f=500$, $(c_x,c_y)=(320,240)$, $\mathbf u=(420,290)$ gives $\hat{\mathbf x}=(0.2,\,0.1,\,1)$. Code: <code>Intrinsics::unproject</code>.</p>
    `);

    // ================================================================ epipolar geometry
    H(String.raw`
      <h3>2 · The epipolar constraint</h3>
      <p>Put the world at camera 1. Camera 2 is related to it by a rotation $R$ and translation $\mathbf t$ (this is $T_{21}$, "camera 2 from camera 1"):</p>
      <div class="eq-card"><div class="eq-label">Relative motion</div>
      $$\mathbf x_2 = R\,\mathbf x_1 + \mathbf t$$
      <div class="parts">
        <span>$\mathbf x_1, \mathbf x_2$</span><span>the same 3D point in camera 1's and camera 2's coordinates</span>
        <span>$R$</span><span>rotation from camera 1's axes to camera 2's</span>
        <span>$\mathbf t$</span><span>camera 1's centre, seen from camera 2 (put $\mathbf x_1 = \mathbf 0$)</span>
      </div></div>
      <p>The 3D point and the two camera centres form a triangle, so the two viewing rays and the <b>baseline</b> (the segment between the centres) lie in one plane: the <b>epipolar plane</b>. Working in camera 2's frame, three vectors lie in that plane:</p>
      <ul>
        <li>$\hat{\mathbf x}_2$: the ray from camera 2 to the point,</li>
        <li>$\mathbf t$: from camera 2 to camera 1,</li>
        <li>$R\hat{\mathbf x}_1$: the direction of ray 1, rotated into camera 2's axes.</li>
      </ul>
      <p>The cross product $\mathbf t\times R\hat{\mathbf x}_1$ is perpendicular to the plane, so its dot product with $\hat{\mathbf x}_2$ is zero. Writing the cross product as the skew matrix $[\mathbf t]_\times$ (chapter 4):</p>
      <div class="eq-card"><div class="eq-label">Epipolar constraint · essential matrix</div>
      $$\hat{\mathbf x}_2^\top E\,\hat{\mathbf x}_1 = 0,\qquad E = [\mathbf t]_\times R$$
      <div class="parts">
        <span>$E$</span><span>the <b>essential matrix</b>, 3×3. It encodes the motion $(R,\mathbf t)$</span>
        <span>$R\hat{\mathbf x}_1$</span><span>ray 1 in camera 2's axes</span>
        <span>$[\mathbf t]_\times R\hat{\mathbf x}_1$</span><span>normal of the epipolar plane</span>
        <span>$\hat{\mathbf x}_2^\top(\cdot) = 0$</span><span>ray 2 lies in that plane</span>
      </div></div>
      <p>It doesn't involve depth at all. Fix $\hat{\mathbf x}_1$ and the constraint is a <b>line</b> in image 2: every depth of the point lands somewhere on it.</p>
      <div class="eq-card"><div class="eq-label">Epipolar line and epipole</div>
      $$\mathbf l_2 = E\,\hat{\mathbf x}_1 = (a,b,c)^\top:\quad a\,x + b\,y + c = 0$$
      <div class="parts">
        <span>$\mathbf l_2$</span><span>line in image 2 where the match of $\hat{\mathbf x}_1$ must be</span>
        <span>epipole $\mathbf e_2$</span><span>the image of camera 1's centre in camera 2 ($\propto \mathbf t$). Every epipolar line passes through it, because $\mathbf t^\top[\mathbf t]_\times = \mathbf 0$</span>
        <span>epipole $\mathbf e_1$</span><span>the image of camera 2's centre in camera 1</span>
      </div></div>
      <p class="mini">Example: camera 2 moved 1 to the right, no rotation: $R=I$, $\mathbf t=(-1,0,0)$. For $\hat{\mathbf x}_1=(0.2,0.1,1)$: $\mathbf l_2=\mathbf t\times\hat{\mathbf x}_1 = (0\cdot1-0\cdot0.1,\ 0\cdot0.2-(-1)\cdot1,\ -1\cdot0.1-0\cdot0.2) = (0, 1, -0.1)$, i.e. the line $y = 0.1$. Sideways motion gives horizontal epipolar lines at the same height.</p>
    `);

    // ---------------------------------------------------------------- widget 1: epipolar explorer
    {
      const W = 320, Hh = 240, f = 240, cx = (W - 1) / 2, cy = (Hh - 1) / 2;
      const K = Kmat(f, cx, cy), Ki = Kinv(f, cx, cy);
      const rs = L.rng(11);
      const scene = Array.from({ length: 28 }, () => [rs.float(-2.6, 2.6, 3), rs.float(-1.6, 1.6, 3), rs.float(3, 9, 3)]);
      const st = { u1: [112, 96] };
      const fig = L.figure(root, "<b>Epipolar explorer.</b> Drag the orange point in image 1. Its match in image 2 must lie on the blue line, wherever the 3D point is (depth ticks 1, 2, 4, 8, 16). Move and turn camera 2; try pure forward motion, and set both moves to 0.");
      const grid = L.el("div", { class: "grid2" });
      fig.add(grid);
      const ca = L.canvas(grid, { aspect: 0.75 });
      const cb = L.canvas(grid, { aspect: 0.75, scroll: true });
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const upd = () => { ca.redraw(); cb.redraw(); info(); };
      const sx = L.slider(ctl, { label: "camera 2 moves right", min: -1, max: 1, step: 0.01, value: 0.6, oninput: upd });
      const sz = L.slider(ctl, { label: "moves forward", min: -1, max: 1, step: 0.01, value: 0.15, oninput: upd });
      const sy = L.slider(ctl, { label: "turns (°)", min: -20, max: 20, step: 0.5, value: -6, oninput: upd });
      const sd = L.slider(ctl, { label: "true depth $z$", min: 1, max: 16, step: 0.1, value: 4, oninput: upd });
      const geo = () => {
        const R12 = la.rotY(sy.value * D2R), c2 = [sx.value, 0, sz.value];
        const R = T(R12), t = la.scale(mv(R, c2), -1);
        const E = mm(la.skew(t), R);
        return { R, t, c2, E, F: mm3(T(Ki), E, Ki), base: Math.hypot(...c2) };
      };
      const toCam2 = (g, x) => la.add(mv(g.R, x), g.t);
      const point3 = (z) => la.scale(mv(Ki, [st.u1[0], st.u1[1], 1]), z);
      const match = (g, z) => { const x2 = toCam2(g, point3(z)); return x2[2] > 0.02 ? proj(K, x2) : null; };
      const inExt = (p) => p && p[0] > -W && p[0] < 2 * W && p[1] > -Hh && p[1] < 2 * Hh;
      ca.draw = (ctx) => {
        const t = L.theme(), g = geo(), P = fitRect(ca, W, Hh);
        imageFrame(ctx, P, t, "image 1");
        clipTo(ctx, P);
        for (const x of scene) { const p = proj(K, x); L.draw.dot(ctx, P.X(p[0]), P.Y(p[1]), 2.2, t.faint); }
        const u2 = match(g, sd.value);
        if (g.base > 0.02 && u2) {
          const seg = clipLine(mv(T(g.F), [u2[0], u2[1], 1]), W, Hh);
          if (seg) L.draw.line(ctx, P.X(seg[0][0]), P.Y(seg[0][1]), P.X(seg[1][0]), P.Y(seg[1][1]), t.accent, 1.5, [5, 4]);
        }
        if (g.base > 0.02 && Math.abs(g.c2[2]) > 1e-3) {
          const e1 = proj(K, g.c2);
          if (inExt(e1)) { drawCross(ctx, P.X(e1[0]), P.Y(e1[1]), t.accent3); L.draw.text(ctx, "e₁", P.X(e1[0]) + 8, P.Y(e1[1]) - 6, t.accent3, { size: 12, bold: true }); }
        }
        ctx.restore();
        L.draw.handle(ctx, P.X(st.u1[0]), P.Y(st.u1[1]), t.accent2);
      };
      cb.draw = (ctx) => {
        const t = L.theme(), g = geo(), P = fitRect(cb, W, Hh);
        imageFrame(ctx, P, t, "image 2");
        clipTo(ctx, P);
        for (const x of scene) { const x2 = toCam2(g, x); if (x2[2] > 0.1) { const p = proj(K, x2); L.draw.dot(ctx, P.X(p[0]), P.Y(p[1]), 2.2, t.faint); } }
        if (g.base <= 0.02) {
          ctx.restore();
          L.draw.text(ctx, "no baseline: t = 0, so E = 0", P.X(W / 2), P.Y(Hh / 2) - 4, t.bad, { size: 12, align: "center", bold: true });
          L.draw.text(ctx, "(no epipolar line, no depth)", P.X(W / 2), P.Y(Hh / 2) + 12, t.bad, { size: 12, align: "center" });
          const u2 = match(g, sd.value);
          if (u2) L.draw.dot(ctx, P.X(u2[0]), P.Y(u2[1]), 6, t.accent2);
          return;
        }
        const l = mv(g.F, [st.u1[0], st.u1[1], 1]);
        const seg = clipLine(l, W, Hh);
        if (seg) L.draw.line(ctx, P.X(seg[0][0]), P.Y(seg[0][1]), P.X(seg[1][0]), P.Y(seg[1][1]), t.accent, 2.2);
        // depth ticks along the line
        for (const z of [1, 2, 4, 8, 16]) {
          const p = match(g, z);
          if (!p) continue;
          L.draw.dot(ctx, P.X(p[0]), P.Y(p[1]), 3.5, t.accent);
          L.draw.text(ctx, String(z), P.X(p[0]) + 5, P.Y(p[1]) - 5, t.muted, { size: 11 });
        }
        const vinf = mv(g.R, mv(Ki, [st.u1[0], st.u1[1], 1]));
        if (vinf[2] > 1e-3) { const p = proj(K, vinf); ctx.save(); ctx.strokeStyle = t.accent; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(P.X(p[0]), P.Y(p[1]), 5, 0, 2 * PI); ctx.stroke(); ctx.restore(); L.draw.text(ctx, "∞", P.X(p[0]) + 6, P.Y(p[1]) + 14, t.muted, { size: 12 }); }
        if (Math.abs(g.t[2]) > 1e-3) {
          const e2 = proj(K, g.t);
          if (inExt(e2)) { drawCross(ctx, P.X(e2[0]), P.Y(e2[1]), t.accent3); L.draw.text(ctx, "e₂", P.X(e2[0]) + 8, P.Y(e2[1]) - 6, t.accent3, { size: 12, bold: true }); }
        }
        const u2 = match(g, sd.value);
        ctx.restore();
        if (u2) L.draw.dot(ctx, P.X(clamp(u2[0], -5, W + 5)), P.Y(clamp(u2[1], -5, Hh + 5)), 6, t.accent2, t.panel);
      };
      const info = () => {
        const g = geo();
        if (g.base <= 0.02) { out.html = "Pure rotation: every depth lands on the same pixel, so the match says nothing about depth."; return; }
        const l = mv(g.F, [st.u1[0], st.u1[1], 1]), s = Math.hypot(l[0], l[1]);
        const u2 = match(g, sd.value);
        const res = u2 ? la.dot([u2[0], u2[1], 1], l) : NaN;
        const ep = Math.abs(g.t[2]) > 1e-3 ? proj(K, g.t) : null;
        out.html = `u₁ = (${st.u1.map((v) => v.toFixed(0)).join(", ")}) · line in image 2: (a,b,c) = (${f3(l[0] / s)}, ${f3(l[1] / s)}, ${f2(l[2] / s)})<br>` +
          (u2 ? `match u₂ = (${u2[0].toFixed(1)}, ${u2[1].toFixed(1)}) · u̇₂ᵀF u̇₁ = ${res.toExponential(1)} (zero: on the line)<br>` : "point is behind camera 2 at this depth<br>") +
          (ep ? `epipole e₂ = (${ep[0].toFixed(0)}, ${ep[1].toFixed(0)})` : "epipole at infinity: t has no forward part, so the epipolar lines are parallel");
      };
      L.drag(ca, () => { const P = fitRect(ca, W, Hh); return [{ x: P.X(st.u1[0]), y: P.Y(st.u1[1]) }]; }, (i, p) => {
        const P = fitRect(ca, W, Hh);
        st.u1 = [clamp(P.iu(p.x), 0, W - 1), clamp(P.iv(p.y), 0, Hh - 1)];
        cb.redraw(); info();
      });
      info();
    }

    H(String.raw`
      <div class="key">A match in image 2 can only be on one line. That turns the 2D search into a 1D search, which is exactly what the cost volume exploits later (chapter 8: each depth layer is a point on the epipolar line).</div>
      <ul>
        <li><b>Forward motion:</b> the epipole moves into the image ("focus of expansion"); lines radiate from it.</li>
        <li><b>No translation:</b> $\mathbf t = \mathbf 0 \Rightarrow E = 0$. Pure rotation tells you nothing about depth.</li>
      </ul>
    `);

    // ================================================================ F
    H(String.raw`
      <h3>3 · The fundamental matrix: the same thing in pixels</h3>
      <p>We don't know $K$ yet (we don't even know $f$), so we fit the constraint to raw pixels. Substitute $\hat{\mathbf x} = K^{-1}\dot{\mathbf u}$:</p>
      <div class="eq-card"><div class="eq-label">Fundamental matrix</div>
      $$\dot{\mathbf u}_2^\top F\,\dot{\mathbf u}_1 = 0,\qquad F = K^{-\top} E\,K^{-1}\quad\Longleftrightarrow\quad E = K^\top F K$$
      <div class="parts">
        <span>$F$</span><span>3×3, works directly on pixel coordinates; fitted from matches without knowing $K$</span>
        <span>$K^{-\top}$</span><span>transpose of $K^{-1}$ (from $\hat{\mathbf x}_2^\top = \dot{\mathbf u}_2^\top K^{-\top}$)</span>
        <span>$F\dot{\mathbf u}_1$</span><span>the epipolar line in image 2, in pixel units</span>
        <span>scale</span><span>$F$ and $2F$ describe the same constraint: $F$ is only defined up to scale</span>
      </div></div>
      <p class="mini">Example (numbers from above, $f=500$, $c=(320,240)$): $F\dot{\mathbf u}_1 = K^{-\top}(E\hat{\mathbf x}_1) = K^{-\top}(0,1,-0.1)^\top = (0,\ 0.002,\ -\tfrac{240}{500}-0.1) = (0, 0.002, -0.58)$: the line $v = 290$, the same row as $\mathbf u_1$.</p>
      <p>How far is a pixel from a line? With the line $(a,b,c)$ in pixel units:</p>
      <div class="eq-card"><div class="eq-label">Point-to-line distance</div>
      $$d = \frac{|a u + b v + c|}{\sqrt{a^2+b^2}}$$
      <div class="parts">
        <span>$a u+bv+c$</span><span>zero on the line; grows linearly as you leave it</span>
        <span>$\sqrt{a^2+b^2}$</span><span>undoes the arbitrary scale of the line, so $d$ is in pixels</span>
      </div></div>
      <p class="mini">Example: line $(3,4,-10)$, pixel $(6,2)$: $d=|18+8-10|/5 = 3.2$ px.</p>
    `);

    // ================================================================ stretch factors
    H(String.raw`
      <h3>4 · Stretch factors (singular values)</h3>
      <p>We need one more tool. Any matrix $A$ acts in three steps: rotate, <b>stretch</b> along perpendicular axes, rotate again. The stretch amounts $\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0$ are its <b>singular values</b>. A library routine (the SVD, like the null-space routine from chapter 2) finds them:</p>
      <div class="eq-card"><div class="eq-label">Singular value decomposition</div>
      $$A = U\begin{pmatrix}\sigma_1&&\\&\sigma_2&\\&&\sigma_3\end{pmatrix}V^\top$$
      <div class="parts">
        <span>$V^\top$</span><span>rotation: lines up the input with the stretch axes</span>
        <span>$\sigma_i$</span><span>stretch factor along axis $i$; a unit circle (sphere) becomes an ellipse (ellipsoid) with semi-axes $\sigma_i$</span>
        <span>$U$</span><span>rotation: places the stretched result</span>
        <span>$\sigma_3=0$</span><span>one direction is squashed flat: the matrix has <b>rank 2</b></span>
      </div></div>
      <p>For a 2×2 matrix you can get them by hand from two facts: the squared stretches add up to the sum of squared entries, and their product is the area factor $|\det A|$:</p>
      <div class="eq-card"><div class="eq-label">2×2 by hand</div>
      $$\sigma_1^2+\sigma_2^2 = \textstyle\sum_{ij} A_{ij}^2 =: S,\qquad \sigma_1\sigma_2 = |\det A| =: D$$
      $$\sigma_{1,2} = \tfrac12\left(\sqrt{S+2D} \pm \sqrt{S-2D}\right)$$
      <div class="parts">
        <span>$S+2D$</span><span>$=(\sigma_1+\sigma_2)^2$</span>
        <span>$S-2D$</span><span>$=(\sigma_1-\sigma_2)^2$</span>
      </div></div>
      <p class="mini">Example: $A=\begin{pmatrix}3&0\\4&5\end{pmatrix}$: $S=50$, $D=15$, $\sqrt{80}=8.944$, $\sqrt{20}=4.472$, so $\sigma_1=6.708$, $\sigma_2=2.236$.</p>
    `);
    {
      const st = { a: [1.6, 0.5], b: [0.6, 1.1] };
      const fig = L.figure(root, "<b>Stretch factors.</b> Drag the two handles: they are the columns of $A$ (where $A$ sends the arrows (1,0) and (0,1)). The unit circle becomes the shaded ellipse; its half-axes are $\\sigma_1$ and $\\sigma_2$. “Flatten” sets $\\sigma_2 = 0$: the closest rank-1 matrix.");
      const c = L.canvas(fig.el, { aspect: 0.62 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const sv2 = () => {
        const [a, cc] = st.a, [b, d] = st.b; // A = [[a, b], [c, d]]
        const p = a * a + cc * cc, q = a * b + cc * d, r = b * b + d * d;
        const m = (p + r) / 2, h = Math.sqrt(((p - r) / 2) ** 2 + q * q);
        const th = 0.5 * Math.atan2(2 * q, p - r);
        const v1 = [Math.cos(th), Math.sin(th)], v2 = [-Math.sin(th), Math.cos(th)];
        const s1 = Math.sqrt(Math.max(0, m + h)), s2 = Math.sqrt(Math.max(0, m - h));
        const A = (v) => [a * v[0] + b * v[1], cc * v[0] + d * v[1]];
        return { s1, s2, v1, v2, u1: A(v1), u2: A(v2), det: a * d - b * cc };
      };
      const view = () => { const s = Math.min(c.w / 6.4, c.h / 4.4); return { s, X: (x) => c.w / 2 + x * s, Y: (y) => c.h / 2 - y * s }; };
      c.draw = (ctx) => {
        const t = L.theme(), V = view(), g = sv2();
        L.draw.line(ctx, 0, V.Y(0), c.w, V.Y(0), t.line, 1);
        L.draw.line(ctx, V.X(0), 0, V.X(0), c.h, t.line, 1);
        const circ = [], ell = [];
        for (let i = 0; i <= 80; i++) {
          const a = (i / 80) * 2 * PI, v = [Math.cos(a), Math.sin(a)];
          circ.push([V.X(v[0]), V.Y(v[1])]);
          ell.push([V.X(st.a[0] * v[0] + st.b[0] * v[1]), V.Y(st.a[1] * v[0] + st.b[1] * v[1])]);
        }
        L.draw.path(ctx, circ, t.faint, 1.5, [4, 4]);
        ctx.save(); ctx.globalAlpha = 0.18; ctx.fillStyle = t.accent; ctx.beginPath(); ell.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.fill(); ctx.restore();
        L.draw.path(ctx, ell, t.accent, 2);
        L.draw.arrow(ctx, V.X(0), V.Y(0), V.X(g.u1[0]), V.Y(g.u1[1]), t.accent2, 2.5);
        if (g.s2 > 0.02) L.draw.arrow(ctx, V.X(0), V.Y(0), V.X(g.u2[0]), V.Y(g.u2[1]), t.accent3, 2.5);
        L.draw.text(ctx, "σ₁", V.X(g.u1[0] * 0.55) + 6, V.Y(g.u1[1] * 0.55) - 6, t.accent2, { size: 13, bold: true });
        if (g.s2 > 0.05) L.draw.text(ctx, "σ₂", V.X(g.u2[0] * 0.55) + 6, V.Y(g.u2[1] * 0.55) - 6, t.accent3, { size: 13, bold: true });
        L.draw.line(ctx, V.X(0), V.Y(0), V.X(st.a[0]), V.Y(st.a[1]), t.muted, 1, [3, 3]);
        L.draw.line(ctx, V.X(0), V.Y(0), V.X(st.b[0]), V.Y(st.b[1]), t.muted, 1, [3, 3]);
        L.draw.handle(ctx, V.X(st.a[0]), V.Y(st.a[1]), t.accent);
        L.draw.handle(ctx, V.X(st.b[0]), V.Y(st.b[1]), t.accent4);
        L.draw.text(ctx, "A·(1,0)", V.X(st.a[0]) + 11, V.Y(st.a[1]) + 4, t.muted, { size: 11 });
        L.draw.text(ctx, "A·(0,1)", V.X(st.b[0]) + 11, V.Y(st.b[1]) + 4, t.muted, { size: 11 });
      };
      const info = () => {
        const g = sv2();
        const S = st.a[0] ** 2 + st.a[1] ** 2 + st.b[0] ** 2 + st.b[1] ** 2;
        out.html = `A = [[${f2(st.a[0])}, ${f2(st.b[0])}], [${f2(st.a[1])}, ${f2(st.b[1])}]] · σ₁ = ${f3(g.s1)}, σ₂ = ${f3(g.s2)}<br>` +
          `σ₁² + σ₂² = ${f3(g.s1 ** 2 + g.s2 ** 2)} = sum of squares ${f3(S)} · σ₁σ₂ = ${f3(g.s1 * g.s2)} = |det A| ${f3(Math.abs(g.det))}`;
      };
      L.drag(c, () => { const V = view(); return [st.a, st.b].map((p) => ({ x: V.X(p[0]), y: V.Y(p[1]) })); }, (i, p) => {
        const V = view();
        const w = [clamp((p.x - c.w / 2) / V.s, -3, 3), clamp((c.h / 2 - p.y) / V.s, -2.1, 2.1)];
        if (i === 0) st.a = w; else st.b = w;
        info();
      });
      L.button(ctl, "Flatten σ₂ (rank 1)", () => {
        const g = sv2();
        // A ← σ₁ u₁ v₁ᵀ with u₁ = A v₁ / σ₁  →  A ← (A v₁) v₁ᵀ
        st.a = [g.u1[0] * g.v1[0], g.u1[1] * g.v1[0]];
        st.b = [g.u1[0] * g.v1[1], g.u1[1] * g.v1[1]];
        c.redraw(); info();
      });
      L.button(ctl, "Reset", () => { st.a = [1.6, 0.5]; st.b = [0.6, 1.1]; c.redraw(); info(); });
      info();
    }
    H(String.raw`
      <p>Why this matters here:</p>
      <ul>
        <li>$[\mathbf t]_\times$ squashes the direction $\mathbf t$ to zero ($\mathbf t\times\mathbf t = \mathbf 0$) and stretches every direction perpendicular to $\mathbf t$ by exactly $|\mathbf t|$. A rotation stretches nothing. So <b>every essential matrix has stretch factors $(\sigma, \sigma, 0)$</b>: two equal, one zero.</li>
        <li>$F = K^{-\top}EK^{-1}$ keeps the zero (rank 2: all epipolar lines meet at the epipole) but in general <b>not</b> the equal pair. We'll use that to find $f$.</li>
      </ul>
    `);

    // ================================================================ 8-point
    H(String.raw`
      <h3>5 · The 8-point algorithm</h3>
      <p>$F$ has 9 unknown entries. Write out $\dot{\mathbf u}_2^\top F\dot{\mathbf u}_1 = 0$ for one match $(x_1,y_1)\leftrightarrow(x_2,y_2)$: it is <b>linear</b> in those entries.</p>
      <div class="eq-card"><div class="eq-label">One match = one linear equation</div>
      $$\big(x_2x_1,\ x_2y_1,\ x_2,\ y_2x_1,\ y_2y_1,\ y_2,\ x_1,\ y_1,\ 1\big)\cdot\big(F_{11},F_{12},F_{13},F_{21},F_{22},F_{23},F_{31},F_{32},F_{33}\big) = 0$$
      <div class="parts">
        <span>coefficient of $F_{ij}$</span><span>$(\text{component } i \text{ of } \dot{\mathbf u}_2)\times(\text{component } j \text{ of } \dot{\mathbf u}_1)$</span>
        <span>stack $n$ matches</span><span>an $n\times 9$ matrix $A$ with $A\mathbf f = \mathbf 0$</span>
        <span>solve</span><span>minimise $|A\mathbf f|$ with $|\mathbf f|=1$ (chapter 2's null-space problem)</span>
      </div></div>
      <p><b>Why 8?</b> 9 entries, but the scale is free, so 8 numbers are unknown: 8 matches pin them down. With more (noisy) matches, the least-squares version finds the best compromise.</p>
      <p>The code solves the null-space problem by building the 9×9 matrix $A^\top A$ (a sum over matches of row·rowᵀ, like the normal equations in chapter 5) and taking its <b>least-stretched direction</b>: the unit $\mathbf f$ that $A^\top A$ stretches least, which is also the one that makes $|A\mathbf f|^2 = \mathbf f^\top A^\top A\,\mathbf f$ smallest.</p>
      <p>Two fixes make it work in practice:</p>
      <div class="eq-card"><div class="eq-label">Hartley normalisation (per image)</div>
      $$\tilde{\mathbf u} = \mathcal T\dot{\mathbf u},\qquad \mathcal T = \begin{pmatrix}s&0&-s\,m_x\\0&s&-s\,m_y\\0&0&1\end{pmatrix},\qquad s = \frac{\sqrt2}{\text{mean distance to }(m_x,m_y)}$$
      <div class="parts">
        <span>$(m_x,m_y)$</span><span>centroid of the points in that image</span>
        <span>$s$</span><span>scale so points sit at distance ≈ √2 from the origin</span>
        <span>undo</span><span>solve for $\tilde F$ on normalised points, then $F = \mathcal T_2^\top\tilde F\,\mathcal T_1$</span>
        <span>why</span><span>raw pixels make $x_2x_1\approx 10^5$ sit next to $1$ in each row; tiny noise then wrecks the smallest direction. Normalised, all columns are $\approx 1$</span>
      </div></div>
      <div class="eq-card"><div class="eq-label">Enforce rank 2</div>
      $$\tilde F = U\,\mathrm{diag}(\sigma_1,\sigma_2,\sigma_3)V^\top\ \longrightarrow\ U\,\mathrm{diag}(\sigma_1,\sigma_2,0)V^\top$$
      <div class="parts">
        <span>why</span><span>a true $F$ has rank 2 (its epipole direction is squashed). A noisy fit doesn't, and then its epipolar lines don't meet in one point</span>
        <span>result</span><span>the closest rank-2 matrix (like "Flatten" in the widget above, one dimension up)</span>
      </div></div>
      <p class="mini">Example row: match $(0.5, -1) \leftrightarrow (2, 0.3)$ gives $(1,\,-2,\,2,\,0.15,\,-0.3,\,0.3,\,0.5,\,-1,\,1)$.</p>
      <pre><code>eight_point(matches):                     // calib.rs
  T1, T2 = hartley(points1), hartley(points2)
  M = zeros(9, 9)
  for (x1,y1) ↔ (x2,y2) in normalised matches:
      r = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
      M += r rᵀ                                // AᵀA
  f = eigenvector of M with smallest eigenvalue
  F̃ = reshape(f, 3×3) (row-major)
  U, σ, V = svd(F̃);  σ3 = 0;  F̃ = U diag(σ) Vᵀ
  return T2ᵀ F̃ T1</code></pre>
    `);

    // ---------------------------------------------------------------- widget 3: 8-point lab
    {
      const W = 320, Hh = 240, f = 260, cx = (W - 1) / 2, cy = (Hh - 1) / 2;
      const K = Kmat(f, cx, cy);
      const R12 = mm(la.rotY(0.1), la.rotX(0.03)), c2 = [0.35, 0.05, 0.9];
      const R = T(R12), t = la.scale(mv(R, c2), -1);
      const rs = L.rng(23);
      const pts = Array.from({ length: 40 }, () => [rs.float(-2, 2, 3), rs.float(-1.4, 1.4, 3), rs.float(3, 7, 3)]);
      const clean1 = pts.map((x) => proj(K, x)), clean2 = pts.map((x) => proj(K, la.add(mv(R, x), t)));
      const eTrue = proj(K, t);
      let noiseR = L.rng(99);
      let nz = [];
      const redrawNoise = () => { nz = pts.map(() => [gauss(noiseR), gauss(noiseR), gauss(noiseR), gauss(noiseR)]); };
      redrawNoise();
      const fig = L.figure(root, "<b>8-point lab.</b> Blue lines: epipolar lines of the <i>estimated</i> $F$ in image 2 for ten matches. Green cross: the true epipole. Add noise, use only 8 matches, and switch off normalisation or rank-2 to see what each fix buys.");
      const c = L.canvas(fig.el, { aspect: 0.75, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      let res = null;
      const compute = () => {
        const n = Math.round(sn.value), s = sN.value;
        const x1 = clean1.map((p, i) => [p[0] + s * nz[i][0], p[1] + s * nz[i][1]]);
        const x2 = clean2.map((p, i) => [p[0] + s * nz[i][2], p[1] + s * nz[i][3]]);
        const F = eightPoint(x1.slice(0, n), x2.slice(0, n), { norm: tN.checked, r2: tR.checked });
        let err = 0;
        for (let i = 0; i < pts.length; i++) err += lineDist(mv(F, [clean1[i][0], clean1[i][1], 1]), clean2[i]);
        const sv = svd3(F).S;
        // epipole = left null vector of F (Fᵀe = 0): least-stretched direction of Fᵀ
        const ee = svd3(T(F)).V;
        const en = [ee[0][2], ee[1][2], ee[2][2]];
        const ep = Math.abs(en[2]) > 1e-9 ? [en[0] / en[2], en[1] / en[2]] : null;
        res = { F, x1, x2, err: err / pts.length, ratio: sv[2] / sv[1], ep, n };
      };
      const upd = () => { compute(); c.redraw(); info(); };
      const sN = L.slider(ctl, { label: "noise (px)", min: 0, max: 3, step: 0.1, value: 1, oninput: upd });
      const sn = L.slider(ctl, { label: "matches used", min: 8, max: 40, step: 1, value: 12, oninput: upd });
      const tN = L.toggle(ctl, "Hartley normalisation", true, upd);
      const tR = L.toggle(ctl, "enforce rank 2", true, upd);
      L.button(ctl, "New noise", () => { noiseR = L.rng((Math.random() * 1e9) >>> 0); redrawNoise(); upd(); });
      c.draw = (ctx) => {
        const t = L.theme(), P = fitRect(c, W, Hh);
        if (!res) return;
        imageFrame(ctx, P, t, "image 2");
        clipTo(ctx, P);
        for (let i = 0; i < 10; i++) {
          const seg = clipLine(mv(res.F, [res.x1[i][0], res.x1[i][1], 1]), W, Hh);
          if (seg) L.draw.line(ctx, P.X(seg[0][0]), P.Y(seg[0][1]), P.X(seg[1][0]), P.Y(seg[1][1]), t.accent, 1.4);
        }
        res.x2.forEach((p, i) => L.draw.dot(ctx, P.X(p[0]), P.Y(p[1]), i < res.n ? 3.2 : 2.2, i < res.n ? t.accent2 : t.faint));
        drawCross(ctx, P.X(eTrue[0]), P.Y(eTrue[1]), t.good, 7);
        if (res.ep && tR.checked) { ctx.save(); ctx.strokeStyle = t.accent; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(P.X(res.ep[0]), P.Y(res.ep[1]), 7, 0, 2 * PI); ctx.stroke(); ctx.restore(); }
        ctx.restore();
      };
      const info = () => {
        if (!res) return;
        out.html = `mean distance of the true matches to the estimated epipolar lines: <b>${res.err.toFixed(2)} px</b><br>` +
          `σ₃/σ₂ of F = ${res.ratio.toExponential(1)} ${res.ratio < 1e-8 ? "(rank 2: all lines meet at one epipole, blue circle)" : "(not rank 2: lines miss each other near the epipole)"}`;
      };
      upd();
    }
    H(String.raw`<p class="mini">Typical result with 1 px noise and 12 matches: about 0.7 px with normalisation, over 10 px without. That's why <code>calib.rs</code> always normalises.</p>`);

    // ================================================================ Sampson
    H(String.raw`
      <h3>6 · Measuring how well a match fits: Sampson distance</h3>
      <p>The <b>algebraic error</b> $e = \dot{\mathbf u}_2^\top F\dot{\mathbf u}_1$ is zero for a perfect match, but its size means nothing: it changes if you rescale $F$, and it isn't in pixels. The code uses the <b>Sampson distance</b>: a first-order estimate of the smallest total squared shift of the two points that makes the match fit exactly.</p>
      <div class="eq-card"><div class="eq-label">Sampson distance (squared, px²) · calib.rs <code>sampson</code></div>
      $$d_S^2 = \frac{e^2}{(F\dot{\mathbf u}_1)_1^2 + (F\dot{\mathbf u}_1)_2^2 + (F^\top\dot{\mathbf u}_2)_1^2 + (F^\top\dot{\mathbf u}_2)_2^2}$$
      <div class="parts">
        <span>$e = \dot{\mathbf u}_2^\top F\dot{\mathbf u}_1$</span><span>algebraic error</span>
        <span>$F\dot{\mathbf u}_1$</span><span>epipolar line in image 2; first two entries = how fast $e$ changes as $\mathbf u_2$ moves</span>
        <span>$F^\top\dot{\mathbf u}_2$</span><span>epipolar line in image 1; same for moving $\mathbf u_1$</span>
        <span>denominator</span><span>makes the result scale-free and in px² (both points may move)</span>
      </div></div>
      <p class="mini">Example: $F=\begin{pmatrix}0&0&2\\0&0&-1\\-2&1&3\end{pmatrix}$, $\mathbf u_1=(1,2)$, $\mathbf u_2=(2,1)$. $F\dot{\mathbf u}_1 = (2,-1,3)$, $e = 2\cdot2 + 1\cdot(-1) + 3 = 6$. $F^\top\dot{\mathbf u}_2 = (-2, 1, 6)$. $d_S^2 = 36/(4+1+4+1) = 3.6$ px². (Distance of $\mathbf u_2$ to its line alone: $6/\sqrt5 = 2.68$ px; Sampson shares the correction between both images.)</p>
      <p>Inlier rule in this implementation: $d_S^2 < 1$ px² while fitting $F$ (<code>inlier_px = 1</code>), and $< 4$ px² when choosing the matches to triangulate.</p>
    `);

    // ================================================================ RANSAC
    H(String.raw`
      <h3>7 · RANSAC: fitting when some matches are wrong</h3>
      <p>KLT sometimes locks onto the wrong spot, or follows a moving object. A few such <b>outliers</b> ruin a least-squares fit. RANSAC ("random sample consensus") fixes this by voting:</p>
      <pre><code>ransac(matches, s, iters, threshold):
  best = none
  repeat iters times:
      sample = s random matches           // s = 8 for F, 4 for H, 2 for a line
      model  = fit(sample)                // exact fit to the minimal sample
      count  = #matches with error(model) < threshold
      if count > best.count: best = (model, count)
  // calib.rs then refits on all inliers of best, twice, keeping a refit
  // only if it has at least as many inliers
  return best</code></pre>
      <p>A sample that contains only inliers produces a model that most of the data agrees with. Any sample containing an outlier produces junk that few points agree with. So the winner of the vote is almost always a clean sample.</p>
    `);
    {
      const st = { pts: [], inl: [], iter: 0, cur: null, curCount: 0, best: null, bestCount: 0, running: false, acc: 0, seed: 5 };
      const fig = L.figure(root, "<b>RANSAC on a line.</b> Press “1 iteration” a few times: two random points (orange) propose a line; points within the threshold band vote for it. The best line so far is blue. Red dashed: ordinary least squares on all points, dragged off by outliers. Green: least squares on the winning inliers.");
      const c = L.canvas(fig.el, { aspect: 0.62, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const nPts = 40;
      const gen = () => {
        const r = L.rng(st.seed);
        const a = r.float(-0.6, 0.6, 2), b = r.float(0.35, 0.65, 2);
        const nOut = Math.round(nPts * so.value);
        const pts = [], inl = [];
        for (let i = 0; i < nPts; i++) {
          if (i < nPts - nOut) { const x = r.float(0.04, 0.96, 3); pts.push([x, a * (x - 0.5) + b + 0.012 * gauss(r)]); inl.push(true); }
          else { pts.push([r.float(0.02, 0.98, 3), r.float(0.02, 0.98, 3)]); inl.push(false); }
        }
        st.pts = pts; st.inl = inl; st.iter = 0; st.cur = null; st.best = null; st.bestCount = 0; st.curCount = 0; st.running = false;
      };
      const distTo = (p, q, r) => { const dx = q[0] - p[0], dy = q[1] - p[1]; return Math.abs(dx * (r[1] - p[1]) - dy * (r[0] - p[0])) / Math.hypot(dx, dy); };
      const lsq = (ps) => { // y = m x + k via normal equations
        let sx = 0, sy = 0, sxx = 0, sxy = 0; const n = ps.length;
        for (const [x, y] of ps) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
        const d = n * sxx - sx * sx; if (Math.abs(d) < 1e-12) return null;
        const m = (n * sxy - sx * sy) / d; return [m, (sy - m * sx) / n];
      };
      const rr = L.rng(77);
      const iterate = () => {
        const i = Math.floor(rr.next() * nPts); let j = Math.floor(rr.next() * (nPts - 1)); if (j >= i) j++;
        const p = st.pts[i], q = st.pts[j];
        if (Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-6) return;
        const count = st.pts.filter((r) => distTo(p, q, r) < sth.value).length;
        st.iter++; st.cur = [i, j]; st.curCount = count;
        if (count > st.bestCount) { st.best = [i, j]; st.bestCount = count; }
      };
      const needed = () => { const w = st.inl.filter(Boolean).length / nPts; return Math.ceil(Math.log(0.01) / Math.log(1 - w * w)); };
      const upd = () => { c.redraw(); info(); };
      const so = L.slider(ctl, { label: "outlier fraction", min: 0, max: 0.8, step: 0.05, value: 0.45, oninput: () => { gen(); upd(); } });
      const sth = L.slider(ctl, { label: "threshold", min: 0.005, max: 0.08, step: 0.005, value: 0.025, oninput: () => { if (st.best) { const [i, j] = st.best; st.bestCount = st.pts.filter((r) => distTo(st.pts[i], st.pts[j], r) < sth.value).length; } upd(); } });
      L.button(ctl, "1 iteration", () => { st.running = false; iterate(); upd(); }, "btn primary");
      L.button(ctl, "Run to 99%", () => { st.running = true; });
      L.button(ctl, "New points", () => { st.seed = (Math.random() * 1e9) >>> 0; gen(); upd(); });
      gen();
      c.draw = (ctx) => {
        const t = L.theme();
        const pad = 14, s = Math.min(c.w - 2 * pad, (c.h - 2 * pad) / 0.62) , ox = (c.w - s) / 2;
        const X = (x) => ox + x * s, Y = (y) => c.h - pad - y * (c.h - 2 * pad);
        ctx.save(); ctx.strokeStyle = t.line; ctx.strokeRect(X(0), Y(1), s, Y(0) - Y(1)); ctx.restore();
        const lineThrough = (p, q, col, w, dash) => {
          const dx = q[0] - p[0], dy = q[1] - p[1];
          if (Math.abs(dx) < 1e-9) { L.draw.line(ctx, X(p[0]), Y(0), X(p[0]), Y(1), col, w, dash); return; }
          const m = dy / dx, y0 = p[1] - m * p[0];
          L.draw.line(ctx, X(0), Y(y0), X(1), Y(y0 + m), col, w, dash);
        };
        ctx.save(); ctx.beginPath(); ctx.rect(X(0), Y(1), s, Y(0) - Y(1)); ctx.clip();
        const all = lsq(st.pts);
        if (all) L.draw.line(ctx, X(0), Y(all[1]), X(1), Y(all[0] + all[1]), t.bad, 1.5, [6, 4]);
        let inBest = null;
        if (st.best) {
          const p = st.pts[st.best[0]], q = st.pts[st.best[1]];
          // threshold band
          const dx = q[0] - p[0], dy = q[1] - p[1], n = Math.hypot(dx, dy), nx = -dy / n * sth.value, ny = dx / n * sth.value;
          ctx.save(); ctx.globalAlpha = 0.14; ctx.fillStyle = t.accent; ctx.beginPath();
          const far = 3;
          ctx.moveTo(X(p[0] - dx * far + nx), Y(p[1] - dy * far + ny)); ctx.lineTo(X(p[0] + dx * far + nx), Y(p[1] + dy * far + ny));
          ctx.lineTo(X(p[0] + dx * far - nx), Y(p[1] + dy * far - ny)); ctx.lineTo(X(p[0] - dx * far - nx), Y(p[1] - dy * far - ny));
          ctx.fill(); ctx.restore();
          lineThrough(p, q, t.accent, 2.5);
          inBest = st.pts.map((r) => distTo(p, q, r) < sth.value);
          const ref = lsq(st.pts.filter((_, k) => inBest[k]));
          if (ref && st.bestCount >= 2) L.draw.line(ctx, X(0), Y(ref[1]), X(1), Y(ref[0] + ref[1]), t.good, 2);
        }
        if (st.cur) lineThrough(st.pts[st.cur[0]], st.pts[st.cur[1]], t.accent2, 1.3);
        ctx.restore();
        st.pts.forEach((p, k) => L.draw.dot(ctx, X(p[0]), Y(p[1]), 3.4, inBest && inBest[k] ? t.accent : t.fg));
        if (st.cur) for (const k of st.cur) { ctx.save(); ctx.strokeStyle = t.accent2; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X(st.pts[k][0]), Y(st.pts[k][1]), 7, 0, 2 * PI); ctx.stroke(); ctx.restore(); }
      };
      const info = () => {
        const w = st.inl.filter(Boolean).length;
        out.html = `iteration ${st.iter} · this sample: ${st.curCount} votes · best: <b>${st.bestCount}</b> votes · true inliers: ${w} of ${nPts}` +
          `<br>iterations for 99% confidence with s = 2, w = ${(w / nPts).toFixed(2)}: N = ${w >= 2 ? needed() : "∞"}`;
      };
      L.loop(c, (_, dt) => {
        if (!st.running) return;
        st.acc += dt;
        while (st.acc > 0.12) { st.acc -= 0.12; iterate(); if (st.iter >= needed()) { st.running = false; break; } }
        upd();
      });
      info();
    }
    H(String.raw`
      <p>How many iterations? If a fraction $w$ of the matches are inliers, a random sample of $s$ is all-inlier with probability $w^s$. After $N$ tries, the chance that <i>none</i> was clean is $(1-w^s)^N$. Demand that this is at most $1-p$:</p>
      <div class="eq-card"><div class="eq-label">RANSAC iteration count</div>
      $$N = \left\lceil \frac{\ln(1-p)}{\ln(1-w^s)} \right\rceil$$
      <div class="parts">
        <span>$w$</span><span>inlier fraction</span>
        <span>$s$</span><span>sample size (8 for $F$)</span>
        <span>$p$</span><span>desired confidence of drawing at least one clean sample (e.g. 0.99)</span>
        <span>$\lceil\cdot\rceil$</span><span>round up</span>
      </div></div>
      <p class="mini">Example: $w=0.5$, $s=8$, $p=0.99$: $w^8 = 1/256$, $N = \ln 0.01/\ln(255/256) = 1177$. Sample size matters enormously: with $s=2$ it would be 17.</p>
    `);
    {
      const fig = L.figure(root, "<b>Iteration calculator.</b> See how fast $N$ explodes with the sample size $s$ and the outlier rate.");
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const info = () => {
        const w = sw.value, s = ss.value, p = sp.value;
        const N = Math.ceil(Math.log(1 - p) / Math.log(1 - w ** s));
        out.html = `w<sup>s</sup> = ${w ** s < 1e-3 ? (w ** s).toExponential(2) : (w ** s).toFixed(4)} · <b>N = ${N.toLocaleString("en-US")}</b> iterations` +
          `<br>This implementation runs a fixed 800 (focal pairs) or 1000 (initial pair) iterations with s = 8: enough for 99% down to w ≈ ${Math.pow(1 - Math.pow(1 - p, 1 / 800), 1 / 8).toFixed(2)} (800) at this p.`;
      };
      const sw = L.slider(ctl, { label: "inlier fraction $w$", min: 0.2, max: 0.99, step: 0.01, value: 0.7, oninput: info });
      const ss = L.slider(ctl, { label: "sample size $s$", min: 1, max: 8, step: 1, value: 8, oninput: info });
      const sp = L.slider(ctl, { label: "confidence $p$", min: 0.5, max: 0.999, step: 0.001, value: 0.99, fmt: (v) => v.toFixed(3), oninput: info });
      info();
    }

    // ================================================================ homography
    H(String.raw`
      <h3>8 · When two views can't be used: the homography test</h3>
      <p>Two situations make $F$ useless even with perfect matches:</p>
      <ul>
        <li><b>Pure rotation</b> ($\mathbf t = \mathbf 0$): no parallax, $E=0$.</li>
        <li><b>All points on one plane</b> (a wall, a table top).</li>
      </ul>
      <p>In both cases every match obeys a simpler rule, a <b>homography</b>: $\dot{\mathbf u}_2 \propto H\dot{\mathbf u}_1$ for a single 3×3 $H$ (8 numbers, fitted from 4 matches). Then a whole family of different $F$ fit the data equally well, and the one RANSAC returns is meaningless.</p>
      <div class="eq-card"><div class="eq-label">Degeneracy test · calib.rs / sfm.rs</div>
      $$\text{reject the pair if}\quad \#\text{inliers}(H) \ \ge\ \rho\cdot\#\text{inliers}(F)$$
      <div class="parts">
        <span>$H$ inliers</span><span>RANSAC with 4-point samples; inlier if $|\pi(H\dot{\mathbf u}_1) - \mathbf u_2| < 1$ px</span>
        <span>$\rho$</span><span>0.9 for focal calibration pairs, 0.85 for the initial pair</span>
        <span>idea</span><span>if the simpler model explains (almost) as much, the extra freedom of $F$ is fitting noise</span>
      </div></div>
      <p class="mini">Example: 400 F-inliers, 350 H-inliers: $350 \ge 0.85\cdot400 = 340$, so the initial-pair search skips this frame and tries a later one.</p>
    `);

    // ================================================================ focal
    H(String.raw`
      <h3>9 · Finding the focal length (self-calibration)</h3>
      <p>We have $F$ from pixels; we want $E = K^\top F K$. If we guess $f$ wrong, $K^\top FK$ is still rank 2 but its two non-zero stretch factors differ. Only the right $f$ gives $\sigma_1 = \sigma_2$ (Mendonça & Cipolla). So: try many $f$, and keep the one where they are most equal.</p>
      <div class="eq-card"><div class="eq-label">Self-calibration cost · calib.rs <code>essential_cost</code></div>
      $$c(f) = \frac{\sigma_1 - \sigma_2}{\sigma_1}\quad\text{of}\quad K_f^\top F K_f,\qquad K_f = \begin{pmatrix}f&0&c_x\\0&f&c_y\\0&0&1\end{pmatrix}$$
      <div class="parts">
        <span>$\sigma_1\ge\sigma_2$</span><span>the two largest stretch factors ($\sigma_3 \approx 0$ anyway)</span>
        <span>divide by $\sigma_1$</span><span>$F$'s scale is arbitrary; the ratio isn't</span>
        <span>$c(f)=0$</span><span>$K_f^\top FK_f$ is a valid essential matrix</span>
        <span>assumptions</span><span>square pixels ($f_x=f_y$), principal point at the image centre</span>
      </div></div>
      <p class="mini">Example: at $f=600$ the stretch factors are $(1, 0.82, 0)$, cost $0.18$; at $f=700$ they are $(1, 0.99, 0)$, cost $0.01$. $700$ is much closer.</p>
      <div class="eq-card"><div class="eq-label">The scan grid (log-spaced)</div>
      $$f_{lo} = \frac{w/2}{\tan(150°/2)},\quad f_{hi} = \frac{w/2}{\tan(10°/2)},\qquad f_i = f_{lo}\left(\frac{f_{hi}}{f_{lo}}\right)^{i/(n-1)},\ i = 0..n-1$$
      <div class="parts">
        <span>field of view</span><span>chapter 3: $\tan(\text{hfov}/2) = (w/2)/f$; scan 150° down to 10°</span>
        <span>log spacing</span><span>equal <i>ratios</i> between steps; the middle of the grid is $\sqrt{f_{lo}f_{hi}}$</span>
        <span>$n$</span><span>800 steps</span>
      </div></div>
      <pre><code>estimate_focal(tracks):                        // calib.rs
  for a = 0, 4, 8, …;  for gap in [10, 20, 30]:     b = a + gap
      matches(a, b): need ≥ 60 and median flow ≥ 8 px
      F, nF = ransac_fundamental(…);  nH = ransac_homography(…)
      if nH ≥ 0.9·nF: skip (degenerate)
      keep F
  for each f_i in the log grid:  curve[i] = mean over kept F of c(f_i)
  f = argmin(curve), refined by a parabola through 3 neighbours
  confidence: repeat with pairs resampled with replacement 400×,
              report the 5th–95th percentile of the minimum</code></pre>
    `);
    {
      const W = 640, Hh = 480, cx = (W - 1) / 2, cy = (Hh - 1) / 2;
      const flo = (W / 2) / Math.tan(75 * D2R), fhi = (W / 2) / Math.tan(5 * D2R);
      const rs = L.rng(41);
      const pts = Array.from({ length: 40 }, () => [rs.float(-2, 2, 3), rs.float(-1.5, 1.5, 3), rs.float(4, 8, 3)]);
      const nr = L.rng(42);
      const nz = pts.map(() => [gauss(nr), gauss(nr), gauss(nr), gauss(nr)]);
      const fig = L.figure(root, "<b>Focal-length scan.</b> A synthetic camera with focal length $f_{true}$ sees 40 points from two poses; $F$ is estimated from the noisy matches (8-point, normalised, rank 2). The curve is $c(f)$. Drag “try $f$” to see the stretch factors. Set the rotation to 0.");
      const c = L.canvas(fig.el, { aspect: 0.58, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      let curve = [], F = null, best = 0;
      const n = 240;
      const grid = Array.from({ length: n }, (_, i) => flo * Math.pow(fhi / flo, i / (n - 1)));
      const cost = (Fm, f) => { const Kf = Kmat(f, cx, cy); const S = svd3(mm3(T(Kf), Fm, Kf)).S; return { c: S[0] > 0 ? (S[0] - S[1]) / S[0] : 1, S }; };
      const compute = () => {
        const ft = sf.value, rot = sr.value * D2R, s = sn.value;
        const K = Kmat(ft, cx, cy);
        const R12 = mm(la.rotY(-rot), la.rotX(0.4 * rot)), c2 = [0.8, 0.1, 0.2];
        const R = T(R12), t = la.scale(mv(R, c2), -1);
        const x1 = [], x2 = [];
        pts.forEach((p, i) => {
          const a = proj(K, p), b = proj(K, la.add(mv(R, p), t));
          x1.push([a[0] + s * nz[i][0], a[1] + s * nz[i][1]]);
          x2.push([b[0] + s * nz[i][2], b[1] + s * nz[i][3]]);
        });
        F = eightPoint(x1, x2);
        curve = grid.map((f) => cost(F, f).c);
        best = 0;
        for (let i = 1; i < n; i++) if (curve[i] < curve[best]) best = i;
      };
      const tryF = () => flo * Math.pow(fhi / flo, st.value);
      const upd = () => { compute(); c.redraw(); info(); };
      const sf = L.slider(ctl, { label: "true focal $f_{true}$ (px)", min: 250, max: 1500, step: 10, value: 700, oninput: upd });
      const sr = L.slider(ctl, { label: "rotation between views (°)", min: 0, max: 20, step: 0.5, value: 10, oninput: upd });
      const sn = L.slider(ctl, { label: "tracking noise (px)", min: 0, max: 2, step: 0.1, value: 0.3, oninput: upd });
      const st = L.slider(ctl, { label: "try $f$", min: 0, max: 1, step: 0.002, value: 0.35, fmt: (v) => Math.round(flo * Math.pow(fhi / flo, v)) + " px", oninput: () => { c.redraw(); info(); } });
      c.draw = (ctx) => {
        const t = L.theme();
        if (!F) return;
        const ymax = Math.max(0.1, ...curve) * 1.05;
        const p = L.plot(c, { x0: Math.log10(flo), x1: Math.log10(fhi), y0: 0, y1: ymax, pad: [14, 12, 30, 40] });
        p.axes(ctx, { xlabel: "f (px, log scale)", ylabel: "c(f)", xticks: 4, yticks: 4, fmt: (v) => (v > 1.5 ? Math.round(10 ** v) : +v.toFixed(2)) });
        L.draw.line(ctx, p.X(Math.log10(sf.value)), p.Y(0), p.X(Math.log10(sf.value)), p.Y(ymax), t.good, 2, [5, 4]);
        L.draw.path(ctx, grid.map((f, i) => [p.X(Math.log10(f)), p.Y(curve[i])]), t.accent, 2.2);
        L.draw.dot(ctx, p.X(Math.log10(grid[best])), p.Y(curve[best]), 5, t.accent2);
        const ft = tryF(), ct = cost(F, ft);
        L.draw.line(ctx, p.X(Math.log10(ft)), p.Y(0), p.X(Math.log10(ft)), p.Y(ymax), t.accent4, 1.5);
        // stretch-factor bars
        const bw = Math.max(12, c.w * 0.035), bh = c.h * 0.32, bx = c.w - 16 - 3 * (bw + 8), by = 18;
        ctx.save(); ctx.fillStyle = t.panel; ctx.globalAlpha = 0.85; ctx.fillRect(bx - 8, by - 6, 3 * (bw + 8) + 10, bh + 30); ctx.restore();
        ct.S.forEach((sv, i) => {
          const h = (sv / ct.S[0]) * bh;
          ctx.save(); ctx.fillStyle = [t.accent2, t.accent3, t.faint][i]; ctx.fillRect(bx + i * (bw + 8), by + bh - h, bw, h); ctx.restore();
          L.draw.text(ctx, ["σ₁", "σ₂", "σ₃"][i], bx + i * (bw + 8) + bw / 2, by + bh + 16, t.muted, { size: 11, align: "center" });
        });
      };
      const info = () => {
        if (!F) return;
        const ft = tryF(), ct = cost(F, ft), fe = grid[best];
        out.html = `minimum at f ≈ <b>${Math.round(fe)} px</b> (true ${sf.value}, error ${(100 * (fe - sf.value) / sf.value).toFixed(1)}%)` +
          `<br>at try f = ${Math.round(ft)}: σ/σ₁ = (1, ${ct.S[1] / ct.S[0] < 0.001 ? "0" : (ct.S[1] / ct.S[0]).toFixed(3)}, ${(ct.S[2] / ct.S[0]).toExponential(0)}) → c = ${ct.c.toFixed(3)}`;
      };
      upd();
    }
    H(String.raw`
      <div class="warn"><b>Rotation is required.</b> With pure translation $E=[\mathbf t]_\times$ is antisymmetric ($E^\top=-E$), and so is $K_f^\top FK_f$ for <i>every</i> $f$. Antisymmetric 3×3 matrices always have stretch factors $(\sigma,\sigma,0)$, so the cost is 0 everywhere: the video contains no information about $f$. Set the rotation slider to 0 to see the flat curve. Averaging over many pairs from a natural hand-held video (which always rotates a little) is what makes the estimate reliable.</div>
      <p>One noisy pair lands within a few percent. <code>calib.rs</code> averages the curves of many pairs, and the result is still only a starting guess: bundle adjustment (§ 15) refines $f$ together with everything else (the repo's synthetic test recovers $f$ to within 2%).</p>
    `);

    // ================================================================ decompose E
    H(String.raw`
      <h3>10 · From $E$ to the motion: four candidates</h3>
      <p>With $f$ known, $E = K^\top FK$. The code first snaps it to an exact essential matrix by setting its stretch factors to $(1,1,0)$ (<code>essential_from_fundamental</code>). The scale of $\mathbf t$ can't be recovered (§ 13), so $|\mathbf t| = 1$.</p>
      <div class="eq-card"><div class="eq-label">Decomposing E · sfm.rs <code>decompose_essential</code></div>
      $$E = U\,\mathrm{diag}(1,1,0)\,V^\top,\quad W = \begin{pmatrix}0&-1&0\\1&0&0\\0&0&1\end{pmatrix}$$
      $$R \in \{\,UWV^\top,\ UW^\top V^\top\,\},\qquad \mathbf t \in \{\,+\mathbf u_3,\ -\mathbf u_3\,\}$$
      <div class="parts">
        <span>$\mathbf u_3$</span><span>third column of $U$: the direction $E^\top$ squashes, i.e. $\mathbf t^\top E = 0$, so $\mathbf t \propto \mathbf u_3$</span>
        <span>$\pm\mathbf u_3$</span><span>$E$ and $-E$ give the same constraint, so the sign of $\mathbf t$ is unknown</span>
        <span>$W$</span><span>a 90° turn about $z$; $W$ vs $W^\top$ gives two rotations that differ by a half-turn about the baseline (the "twisted pair")</span>
        <span>signs</span><span>if $\det U<0$ or $\det V<0$, flip its last column so both $R$ are proper rotations ($\det = +1$)</span>
      </div></div>
      <p>All four $(R,\mathbf t)$ satisfy the epipolar constraint perfectly. But only one puts the 3D points <b>in front of both cameras</b> (positive depth, "cheirality"). The code triangulates every inlier match with each candidate and keeps the one with the most points having $z_1>0$ and $z_2>0$.</p>
      <pre><code>decompose_essential(E, matches):
  U, _, V = svd(E)             // sorted: zero stretch last
  if det U < 0: U[:,2] *= -1;   if det V < 0: V[:,2] *= -1
  for (R, t) in [(UWVᵀ, u3), (UWVᵀ, -u3), (UWᵀVᵀ, u3), (UWᵀVᵀ, -u3)]:
      good = #matches whose triangulated x has x.z > 0 and (R x + t).z > 0
  return candidate with most good    // also used: need good ≥ 50</code></pre>
    `);
    {
      const rs = L.rng(31);
      const base = Array.from({ length: 14 }, () => [rs.float(-1.8, 2.2, 2), rs.float(-0.5, 0.5, 2), rs.float(2.5, 5.5, 2)]);
      const fig = L.figure(root, "<b>The four candidates.</b> Top-down view (x right, z = forward up). Camera 1 at the bottom centre, camera 2 wherever each candidate puts it. Green points are in front of both cameras, red ones are not. Change the true motion; exactly one candidate always wins.");
      const c = L.canvas(fig.el, { aspect: 0.95, scroll: true, maxHeight: 640 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const upd = () => c.redraw();
      const sphi = L.slider(ctl, { label: "direction of motion (°, 0 = right, 90 = forward)", min: -30, max: 210, step: 1, value: 15, oninput: upd });
      const sth = L.slider(ctl, { label: "camera 2 turns (°)", min: -40, max: 40, step: 1, value: -12, oninput: upd });
      const solve = () => {
        const phi = sphi.value * D2R;
        const c2 = [Math.cos(phi), 0, Math.sin(phi)];
        const R12 = la.rotY(sth.value * D2R), R = T(R12), t = la.scale(mv(R, c2), -1);
        const pts = base.filter((x) => la.add(mv(R, x), t)[2] > 0.3);
        const m = pts.map((x) => [x, la.add(mv(R, x), t)]);
        const cands = decompose(mm(la.skew(t), R));
        return cands.map(([Rc, tc]) => {
          let good = 0;
          const rec = m.map(([a, b]) => {
            const X = triangulate([[eye(3), [0, 0, 0], a], [Rc, tc, b]]);
            const ok = !!X && X[2] > 0 && la.add(mv(Rc, X), tc)[2] > 0;
            if (ok) good++;
            return { X, ok };
          });
          const C = la.scale(mv(T(Rc), tc), -1);
          const fwd = [Rc[2][0], Rc[2][1], Rc[2][2]];
          return { Rc, tc, C, fwd, flipped: Rc[1][1] < 0, rec, good, total: m.length };
        });
      };
      const names = ["A: UWVᵀ, +u₃", "B: UWVᵀ, −u₃", "C: UWᵀVᵀ, +u₃", "D: UWᵀVᵀ, −u₃"];
      const camGlyph = (ctx, x, y, dx, dy, col, label) => {
        const a = Math.atan2(dy, dx), L0 = 16;
        ctx.save(); ctx.fillStyle = col; ctx.globalAlpha = 0.85; ctx.beginPath();
        ctx.moveTo(x, y); ctx.lineTo(x + L0 * Math.cos(a - 0.45), y + L0 * Math.sin(a - 0.45)); ctx.lineTo(x + L0 * Math.cos(a + 0.45), y + L0 * Math.sin(a + 0.45));
        ctx.closePath(); ctx.fill(); ctx.restore();
        if (label) L.draw.text(ctx, label, x + 8, y + 14, col, { size: 11, bold: true });
      };
      c.draw = (ctx) => {
        const t = L.theme(), cs = solve();
        let win = 0;
        cs.forEach((cd, i) => { if (cd.good > cs[win].good) win = i; });
        const gap = 6, pw = (c.w - gap) / 2, ph = (c.h - gap) / 2;
        cs.forEach((cd, i) => {
          const px = (i % 2) * (pw + gap), py = Math.floor(i / 2) * (ph + gap);
          const sc = Math.min(pw / 7, (ph - 20) / 10);
          const X = (x) => px + pw / 2 + x * sc, Z = (z) => py + ph - 8 - (z + 3.2) * sc;
          ctx.save();
          ctx.fillStyle = t.panel2; ctx.fillRect(px, py, pw, ph);
          ctx.strokeStyle = i === win ? t.good : t.line; ctx.lineWidth = i === win ? 2.5 : 1; ctx.strokeRect(px + 1, py + 1, pw - 2, ph - 2);
          ctx.beginPath(); ctx.rect(px, py, pw, ph); ctx.clip();
          L.draw.line(ctx, px, Z(0), px + pw, Z(0), t.line, 1, [3, 4]);
          for (const r of cd.rec) if (r.X) {
            L.draw.line(ctx, X(0), Z(0), X(r.X[0]), Z(r.X[2]), t.line, 0.8);
            L.draw.dot(ctx, X(r.X[0]), Z(r.X[2]), 3.2, r.ok ? t.good : t.bad);
          }
          camGlyph(ctx, X(0), Z(0), 0, -1, t.fg, "1");
          camGlyph(ctx, X(cd.C[0]), Z(cd.C[2]), cd.fwd[0], -cd.fwd[2], t.accent, cd.flipped ? "2 (upside down)" : "2");
          ctx.restore();
          L.draw.text(ctx, names[i], px + 6, py + 15, t.muted, { size: 11, bold: true, mono: true });
          L.draw.text(ctx, `${cd.good}/${cd.total} in front`, px + pw - 6, py + 15, i === win ? t.good : t.bad, { size: 11, bold: true, align: "right" });
        });
      };
    }

    // ================================================================ triangulation
    H(String.raw`
      <h3>11 · Triangulation: where two rays meet</h3>
      <p>Given a match and both poses, find the 3D point. With noise the two rays never meet exactly, so again we solve a small null-space problem (the <b>DLT</b>, "direct linear transform"). Write camera $i$'s pose as rows: $[R\,|\,\mathbf t] = \begin{pmatrix}\mathbf p_1^\top\\\mathbf p_2^\top\\\mathbf p_3^\top\end{pmatrix}$ (each row has 4 entries), and the point as $\tilde{\mathbf X} = (X,Y,Z,1)$. The normalised observation $(x, y)$ must equal $(\mathbf p_1^\top\tilde{\mathbf X}/\mathbf p_3^\top\tilde{\mathbf X},\ \mathbf p_2^\top\tilde{\mathbf X}/\mathbf p_3^\top\tilde{\mathbf X})$. Multiply out the division:</p>
      <div class="eq-card"><div class="eq-label">DLT triangulation · sfm.rs <code>triangulate</code></div>
      $$\big(x\,\mathbf p_3 - \mathbf p_1\big)^\top\tilde{\mathbf X} = 0,\qquad \big(y\,\mathbf p_3 - \mathbf p_2\big)^\top\tilde{\mathbf X} = 0$$
      <div class="parts">
        <span>per view</span><span>2 linear equations in the 4 entries of $\tilde{\mathbf X}$</span>
        <span>2+ views</span><span>stack to 4+ rows; minimise $|A\tilde{\mathbf X}|$ with $|\tilde{\mathbf X}|=1$ (smallest eigenvector of the 4×4 $A^\top A$)</span>
        <span>finish</span><span>divide by the 4th entry: $\mathbf x = (\tilde X_1,\tilde X_2,\tilde X_3)/\tilde X_4$</span>
      </div></div>
      <p class="mini">Example (side-by-side cameras): camera 2 at $(b,0,0)$ with no rotation, so $\mathbf t = (-b,0,0)$. Then $x_1 = X/Z$, $x_2 = (X-b)/Z$, so $Z = b/(x_1-x_2)$. With $b = 0.2$, $x_1 = 0.15$, $x_2 = 0.1$: $Z = 0.2/0.05 = 4$, $X = x_1 Z = 0.6$.</p>
      <p>Accuracy depends on the <b>parallax angle</b> between the two rays: small angle, long thin intersection region, bad depth.</p>
      <div class="eq-card"><div class="eq-label">Parallax angle</div>
      $$\alpha = \arccos\frac{(\mathbf x-\mathbf c_1)\cdot(\mathbf x-\mathbf c_2)}{|\mathbf x-\mathbf c_1|\,|\mathbf x-\mathbf c_2|}$$
      <div class="parts">
        <span>$\mathbf c_1, \mathbf c_2$</span><span>camera centres</span>
        <span>rule here</span><span>a track is triangulated only if its rays differ by ≥ 1.5° (0.45° when seeding the initial pair), and its reprojection error is &lt; 3 px in every view</span>
      </div></div>
      <p class="mini">Example: $\mathbf x=(0.6,0,4)$ from $(0,0,0)$ and $(0.2,0,0)$: $\arctan(0.6/4) - \arctan(0.4/4) = 8.53° - 5.71° = 2.82°$.</p>
    `);
    {
      const st = { c2: [1.0, 0], X: [0.5, 4.5], seed: 3 };
      const f = 500, NS = 70;
      let samples = [];
      const regen = () => { const r = L.rng(st.seed); samples = Array.from({ length: NS }, () => [gauss(r), gauss(r)]); };
      regen();
      const fig = L.figure(root, "<b>Two rays, one point.</b> Top-down view. Drag the point and camera 2 (both cameras look straight up the page, $f$ = 500 px). The cloud shows 70 triangulations with random pixel noise. Shrink the baseline or push the point far away and watch the depth error explode.");
      const c = L.canvas(fig.el, { aspect: 0.8 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const upd = () => { c.redraw(); info(); };
      const sn = L.slider(ctl, { label: "pixel noise σ (px)", min: 0, max: 3, step: 0.1, value: 1, oninput: upd });
      L.button(ctl, "New noise", () => { st.seed = (Math.random() * 1e9) >>> 0; regen(); upd(); });
      const view = () => { const s = Math.min(c.w / 6.4, c.h / 8.6); return { s, X: (x) => c.w / 2 + x * s, Y: (z) => c.h - 0.9 * s - z * s, ix: (px) => (px - c.w / 2) / s, iz: (py) => (c.h - 0.9 * s - py) / s }; };
      const obs = (cam, X) => { const dz = X[1] - cam[1]; return dz > 0.05 ? (X[0] - cam[0]) / dz : null; };
      // 2D DLT: homogeneous (X, Z, 1); row for camera at (cx, cz), R = I: (−1, x, cx − x·cz)
      const tri2 = (x1, x2) => {
        const r1 = [-1, x1, 0 - x1 * 0], r2 = [-1, x2, st.c2[0] - x2 * st.c2[1]];
        const v = cross(r1, r2);
        return Math.abs(v[2]) < 1e-12 ? null : [v[0] / v[2], v[1] / v[2]];
      };
      const run = () => {
        const x1 = obs([0, 0], st.X), x2 = obs(st.c2, st.X);
        if (x1 === null || x2 === null) return null;
        const s = sn.value / f;
        const est = samples.map(([a, b]) => tri2(x1 + s * a, x2 + s * b)).filter(Boolean);
        const d1 = [st.X[0], st.X[1]], d2 = [st.X[0] - st.c2[0], st.X[1] - st.c2[1]];
        const ang = Math.acos(clamp(la.dot(d1, d2) / (Math.hypot(...d1) * Math.hypot(...d2)), -1, 1)) / D2R;
        const zs = est.map((e) => e[1]);
        const mz = zs.reduce((a, b) => a + b, 0) / zs.length;
        const sd = Math.sqrt(zs.reduce((a, b) => a + (b - mz) ** 2, 0) / zs.length);
        return { x1, x2, est, ang, sd };
      };
      c.draw = (ctx) => {
        const t = L.theme(), V = view(), r = run();
        for (let z = 0; z <= 8; z += 1) L.draw.line(ctx, 0, V.Y(z), c.w, V.Y(z), t.line, 0.6);
        L.draw.text(ctx, "z = 0", 4, V.Y(0) - 4, t.faint, { size: 11 });
        L.draw.text(ctx, "8", 4, V.Y(8) + 12, t.faint, { size: 11 });
        const cams = [[0, 0], st.c2];
        if (r) {
          const s = sn.value / f;
          [[cams[0], r.x1 + s * samples[0][0]], [cams[1], r.x2 + s * samples[0][1]]].forEach(([cam, x]) => {
            const len = 12, n = Math.hypot(x, 1);
            L.draw.line(ctx, V.X(cam[0]), V.Y(cam[1]), V.X(cam[0] + (x / n) * len), V.Y(cam[1] + len / n), t.accent, 1.2);
          });
          for (const e of r.est) L.draw.dot(ctx, V.X(e[0]), V.Y(e[1]), 2, t.accent2);
        }
        cams.forEach((cam, i) => {
          ctx.save(); ctx.fillStyle = i ? t.accent4 : t.fg; ctx.beginPath();
          ctx.moveTo(V.X(cam[0]), V.Y(cam[1])); ctx.lineTo(V.X(cam[0]) - 9, V.Y(cam[1]) - 15); ctx.lineTo(V.X(cam[0]) + 9, V.Y(cam[1]) - 15); ctx.closePath(); ctx.fill(); ctx.restore();
          L.draw.text(ctx, i ? "cam 2" : "cam 1", V.X(cam[0]), V.Y(cam[1]) + 15, t.muted, { size: 11, align: "center" });
        });
        L.draw.handle(ctx, V.X(st.c2[0]), V.Y(st.c2[1]), t.accent4);
        L.draw.handle(ctx, V.X(st.X[0]), V.Y(st.X[1]), t.accent3);
      };
      const info = () => {
        const r = run();
        if (!r) { out.html = "The point must be in front of both cameras."; return; }
        out.html = `parallax angle α = <b>${r.ang.toFixed(2)}°</b> ${r.ang < 1.5 ? "(below 1.5°: this implementation would not triangulate it yet)" : ""}` +
          `<br>true depth z = ${r.x1 !== null ? st.X[1].toFixed(2) : "–"} · spread of the estimates (std) = <b>${r.sd.toFixed(3)}</b>`;
      };
      L.drag(c, () => { const V = view(); return [{ x: V.X(st.c2[0]), y: V.Y(st.c2[1]) }, { x: V.X(st.X[0]), y: V.Y(st.X[1]) }]; }, (i, p) => {
        const V = view(), w = [clamp(V.ix(p.x), -3, 3), clamp(V.iz(p.y), -0.5, 8)];
        if (i === 0) st.c2 = [w[0], clamp(w[1], -0.5, 3)]; else st.X = w;
        info();
      });
      info();
    }

    // ================================================================ initial pair
    H(String.raw`
      <h3>12 · Choosing the initial pair</h3>
      <p>Frames 0 and 1 are nearly identical: tiny parallax, useless depths, and a homography fits them. So <code>sfm.rs</code> walks forward from frame 10 until the pair (0, b) is good enough:</p>
      <pre><code>best = none
for b = 10 … 59:
    x1, x2 = tracks alive in both frame 0 and frame b
    if count < 50:               stop (too few tracks left)
    if median |x2 − x1| < 25 px: continue (not enough motion)
    F, nF = RANSAC 8-point (1000 iterations, 1 px)
    nH    = RANSAC homography
    if nH ≥ 0.85·nF:             continue (degenerate)
    inliers = matches with Sampson < 4 px²
    (R, t), good = decompose(Kᵀ F K, inliers)
    if good < 50:                continue
    angle = median parallax angle of the triangulated inliers
    remember b if angle is the largest so far
    if angle ≥ 1.5°:             stop searching
use the remembered b (or fail: "no frame pair with enough parallax")</code></pre>
      <p>Then frame 0 gets pose $I$, frame b gets $(R, \mathbf t)$, and all tracks seen in both are triangulated (with the relaxed 0.45° parallax rule, to seed the map).</p>
    `);

    // ================================================================ scale
    H(String.raw`
      <h3>13 · Scale ambiguity</h3>
      <p>Double every distance in the world (the points <i>and</i> the camera baseline) and every image stays exactly the same: $\pi(K(s\mathbf x)) = \pi(K\mathbf x)$ because projection divides by depth. So from images alone the reconstruction is known only up to one global scale factor.</p>
    `);
    {
      const rs = L.rng(57);
      const pts = Array.from({ length: 7 }, () => [rs.float(-1.3, 1.5, 2), rs.float(1.4, 3.6, 2)]);
      const c2 = [0.6, 0], yaw = -5 * D2R, f = 150;
      const fig = L.figure(root, "<b>Same images, any scale.</b> The slider scales the whole scene (points and camera 2's position) about camera 1. The two 1D images below never change. “Normalise” picks the scale this implementation uses: median point depth = 1.");
      const c = L.canvas(fig.el, { aspect: 0.85, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const upd = () => { c.redraw(); info(); };
      const ss = L.slider(ctl, { label: "scale $s$", min: 0.25, max: 2.5, step: 0.01, value: 1, oninput: upd });
      const med = median(pts.map((p) => p[1]));
      L.button(ctl, "Normalise (median depth 1)", () => { ss.value = +(1 / med).toFixed(3); upd(); });
      const cols = (t) => [t.accent, t.accent2, t.accent3, t.accent4, t.good, t.bad, t.fg];
      const img = (cam, rot, p) => {
        const dx = p[0] - cam[0], dz = p[1] - cam[1];
        const xc = Math.cos(rot) * dx - Math.sin(rot) * dz, zc = Math.sin(rot) * dx + Math.cos(rot) * dz;
        return zc > 0.01 ? f * xc / zc : null;
      };
      c.draw = (ctx) => {
        const t = L.theme(), s = ss.value, C = cols(t);
        const topH = c.h * 0.7, sc = Math.min(c.w / 7, (topH - 20) / 10);
        const X = (x) => c.w / 2 + x * sc, Z = (z) => topH - 12 - z * sc;
        for (let z = 0; z <= 9; z++) { L.draw.line(ctx, 0, Z(z), c.w, Z(z), t.line, 0.6); if (z % 3 === 0) L.draw.text(ctx, String(z), 4, Z(z) - 3, t.faint, { size: 11 }); }
        const cam2 = [c2[0] * s, c2[1] * s];
        pts.forEach((p, i) => {
          const q = [p[0] * s, p[1] * s];
          L.draw.line(ctx, X(0), Z(0), X(q[0]), Z(q[1]), t.line, 0.8);
          L.draw.line(ctx, X(cam2[0]), Z(cam2[1]), X(q[0]), Z(q[1]), t.line, 0.8);
          L.draw.dot(ctx, X(q[0]), Z(q[1]), 4.5, C[i]);
        });
        L.draw.dot(ctx, X(0), Z(0), 6, t.fg);
        L.draw.dot(ctx, X(cam2[0]), Z(cam2[1]), 6, t.muted);
        L.draw.text(ctx, "cam 1", X(0) - 8, Z(0) + 16, t.muted, { size: 11, align: "right" });
        L.draw.text(ctx, "cam 2", X(cam2[0]) + 8, Z(cam2[1]) + 16, t.muted, { size: 11 });
        // 1D images
        const stripW = Math.min(c.w - 90, 360), sx0 = 70;
        [[0, 0, [0, 0]], [1, yaw, cam2]].forEach(([k, rot, cam], row) => {
          const y = topH + 16 + row * ((c.h - topH - 16) / 2);
          L.draw.text(ctx, `image ${k + 1}`, 6, y + 4, t.muted, { size: 11, bold: true });
          L.draw.line(ctx, sx0, y, sx0 + stripW, y, t.faint, 6);
          pts.forEach((p, i) => {
            const u = img(cam, rot, [p[0] * s, p[1] * s]);
            if (u === null) return;
            const px = sx0 + stripW / 2 + (u / 150) * (stripW / 2);
            L.draw.line(ctx, px, y - 8, px, y + 8, C[i], 3);
          });
        });
      };
      const info = () => {
        const s = ss.value;
        out.html = `baseline |c₂| = ${(Math.hypot(...c2) * s).toFixed(2)} · median depth = ${(med * s).toFixed(2)} · images: unchanged`;
      };
      info();
    }
    H(String.raw`
      <div class="eq-card"><div class="eq-label">Fixing the gauge · sfm.rs step 5</div>
      $$s = \frac{1}{\operatorname{median}_j z_j},\qquad \mathbf x_j \leftarrow s\,\mathbf x_j,\qquad \mathbf t_i \leftarrow s\,\mathbf t_i$$
      <div class="parts">
        <span>$z_j$</span><span>depth of point $j$ in frame 0 (the world frame)</span>
        <span>$\mathbf t_i$</span><span>translation of every pose; rotations are unchanged</span>
        <span>result</span><span>world = camera of frame 0, median scene depth = 1 unit. All later maps (chapters 8–12) use these units</span>
      </div></div>
      <p class="mini">Example: depths $2.1, 3.4, 2.8, 5.0, 3.1$; median $3.1$; $s = 0.323$. A baseline of 0.62 becomes 0.2.</p>
      <p>The same freedom exists for the whole world frame (where "zero" is and which way the axes point). Fixing it is called choosing the <b>gauge</b>: here, frame 0's camera is the world, and bundle adjustment keeps frame 0 fixed.</p>
    `);

    // ================================================================ PnP
    H(String.raw`
      <h3>14 · Adding frames: pose from known points (PnP)</h3>
      <p>Once some 3D points exist, any frame that sees them can be posed directly: find the $T_{cw}$ that makes the known points project onto their tracked pixels. This is "Perspective-n-Point", solved here with Gauss–Newton (chapter 5).</p>
      <div class="eq-card"><div class="eq-label">Reprojection residual</div>
      $$\mathbf r_j = \mathbf u_j - \pi\big(K\,\mathbf x_{c,j}\big),\qquad \mathbf x_{c,j} = T_{cw}\,\mathbf x_{w,j}$$
      <div class="parts">
        <span>$\mathbf u_j$</span><span>tracked pixel of point $j$ in this frame</span>
        <span>$\mathbf x_{w,j}$</span><span>its 3D position (world)</span>
        <span>$\mathbf r_j$</span><span>2D error in pixels; minimise $\sum_j \rho(|\mathbf r_j|)$ with a Huber $\rho$ (threshold 1.5 px)</span>
      </div></div>
      <p>The Jacobian follows the chain rule (chapter 4 gave the second factor, the derivative of a transformed point with respect to a small twist $\psi$ applied on the left):</p>
      <div class="eq-card"><div class="eq-label">PnP Jacobian (2×6) · sfm.rs <code>solve_pose</code></div>
      $$\frac{\partial\,\pi(K\mathbf x_c)}{\partial\psi} = \underbrace{\begin{pmatrix}\tfrac fz&0&-\tfrac{f x}{z^2}\\[2pt]0&\tfrac fz&-\tfrac{f y}{z^2}\end{pmatrix}}_{\partial\pi/\partial\mathbf x_c\ (2\times3)}\ \underbrace{\big[\,I_3\ \big|\ -[\mathbf x_c]_\times\big]}_{\partial\mathbf x_c/\partial\psi\ (3\times6)}$$
      <div class="parts">
        <span>$(x,y,z) = \mathbf x_c$</span><span>the point in camera coordinates</span>
        <span>$f/z$</span><span>moving the point sideways by 1 moves the pixel by $f/z$</span>
        <span>$-fx/z^2$</span><span>moving it away shrinks its offset from the centre</span>
        <span>$\psi = (\mathbf v, \boldsymbol\omega)$</span><span>translation first (this implementation); update $T_{cw} \leftarrow \exp(\psi)\,T_{cw}$</span>
      </div></div>
      <p class="mini">Example: $f=500$, $\mathbf x_c = (0.4,-0.2,2)$: $\partial\pi/\partial\mathbf x_c = \begin{pmatrix}250&0&-50\\0&250&25\end{pmatrix}$.</p>
      <p>Each point adds $w J^\top J$ to a 6×6 matrix and $w J^\top\mathbf r$ to a 6-vector; solve, update, repeat 15 times. The weight $w$ implements Huber: $w = 1$ if $|\mathbf r|\le 1.5$ px, else $w = 1.5/|\mathbf r|$ (large errors count linearly, not quadratically).</p>
      <pre><code>solve_pose(points3d, pixels, T_cw):          // start: previous frame's pose
  repeat 15 times:
      Hm = 0 (6×6), g = 0 (6)
      for (x_w, u) in matches:
          x_c = T_cw · x_w;  if x_c.z ≤ 0: skip
          r = u − project(K, x_c)
          w = 1 if |r| ≤ 1.5 else 1.5/|r|
          J = dproj(x_c, f) · [I | −[x_c]×]
          Hm += w JᵀJ;   g += w Jᵀr
      δ = solve(Hm, g)                       // Cholesky
      T_cw = exp(δ) · T_cw
      if |δ| < 1e-10: break

for each frame f = 1 … 59:                   // sfm.rs, incremental
    need ≥ 15 known points visible, else fail
    pose[f] = solve_pose(…)
    triangulate tracks now seen in ≥ 2 posed frames (1.5°, < 3 px)</code></pre>
    `);

    // ================================================================ BA
    H(String.raw`
      <h3>15 · Bundle adjustment</h3>
      <p>PnP and triangulation each fix one thing while holding the rest. Errors accumulate. <b>Bundle adjustment</b> refines everything at once: all poses (except frame 0, the gauge), all points, and the focal length.</p>
      <div class="eq-card"><div class="eq-label">Bundle adjustment cost · sfm.rs <code>bundle_adjust</code></div>
      $$\min_{\{T_i\},\{\mathbf x_j\},f}\ \sum_{(i,j)\ \text{observed}} \rho\Big(\big|\mathbf u_{ij} - \pi\big(K_f\,T_i\,\mathbf x_j\big)\big|\Big)$$
      <div class="parts">
        <span>$(i,j)$</span><span>frame $i$ saw track $j$ at pixel $\mathbf u_{ij}$</span>
        <span>$T_i$</span><span>$T_{cw}$ of frame $i$ (6 unknowns each; frame 0 fixed)</span>
        <span>$\mathbf x_j$</span><span>3D point $j$ (3 unknowns each)</span>
        <span>$f$</span><span>focal length (1 unknown); $\partial\pi/\partial f = (x/z,\ y/z)$</span>
        <span>$\rho$</span><span>Huber, 1.5 px</span>
      </div></div>
      <p>It's Levenberg–Marquardt (chapter 5): build $J^\top J$ and $J^\top\mathbf r$, damp the diagonal, solve, accept if the cost drops. The problem is size: 59 moving frames and 2000 points is $6\cdot59+1+3\cdot2000 = 6355$ unknowns. But the matrix has structure. Order the unknowns cameras-first, points-second:</p>
      <div class="eq-card"><div class="eq-label">Block structure and the Schur complement</div>
      $$\begin{pmatrix}C & W\\ W^\top & V\end{pmatrix}\begin{pmatrix}\delta_c\\ \delta_p\end{pmatrix} = \begin{pmatrix}\mathbf g_c\\ \mathbf g_p\end{pmatrix}$$
      $$\big(C - W V^{-1} W^\top\big)\,\delta_c = \mathbf g_c - W V^{-1}\mathbf g_p,\qquad \delta_p = V^{-1}\big(\mathbf g_p - W^\top\delta_c\big)$$
      <div class="parts">
        <span>$C$</span><span>camera × camera block (and focal); small: 355 × 355</span>
        <span>$V$</span><span>point × point block: <b>block-diagonal</b>, one 3×3 per point (a residual involves only one point), so $V^{-1}$ is just many 3×3 inverses</span>
        <span>$W$</span><span>camera × point coupling: non-zero where camera $i$ sees point $j$</span>
        <span>step 1</span><span>eliminate the points: solve the small reduced system for the cameras</span>
        <span>step 2</span><span>back-substitute: each point's update separately</span>
      </div></div>
      <p class="mini">Example with one camera number and one point number: $C=5$, $W=2$, $V=1$, $g_c=3$, $g_p=1$. Reduced: $(5-2\cdot1\cdot2)\,\delta_c = 3-2\cdot1\cdot1$, so $\delta_c = 1$; then $\delta_p = (1-2\cdot1)/1 = -1$. Check: $5\cdot1 + 2\cdot(-1) = 3$ ✓, $2\cdot1+1\cdot(-1) = 1$ ✓.</p>
    `);
    {
      const fig = L.figure(root, "<b>Where the zeros are.</b> Left: the full matrix $J^\\top J$ of a small bundle adjustment (each track seen by a few consecutive frames). Blue: camera blocks $C$; green: point blocks $V$ (only on the diagonal); orange: coupling $W$. Right: the reduced camera system after eliminating the points.");
      const c = L.canvas(fig.el, { aspect: 0.52, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const upd = () => { c.redraw(); info(); };
      const sc = L.slider(ctl, { label: "moving frames", min: 2, max: 6, step: 1, value: 3, oninput: upd });
      const sp = L.slider(ctl, { label: "points", min: 4, max: 24, step: 1, value: 10, oninput: upd });
      const vis = (nc, np) => Array.from({ length: np }, (_, j) => { const a = Math.floor((j * nc) / np); return [a, Math.min(nc - 1, a + 1)]; });
      c.draw = (ctx) => {
        const t = L.theme(), nc = sc.value, np = sp.value, N = 6 * nc + 3 * np, M = 6 * nc;
        const size = Math.min(c.w * 0.6 - 12, c.h - 24), cell = size / N;
        const ox = 4, oy = 4;
        const block = (r0, c0, h, w, col) => { ctx.fillStyle = col; ctx.fillRect(ox + c0 * cell, oy + r0 * cell, w * cell, h * cell); };
        ctx.save();
        ctx.fillStyle = t.panel2; ctx.fillRect(ox, oy, size, size);
        const V = vis(nc, np);
        for (let i = 0; i < nc; i++) block(6 * i, 6 * i, 6, 6, t.accent);
        V.forEach(([a, b], j) => {
          block(M + 3 * j, M + 3 * j, 3, 3, t.good);
          for (const i of new Set([a, b])) { block(6 * i, M + 3 * j, 6, 3, t.accent2); block(M + 3 * j, 6 * i, 3, 6, t.accent2); }
          if (a !== b) { block(6 * a, 6 * b, 6, 6, t.accent); block(6 * b, 6 * a, 6, 6, t.accent); }
        });
        ctx.strokeStyle = t.line; ctx.strokeRect(ox, oy, size, size);
        // reduced system
        const rs = Math.min(c.w - ox - size - 24, size * 0.7), rc = rs / M, rx = ox + size + 20, ry = oy + (size - rs) / 2;
        ctx.fillStyle = t.panel2; ctx.fillRect(rx, ry, rs, rs);
        ctx.fillStyle = t.accent;
        for (let i = 0; i < nc; i++) for (let k = 0; k < nc; k++) {
          const share = i === k || V.some(([a, b]) => (a === i && b === k) || (a === k && b === i));
          if (share) ctx.fillRect(rx + 6 * k * rc, ry + 6 * i * rc, 6 * rc, 6 * rc);
        }
        ctx.strokeStyle = t.line; ctx.strokeRect(rx, ry, rs, rs);
        ctx.restore();
        L.draw.text(ctx, `${N}×${N}`, ox + size / 2, oy + size + 15, t.muted, { size: 11, align: "center" });
        L.draw.text(ctx, `${M}×${M}`, rx + rs / 2, ry + rs + 14, t.muted, { size: 11, align: "center" });
      };
      const info = () => {
        const nc = sc.value, np = sp.value;
        out.html = `full system ${6 * nc + 3 * np} unknowns → reduced ${6 * nc} (cameras only) + ${np} separate 3×3 solves` +
          `<br>real bootstrap: 59 frames + focal, ~2000 points: ${6 * 59 + 1 + 6000} → <b>${6 * 59 + 1}</b>`;
      };
      info();
    }
    H(String.raw`
      <pre><code>bundle_adjust(poses, points, f):             // LM, up to 40 iterations
  μ = 1e-3
  loop:
      build C, W, V, g_c, g_p from all observations (Huber weights)
      for up to 6 attempts:
          add μ·(1 + diagonal) to the diagonals of C and each V_j
          S = C − Σ_j W_j V_j⁻¹ W_jᵀ;   b = g_c − Σ_j W_j V_j⁻¹ g_p_j
          δc = cholesky_solve(S, b)
          δp_j = V_j⁻¹ (g_p_j − W_jᵀ δc)   for every point
          try: poses ← exp(δ)·pose, points += δp, f += δf
          if cost dropped: accept, μ /= 3, next iteration
          else: μ *= 5
      stop if no attempt helped or the cost improved by < 1e-7 (relative)

bundle_adjust(…)
drop observations with reprojection error ≥ 3 px
bundle_adjust(…)                             // final RMS reported</code></pre>
    `);

    // ================================================================ handover
    H(String.raw`
      <h3>16 · Handing over to DTAM</h3>
      <table class="plain">
        <tr><th>outcome</th><th>what <code>slam.rs</code> does</th></tr>
        <tr><td>failure (no good pair, or &lt; 15 known points visible in some frame)</td><td>drop the oldest 15 frames of the 60-frame window, keep collecting, retry ("keep moving the camera sideways")</td></tr>
        <tr><td>success</td><td>poses $T_{wc}$ for all window frames (world = first frame's camera, median depth 1), refined $f$, 3D points</td></tr>
        <tr><td>first keyframe</td><td>the <b>middle</b> frame of the window; its inverse-depth search range comes from the bootstrap points (2nd–98th percentile of $\xi$, widened ×0.5 / ×1.6), and all window frames go into its cost volume (chapter 8)</td></tr>
      </table>
      <div class="key">The bootstrap only runs once. After it, poses come from dense whole-image alignment against the model (chapter 11), and corners are never used again.</div>
      <h3>Summary</h3>
      <table class="plain">
        <tr><th>step</th><th>key equation</th><th>values here</th></tr>
        <tr><td>epipolar constraint</td><td>$\hat{\mathbf x}_2^\top[\mathbf t]_\times R\,\hat{\mathbf x}_1=0$</td><td></td></tr>
        <tr><td>fundamental matrix</td><td>$F=K^{-\top}EK^{-1}$</td><td>8-point, Hartley, rank 2</td></tr>
        <tr><td>robust fit</td><td>$N=\ln(1-p)/\ln(1-w^s)$</td><td>800 / 1000 iterations, Sampson &lt; 1 px²</td></tr>
        <tr><td>degeneracy</td><td>$n_H \ge \rho\,n_F$</td><td>ρ = 0.9 / 0.85</td></tr>
        <tr><td>focal</td><td>$\min_f (\sigma_1-\sigma_2)/\sigma_1$</td><td>FOV 10°–150°, 800 log steps</td></tr>
        <tr><td>motion</td><td>4 candidates, cheirality</td><td>≥ 50 points in front</td></tr>
        <tr><td>points</td><td>DLT triangulation</td><td>parallax ≥ 1.5°, error &lt; 3 px</td></tr>
        <tr><td>more frames</td><td>PnP, Gauss–Newton</td><td>15 iterations, Huber 1.5 px</td></tr>
        <tr><td>refine</td><td>BA, LM + Schur</td><td>40 iterations, drop &gt; 3 px</td></tr>
        <tr><td>gauge</td><td>$s = 1/\mathrm{median}\,z$</td><td>frame 0 = world</td></tr>
      </table>
    `);

    // ================================================================ quiz
    const R2 = (v) => Math.round(v * 100) / 100;
    /** Number for TeX, parenthesised when negative (so 2·(−3), not 2·−3). */
    const pn = (v, d = 4) => (v < 0 ? `(${L.fmt(v, d)})` : L.fmt(v, d));
    L.quiz(root, "bootstrap", [
      { id: "coplanar", type: "mc",
        q: String.raw`Geometrically, what does $\hat{\mathbf x}_2^\top[\mathbf t]_\times R\,\hat{\mathbf x}_1 = 0$ state?`,
        choices: [
          "The ray from camera 2, the baseline and the ray from camera 1 lie in one plane",
          "The two rays are parallel",
          "The two pixels have the same brightness",
          "The point is at depth 1 in both cameras",
        ],
        answer: 0,
        explain: String.raw`$[\mathbf t]_\times R\hat{\mathbf x}_1 = \mathbf t\times R\hat{\mathbf x}_1$ is the normal of the plane spanned by the baseline and ray 1; a zero dot product with ray 2 means ray 2 lies in that plane (the epipolar plane). Depth and brightness don't appear.` },
      { id: "epiline", type: "num",
        gen: (r) => {
          const t = [r.nz(2), r.int(-1, 1), r.int(-1, 1)];
          const a = r.int(-4, 4) / 10, b = r.int(-4, 4) / 10;
          const l = cross(t, [a, b, 1]);
          return {
            q: String.raw`Camera 2 has no rotation relative to camera 1 ($R = I$) and $\mathbf t = (${t.join(", ")})$. For $\hat{\mathbf x}_1 = (${a}, ${b}, 1)$, compute the epipolar line $\mathbf l_2 = E\hat{\mathbf x}_1 = [\mathbf t]_\times\hat{\mathbf x}_1$ (no rescaling).`,
            answer: l, labels: ["$a$", "$b$", "$c$"], tol: 1e-4,
            explain: String.raw`With $R = I$, $E\hat{\mathbf x}_1 = \mathbf t\times\hat{\mathbf x}_1 = (t_y\cdot1 - t_z\hat y,\ t_z\hat x - t_x\cdot 1,\ t_x\hat y - t_y\hat x) = (${L.fmt(l[0], 4)}, ${L.fmt(l[1], 4)}, ${L.fmt(l[2], 4)})$. Any match $\hat{\mathbf x}_2$ must satisfy $a\hat x_2 + b\hat y_2 + c = 0$.`,
          };
        } },
      { id: "linedist", type: "num",
        gen: (r) => {
          const a = r.nz(5), b = r.nz(5), c = r.int(-30, 30), u = r.int(0, 20), v = r.int(0, 20);
          const d = Math.abs(a * u + b * v + c) / Math.hypot(a, b);
          return {
            q: String.raw`The epipolar line in image 2 is $${a}u + ${b}v + ${c} = 0$ (pixels). How far (in px, to 2 decimals) is the tracked point $\mathbf u_2 = (${u}, ${v})$ from it?`.replace(/\+ -/g, "- "),
            answer: d, tol: 0.011,
            explain: String.raw`$d = |a u + b v + c|/\sqrt{a^2+b^2} = |${a * u} + ${pn(b * v)} + ${pn(c)}|/\sqrt{${a * a + b * b}} = ${Math.abs(a * u + b * v + c)}/${L.fmt(Math.hypot(a, b), 3)} = ${L.fmt(d, 3)}$ px.`,
          };
        } },
      { id: "eightrow", type: "num",
        gen: (r) => {
          const x1 = r.int(-9, 9) / 10, y1 = r.nz(9) / 10, x2 = r.int(-9, 9) / 10, y2 = r.nz(9) / 10;
          return {
            q: String.raw`In the 8-point algorithm, the match $(x_1, y_1) = (${x1}, ${y1})$ ↔ $(x_2, y_2) = (${x2}, ${y2})$ gives one row of $A$. What are the coefficients that multiply $F_{12}$, $F_{23}$ and $F_{31}$?`,
            answer: [x2 * y1, y2, x1], labels: ["$F_{12}$", "$F_{23}$", "$F_{31}$"], tol: 1e-4,
            explain: String.raw`The coefficient of $F_{ij}$ is (entry $i$ of $\dot{\mathbf u}_2$)·(entry $j$ of $\dot{\mathbf u}_1$), with $\dot{\mathbf u} = (x, y, 1)$. $F_{12}$: $x_2 y_1 = ${pn(x2)}\cdot${pn(y1)} = ${L.fmt(x2 * y1, 4)}$. $F_{23}$: $y_2\cdot1 = ${y2}$. $F_{31}$: $1\cdot x_1 = ${x1}$.`,
          };
        } },
      { id: "hartley", type: "num",
        gen: (r) => {
          const mx = r.int(100, 500), my = r.int(100, 400), a = r.int(20, 120), b = r.int(20, 120);
          const s = Math.SQRT2 / ((a + b) / 2);
          return {
            q: String.raw`Four points in one image: $(${mx + a}, ${my})$, $(${mx - a}, ${my})$, $(${mx}, ${my + b})$, $(${mx}, ${my - b})$. Hartley normalisation moves the centroid to 0 and scales by $s = \sqrt2/\text{(mean distance to the centroid)}$. Find $s$ (to 4 significant digits) and the normalised $x$-coordinate of the first point.`,
            answer: [s, s * a], labels: ["$s$", "$\\tilde x_1$"], rtol: 0.005,
            explain: String.raw`Centroid $(${mx}, ${my})$. Distances: $${a}, ${a}, ${b}, ${b}$, mean $${L.fmt((a + b) / 2)}$. $s = \sqrt2/${L.fmt((a + b) / 2)} = ${L.fmt(s, 5)}$. First point: $s\,(${mx + a} - ${mx}) = ${L.fmt(s, 5)}\cdot${a} = ${L.fmt(s * a, 4)}$ (≈ √2-sized, as intended).`,
          };
        } },
      { id: "rank2", type: "multi",
        q: "Why does the 8-point algorithm force the fitted $F$ to rank 2 (set its smallest stretch factor to 0)? Select all that apply.",
        choices: [
          "Every true fundamental matrix has rank 2, because the epipole direction is squashed to zero",
          "Without it, the epipolar lines of different points don't all pass through one epipole",
          "It gives $F$ unit length",
          "It removes the outliers among the matches",
        ],
        answer: [0, 1],
        explain: "$E = [\\mathbf t]_\\times R$ squashes one direction, so $E$ and $F = K^{-\\top}EK^{-1}$ have rank 2; the squashed direction is the epipole, where all lines meet. Noise breaks that; zeroing $\\sigma_3$ restores it with the smallest change. Scale and outliers are handled elsewhere (the $|\\mathbf f| = 1$ constraint, RANSAC)." },
      { id: "sampson", type: "num",
        gen: (r) => {
          const p = r.nz(3), q = r.nz(3), a = r.nz(3), b = r.nz(3), w = r.int(-5, 5);
          const u1 = [r.int(-3, 3), r.int(-3, 3)], u2 = [r.int(-3, 3), r.int(-3, 3)];
          const e = p * u2[0] + q * u2[1] + a * u1[0] + b * u1[1] + w;
          const den = p * p + q * q + a * a + b * b;
          const ds = (e * e) / den;
          return {
            q: String.raw`$F = \begin{pmatrix}0&0&${p}\\0&0&${q}\\${a}&${b}&${w}\end{pmatrix}$, $\mathbf u_1 = (${u1[0]}, ${u1[1]})$, $\mathbf u_2 = (${u2[0]}, ${u2[1]})$. Compute the squared Sampson distance $d_S^2$ (px², to 2 decimals).`,
            answer: ds, tol: 0.011,
            explain: String.raw`$F\dot{\mathbf u}_1 = (${p}, ${q}, ${a * u1[0] + b * u1[1] + w})$, so $e = \dot{\mathbf u}_2^\top F\dot{\mathbf u}_1 = ${pn(p)}\cdot${pn(u2[0])} + ${pn(q)}\cdot${pn(u2[1])} + ${pn(a * u1[0] + b * u1[1] + w)} = ${e}$. $F^\top\dot{\mathbf u}_2 = (${a}, ${b}, \dots)$. $d_S^2 = e^2/(${pn(p)}^2+${pn(q)}^2+${pn(a)}^2+${pn(b)}^2) = ${e * e}/${den} = ${L.fmt(ds, 4)}$.`,
          };
        } },
      { id: "hartleywhy", type: "mc",
        q: "Why does the 8-point algorithm normalise the pixel coordinates first?",
        choices: [
          "Raw pixels put entries like $x_2x_1 \\approx 10^5$ next to $1$ in each row, so small noise badly distorts the least-stretched direction",
          "It removes lens distortion",
          "It converts pixels to metres, which the algorithm requires",
          "It guarantees that $F$ has rank 2",
        ],
        answer: 0,
        explain: "Normalisation (centroid 0, mean distance √2) makes all columns of $A$ comparable in size, which makes the null-space solution far less sensitive to noise (see the 8-point lab: ~0.7 px vs &gt;10 px error). It's undone afterwards by $F = \\mathcal T_2^\\top\\tilde F\\mathcal T_1$." },
      { id: "ransacN", type: "num",
        gen: (r) => {
          const w = r.pick([0.5, 0.6, 0.7, 0.8, 0.9]), s = r.pick([2, 4, 7, 8]), p = r.pick([0.95, 0.99]);
          const N = Math.ceil(Math.log(1 - p) / Math.log(1 - w ** s));
          return {
            q: String.raw`A RANSAC model needs samples of $s = ${s}$ matches. ${Math.round(w * 100)}% of the matches are inliers. How many iterations $N$ guarantee a ${p * 100}% chance of drawing at least one all-inlier sample? (Round up to an integer.)`,
            answer: N, tol: 0.5,
            explain: String.raw`$w^s = ${w}^{${s}} = ${L.fmt(w ** s, 5)}$. $N = \lceil\ln(1-${p})/\ln(1-${L.fmt(w ** s, 5)})\rceil = \lceil ${L.fmt(Math.log(1 - p), 4)}/${L.fmt(Math.log(1 - w ** s), 5)}\rceil = ${N}$.`,
          };
        } },
      { id: "degenerate", type: "multi",
        q: "In which situations does a homography explain the matches about as well as $F$, so the pair is rejected? Select all that apply.",
        choices: [
          "The camera only rotated between the two frames",
          "All tracked points lie on one flat wall",
          "The camera moved sideways past objects at many different depths",
          "The camera moved forward through a room full of furniture",
        ],
        answer: [0, 1],
        explain: "Pure rotation and planar scenes both give $\\dot{\\mathbf u}_2 \\propto H\\dot{\\mathbf u}_1$. Then many different $F$ fit, so $F$ (and the motion from it) is unreliable. Translation with depth variation produces parallax that no single $H$ can explain." },
      { id: "sv2", type: "num",
        gen: (r) => {
          let a, b, c, d;
          do { a = r.int(-4, 4); b = r.int(-4, 4); c = r.int(-4, 4); d = r.int(-4, 4); } while (a * d - b * c === 0);
          const S = a * a + b * b + c * c + d * d, D = Math.abs(a * d - b * c);
          const p = Math.sqrt(S + 2 * D), m = Math.sqrt(S - 2 * D);
          return {
            q: String.raw`Find the stretch factors (singular values) $\sigma_1 \ge \sigma_2$ of $A = \begin{pmatrix}${a}&${b}\\${c}&${d}\end{pmatrix}$, to 2 decimals.`,
            answer: [(p + m) / 2, (p - m) / 2], labels: ["$\\sigma_1$", "$\\sigma_2$"], tol: 0.011,
            explain: String.raw`Sum of squares $S = ${S}$, $|\det A| = D = ${D}$. $\sigma_1+\sigma_2 = \sqrt{S+2D} = \sqrt{${S + 2 * D}} = ${L.fmt(p, 4)}$, $\sigma_1-\sigma_2 = \sqrt{S-2D} = \sqrt{${S - 2 * D}} = ${L.fmt(m, 4)}$. So $\sigma_1 = ${L.fmt((p + m) / 2, 4)}$, $\sigma_2 = ${L.fmt((p - m) / 2, 4)}$.`,
          };
        } },
      { id: "kfk", type: "num",
        gen: (r) => {
          const g = r.pick([400, 500, 600, 800, 1000]);
          const F12 = r.nz(9) / 1e6, F13 = r.nz(9) / 1e3;
          return {
            q: String.raw`Self-calibration tries $f = ${g}$ with the principal point at the origin, so $K_f = \mathrm{diag}(${g}, ${g}, 1)$. For $F_{12} = ${L.fmt(F12 * 1e6)}\times10^{-6}$ and $F_{13} = ${L.fmt(F13 * 1e3)}\times10^{-3}$, what are the entries $E_{12}$ and $E_{13}$ of $E = K_f^\top F K_f$?`,
            answer: [g * g * F12, g * F13], labels: ["$E_{12}$", "$E_{13}$"], rtol: 0.001,
            explain: String.raw`With diagonal $K_f = \mathrm{diag}(k_1,k_2,k_3) = (${g}, ${g}, 1)$, $E_{ij} = k_i k_j F_{ij}$. $E_{12} = ${g}^2\cdot ${pn(F12 * 1e6)}\times10^{-6} = ${L.fmt(g * g * F12, 4)}$; $E_{13} = ${g}\cdot1\cdot${pn(F13 * 1e3)}\times10^{-3} = ${L.fmt(g * F13, 4)}$. The scan repeats this for every $f$ and checks whether $E$'s two stretch factors match.`,
          };
        } },
      { id: "fovgrid", type: "num",
        gen: (r) => {
          const w = r.pick([640, 960, 1280, 1920]);
          const lo = (w / 2) / Math.tan(75 * D2R), hi = (w / 2) / Math.tan(5 * D2R);
          return {
            q: String.raw`The focal scan covers horizontal fields of view from 150° down to 10° for an image $${w}$ px wide, on a log-spaced grid. Find $f_{lo}$ (150°), $f_{hi}$ (10°) and the grid's middle value.`,
            answer: [lo, hi, Math.sqrt(lo * hi)], labels: ["$f_{lo}$", "$f_{hi}$", "middle"], rtol: 0.01,
            explain: String.raw`$f = (w/2)/\tan(\text{fov}/2)$. $f_{lo} = ${w / 2}/\tan 75° = ${L.fmt(lo, 2)}$, $f_{hi} = ${w / 2}/\tan 5° = ${L.fmt(hi, 1)}$. On a log grid the middle ($i/(n-1) = 1/2$) is $f_{lo}(f_{hi}/f_{lo})^{1/2} = \sqrt{f_{lo}f_{hi}} = ${L.fmt(Math.sqrt(lo * hi), 1)}$.`,
          };
        } },
      { id: "puretrans", type: "mc",
        q: "Between two frames the camera only translated (no rotation). What does the focal-length cost $c(f)$ look like for that pair?",
        choices: [
          "Flat at 0: every $f$ gives two equal stretch factors, so the pair says nothing about $f$",
          "A sharp minimum at the true $f$",
          "A minimum at the smallest $f$ scanned",
          "It is undefined because $F = 0$",
        ],
        answer: 0,
        explain: "Then $E = [\\mathbf t]_\\times$ is antisymmetric, and $K_f^\\top F K_f = (K^{-1}K_f)^\\top E (K^{-1}K_f)$ stays antisymmetric for every $f$. Antisymmetric 3×3 matrices always have stretch factors $(\\sigma,\\sigma,0)$. $F$ is not zero (only pure <i>rotation</i> gives $E = 0$)." },
      { id: "cheirality", type: "num",
        gen: (r) => {
          const th = r.pick([-20, -10, 10, 20, 30]), tx = r.int(-10, 10) / 10, tz = r.int(-20, 10) / 10;
          const x = r.int(-10, 10) / 10, y = r.int(-5, 5) / 10, z = r.int(15, 50) / 10;
          const z2 = -Math.sin(th * D2R) * x + Math.cos(th * D2R) * z + tz;
          return {
            q: String.raw`A candidate motion is $R = R_y(${th}°) = \begin{pmatrix}\cos\theta&0&\sin\theta\\0&1&0\\-\sin\theta&0&\cos\theta\end{pmatrix}$, $\mathbf t = (${tx}, 0, ${tz})$. A match triangulates to $\mathbf x_1 = (${x}, ${y}, ${z})$ in camera 1. What is its depth $z_2$ in camera 2 (to 2 decimals)? (If negative, this candidate fails the cheirality test for this point.)`,
            answer: z2, tol: 0.011,
            explain: String.raw`$z_2 = (R\mathbf x_1 + \mathbf t)_z = -\sin\theta\,x + \cos\theta\,z + t_z = -\sin(${th}°)\cdot${pn(x)} + \cos(${th}°)\cdot${z} + ${pn(tz)} = ${L.fmt(z2, 3)}$. ${z2 > 0 ? "Positive: in front of camera 2 (and $z_1 = " + z + " > 0$)." : "Negative: behind camera 2, so this point votes against the candidate."}`,
          };
        } },
      { id: "fourcands", type: "mc",
        q: "Decomposing $E$ yields four $(R, \\mathbf t)$ candidates. How does this implementation pick one?",
        choices: [
          "Triangulate the inlier matches with each candidate and keep the one with the most points in front of both cameras",
          "Keep the candidate with the smallest rotation angle",
          "Keep the candidate whose $\\mathbf t$ has a positive $z$ component",
          "Any of them works: they give the same 3D points",
        ],
        answer: 0,
        explain: "All four satisfy the epipolar constraint equally; only one places the scene in front of both cameras (cheirality). The others put points behind a camera or turn camera 2 upside down. The winner must also have ≥ 50 good points." },
      { id: "stereo", type: "num",
        gen: (r) => {
          const b = r.pick([0.1, 0.2, 0.25, 0.5]), Z = r.pick([2, 2.5, 4, 5]), X = r.int(-10, 10) / 10;
          const x1 = X / Z, x2 = (X - b) / Z;
          return {
            q: String.raw`Camera 2 sits at $(${b}, 0, 0)$ in camera 1's frame, with no rotation. A point is observed at normalised $x_1 = ${L.fmt(x1, 4)}$ in camera 1 and $x_2 = ${L.fmt(x2, 4)}$ in camera 2. Triangulate: find its $X$ and $Z$.`,
            answer: [X, Z], labels: ["$X$", "$Z$"], tol: 0.01,
            explain: String.raw`$x_1 = X/Z$ and $x_2 = (X - b)/Z$, so $x_1 - x_2 = b/Z$: $Z = ${b}/(${L.fmt(x1, 4)} - ${pn(x2)}) = ${L.fmt(Z)}$, $X = x_1 Z = ${L.fmt(X)}$. (The DLT rows $x\,\mathbf p_3 - \mathbf p_1$ give exactly these equations.)`,
          };
        } },
      { id: "parallax", type: "num",
        gen: (r) => {
          const b = r.pick([0.05, 0.1, 0.2, 0.3]), px = r.int(-10, 10) / 10, pz = r.pick([1, 2, 3, 4, 6]);
          const d1 = [px, pz], d2 = [px - b, pz];
          const ang = Math.acos(la.dot(d1, d2) / (Math.hypot(...d1) * Math.hypot(...d2))) / D2R;
          return {
            q: String.raw`Camera centres at $(0,0,0)$ and $(${b}, 0, 0)$; a point at $(${px}, 0, ${pz})$. What is the parallax angle between the two rays, in degrees (to 2 decimals)? Would this implementation triangulate it (threshold 1.5°)?`,
            answer: ang, tol: 0.011,
            explain: String.raw`Ray directions $(${px}, 0, ${pz})$ and $(${L.fmt(px - b)}, 0, ${pz})$. $\cos\alpha = ${L.fmt(la.dot(d1, d2), 4)}/(${L.fmt(Math.hypot(...d1), 4)}\cdot${L.fmt(Math.hypot(...d2), 4)})$, $\alpha = ${L.fmt(ang, 3)}°$ (equivalently $|\arctan(${px}/${pz}) - \arctan(${pn(px - b)}/${pz})|$). ${ang >= 1.5 ? "≥ 1.5°: yes." : "< 1.5°: not yet; it waits for more baseline."}`,
          };
        } },
      { id: "scale", type: "mc",
        q: "Which of these can two views of a static scene (matches only) <b>not</b> determine?",
        choices: [
          "The overall scale (how many metres one unit is)",
          "The direction of the camera's translation",
          "The rotation between the views",
          "The ratio of two points' depths",
        ],
        answer: 0,
        explain: "Scaling points and translations together leaves every image unchanged ($\\pi$ divides by depth). Direction of travel, rotation and depth <i>ratios</i> are all fixed. This implementation sets the scale so the median depth is 1." },
      { id: "median", type: "num",
        gen: (r) => {
          const zs = Array.from({ length: 5 }, () => r.int(12, 60) / 10);
          const m = median(zs), s = 1 / m, base = r.int(2, 9) / 10;
          return {
            q: String.raw`After bundle adjustment the point depths (in frame 0) are $${zs.join(",\\ ")}$. The code rescales so the median depth is 1. What is the factor $s$, and what does a camera translation of length $${base}$ become?`,
            answer: [s, base * s], labels: ["$s$", "new length"], rtol: 0.005,
            explain: String.raw`Sorted: $${zs.slice().sort((a, b) => a - b).join(",\\ ")}$, median $${m}$. $s = 1/${m} = ${L.fmt(s, 4)}$. Translations scale too: $${base}\cdot${L.fmt(s, 4)} = ${L.fmt(base * s, 4)}$. Rotations don't change.`,
          };
        } },
      { id: "pnpjac", type: "num",
        gen: (r) => {
          const f = r.pick([300, 400, 500, 600]), x = r.int(-8, 8) / 10, y = r.int(-8, 8) / 10, z = r.pick([1, 1.5, 2, 2.5, 4]);
          return {
            q: String.raw`PnP linearises the projection. For $f = ${f}$ and a point $\mathbf x_c = (${x}, ${y}, ${z})$ in camera coordinates, give the entries $(1,1)$, $(1,3)$ and $(2,3)$ of $\partial\pi(K\mathbf x_c)/\partial\mathbf x_c$.`,
            answer: [f / z, -f * x / (z * z), -f * y / (z * z)], labels: ["$J_{11}$", "$J_{13}$", "$J_{23}$"], rtol: 0.005, tol: 0.01,
            explain: String.raw`$u = f x/z + c_x$: $\partial u/\partial x = f/z = ${L.fmt(f / z, 3)}$, $\partial u/\partial z = -f x/z^2 = -${f}\cdot${pn(x)}/${z * z} = ${L.fmt(-f * x / (z * z), 3)}$. $v = f y/z + c_y$: $\partial v/\partial z = -f y/z^2 = ${L.fmt(-f * y / (z * z), 3)}$. Multiply by $[I\,|\,-[\mathbf x_c]_\times]$ to get the 2×6 Jacobian in $\psi$.`,
          };
        } },
      { id: "reproj", type: "num",
        gen: (r) => {
          const f = r.pick([400, 500]), cx = 320, cy = 240;
          const x = r.int(-5, 5) / 10, y = r.int(-5, 5) / 10, z = r.pick([2, 2.5, 4, 5]);
          const pu = f * x / z + cx, pv = f * y / z + cy;
          const du = r.int(-30, 30) / 10, dv = r.int(-30, 30) / 10;
          const u = R2(pu + du), v = R2(pv + dv);
          const ru = u - pu, rv = v - pv, n = Math.hypot(ru, rv), w = n <= 1.5 ? 1 : 1.5 / n;
          return {
            q: String.raw`$f = ${f}$, $(c_x, c_y) = (${cx}, ${cy})$. A known point is at $\mathbf x_c = (${x}, ${y}, ${z})$ in the current pose estimate and was tracked at $\mathbf u = (${u}, ${v})$. Compute the residual $\mathbf r = \mathbf u - \pi(K\mathbf x_c)$ and its Huber weight $w$ (threshold 1.5 px).`,
            answer: [ru, rv, w], labels: ["$r_u$", "$r_v$", "$w$"], tol: 0.011,
            explain: String.raw`$\pi(K\mathbf x_c) = (${f}\cdot${pn(x)}/${z} + ${cx},\ ${f}\cdot${pn(y)}/${z} + ${cy}) = (${L.fmt(pu)}, ${L.fmt(pv)})$. $\mathbf r = (${L.fmt(ru, 3)}, ${L.fmt(rv, 3)})$, $|\mathbf r| = ${L.fmt(n, 3)}$. ${n <= 1.5 ? "$\\le 1.5$, so $w = 1$." : "$> 1.5$, so $w = 1.5/" + L.fmt(n, 3) + " = " + L.fmt(w, 3) + "$."}`,
          };
        } },
      { id: "schur", type: "num",
        gen: (r) => {
          let c, w, v;
          do { c = r.int(3, 9); w = r.nz(3); v = r.int(1, 4); } while (c * v - w * w <= 0);
          const gc = r.int(-6, 6), gp = r.int(-6, 6);
          const dc = (gc - (w * gp) / v) / (c - (w * w) / v), dp = (gp - w * dc) / v;
          return {
            q: String.raw`A toy bundle adjustment has one camera unknown and one point unknown: $\begin{pmatrix}${c}&${w}\\${w}&${v}\end{pmatrix}\begin{pmatrix}\delta_c\\\delta_p\end{pmatrix} = \begin{pmatrix}${gc}\\${gp}\end{pmatrix}$. Solve it with the Schur complement (eliminate the point first). Give $\delta_c$ and $\delta_p$ to 2 decimals.`,
            answer: [dc, dp], labels: ["$\\delta_c$", "$\\delta_p$"], tol: 0.011,
            explain: String.raw`Reduced: $(C - W V^{-1} W)\,\delta_c = g_c - W V^{-1} g_p$: $(${c} - ${w * w}/${v})\,\delta_c = ${gc} - ${pn(w)}\cdot${pn(gp)}/${v}$, i.e. $${L.fmt(c - (w * w) / v, 4)}\,\delta_c = ${L.fmt(gc - (w * gp) / v, 4)}$, $\delta_c = ${L.fmt(dc, 4)}$. Back-substitute: $\delta_p = (g_p - W\delta_c)/V = (${gp} - ${pn(w)}\cdot${pn(dc)})/${v} = ${L.fmt(dp, 4)}$.`,
          };
        } },
      { id: "basize", type: "num",
        gen: (r) => {
          const n = r.pick([20, 30, 40, 60]), P = r.int(5, 30) * 100;
          const nc = 6 * (n - 1) + 1;
          return {
            q: String.raw`Bundle adjustment over $${n}$ frames (frame 0 held fixed, focal length refined) and $${P}$ points. How many unknowns does the reduced camera system have, and how many does the full system have?`,
            answer: [nc, nc + 3 * P], labels: ["reduced", "full"], tol: 0.5,
            explain: String.raw`Each moving frame has 6 pose unknowns: $6\cdot${n - 1} = ${6 * (n - 1)}$, plus 1 for $f$: $${nc}$. Each point adds 3: full $= ${nc} + 3\cdot${P} = ${nc + 3 * P}$. After eliminating points, only the $${nc}\times${nc}$ system needs a dense solve.`,
          };
        } },
      { id: "schurwhy", type: "mc",
        q: "Why does bundle adjustment eliminate the points first (Schur complement) instead of solving the whole system directly?",
        choices: [
          "Each residual involves one point, so the point block is block-diagonal with 3×3 blocks: cheap to invert, leaving a small camera-only system",
          "Points are less accurate than cameras, so they should be solved last",
          "Levenberg–Marquardt only works on camera parameters",
          "It removes outlier observations",
        ],
        answer: 0,
        explain: "$V$ is block-diagonal, so $V^{-1}$ costs one 3×3 inverse per point. The reduced system has size 6·(frames−1)+1 (355 here) instead of thousands; afterwards each point's update is computed separately." },
    ]);
  },
});
