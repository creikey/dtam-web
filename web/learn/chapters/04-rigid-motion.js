// Chapter 4: rotations, poses, pixel transfer between cameras, skew matrix, exp map, twists and SE(3) generators.
DTAM.chapter({
  id: "motion",
  order: 4,
  title: "Moving cameras: rotations, poses and the exponential map",
  subtitle: "How a point looks from another camera, and how to nudge a pose",
  minutes: 60,
  render(root, L) {
    const { la } = L;
    const deg = Math.PI / 180;
    const f2 = (x) => (Math.abs(x) < 5e-3 ? 0 : x).toFixed(2), f1 = (x) => (+x).toFixed(1), f3 = (x) => (Math.abs(x) < 5e-4 ? 0 : x).toFixed(3);
    const v3 = (p, f = f2) => `(${f(p[0])}, ${f(p[1])}, ${f(p[2])})`;

    // ------------------------------------------------------------ rigid-transform helpers {R (rows), t}
    const apply = (T, x) => la.add(la.matVec(T.R, x), T.t);
    const inv = (T) => { const Rt = la.T(T.R); return { R: Rt, t: la.scale(la.matVec(Rt, T.t), -1) }; };
    const compose = (A, B) => ({ R: la.matMul(A.R, B.R), t: la.add(la.matVec(A.R, B.t), A.t) });
    // Exponential map of a twist psi = (v, w), translation first (as geom.rs Se3::exp).
    function expSE3(psi) {
      const v = psi.slice(0, 3), w = psi.slice(3, 6);
      const th = Math.hypot(...w), W = la.skew(w), W2 = la.matMul(W, W), I = la.eye(3);
      let A, B, C;
      if (th < 1e-8) { A = 1 - th * th / 6; B = 0.5 - th * th / 24; C = 1 / 6 - th * th / 120; }
      else { A = Math.sin(th) / th; B = (1 - Math.cos(th)) / (th * th); C = (th - Math.sin(th)) / (th * th * th); }
      const R = I.map((row, i) => row.map((x, j) => x + A * W[i][j] + B * W2[i][j]));
      const V = I.map((row, i) => row.map((x, j) => x + B * W[i][j] + C * W2[i][j]));
      return { R, t: la.matVec(V, v) };
    }

    // ------------------------------------------------------------ small 3D viewer (drag to orbit)
    // Axes as in the camera: x right, y DOWN, z forward.
    function orbit(c, o = {}) {
      const v = { yaw: o.yaw ?? -0.75, pitch: o.pitch ?? 0.38, dist: o.dist ?? 8, target: o.target ?? [0, 0, 2], zoom: o.zoom ?? 1 };
      v.P = (p) => {
        const x = p[0] - v.target[0], y = p[1] - v.target[1], z = p[2] - v.target[2];
        const cy = Math.cos(v.yaw), sy = Math.sin(v.yaw), cp = Math.cos(v.pitch), sp = Math.sin(v.pitch);
        const x1 = cy * x - sy * z, z1 = sy * x + cy * z;
        const y2 = cp * y - sp * z1, z2 = sp * y + cp * z1;
        const dep = z2 + v.dist;
        if (dep < 0.3) return null;
        const s = (v.zoom * Math.min(c.w, c.h * 1.5) * v.dist) / 9;
        return [c.w / 2 + (s * x1) / dep, c.h / 2 + (s * y2) / dep];
      };
      v.seg = (ctx, a, b, col, w = 1.5, dash) => { const A = v.P(a), B = v.P(b); if (A && B) L.draw.line(ctx, A[0], A[1], B[0], B[1], col, w, dash); };
      v.arrow = (ctx, a, b, col, w = 2) => { const A = v.P(a), B = v.P(b); if (A && B && Math.hypot(B[0] - A[0], B[1] - A[1]) > 2) L.draw.arrow(ctx, A[0], A[1], B[0], B[1], col, w, 8); };
      v.dot = (ctx, p, r, col, stroke) => { const A = v.P(p); if (A) L.draw.dot(ctx, A[0], A[1], r, col, stroke); };
      v.label = (ctx, p, s, col, dx = 6, dy = -6) => { const A = v.P(p); if (A) L.draw.text(ctx, s, A[0] + dx, A[1] + dy, col, { size: 12 }); };
      v.poly = (ctx, pts, col, alpha) => {
        const Q = pts.map(v.P);
        if (Q.some((q) => !q)) return;
        ctx.save(); ctx.fillStyle = col; ctx.globalAlpha = alpha;
        ctx.beginPath(); ctx.moveTo(Q[0][0], Q[0][1]); for (const q of Q.slice(1)) ctx.lineTo(q[0], q[1]); ctx.closePath(); ctx.fill();
        ctx.restore();
      };
      let last = null;
      c.el.addEventListener("pointerdown", (e) => { last = { x: e.clientX, y: e.clientY }; c.el.setPointerCapture(e.pointerId); e.preventDefault(); });
      c.el.addEventListener("pointermove", (e) => {
        if (!last) return;
        v.yaw -= (e.clientX - last.x) * 0.01;
        v.pitch = Math.max(-1.45, Math.min(1.45, v.pitch + (e.clientY - last.y) * 0.01));
        last = { x: e.clientX, y: e.clientY };
        c.redraw();
      });
      const end = () => { last = null; };
      c.el.addEventListener("pointerup", end);
      c.el.addEventListener("pointercancel", end);
      c.el.style.cursor = "grab";
      return v;
    }
    // Camera frustum for pose T (camera -> world), image plane at depth D.
    function drawFrustum(ctx, v, K, D, col, T) {
      const cs = [[-0.5, -0.5], [K.w - 0.5, -0.5], [K.w - 0.5, K.h - 0.5], [-0.5, K.h - 0.5]]
        .map(([u, vv]) => apply(T, [((u - K.cx) / K.f) * D, ((vv - K.cy) / K.f) * D, D]));
      for (let i = 0; i < 4; i++) {
        v.seg(ctx, T.t, cs[i], col, 1.2);
        v.seg(ctx, cs[i], cs[(i + 1) % 4], col, i === 0 ? 3 : 1.5);
      }
      v.dot(ctx, T.t, 4, col);
    }
    function triad(ctx, v, T, t, len = 0.6, w = 2.2) {
      const o = T.t;
      const cols = [t.bad, t.good, t.accent];
      ["x", "y", "z"].forEach((n, i) => {
        const e = [0, 0, 0]; e[i] = len;
        const tip = apply(T, e);
        v.arrow(ctx, o, tip, cols[i], w);
        v.label(ctx, tip, n, cols[i], 4, -4);
      });
    }
    const mtex = (M) => String.raw`\begin{pmatrix}` + M.map((r) => r.map((x) => L.fmt(Math.abs(x) < 1e-12 ? 0 : x, 4)).join("&")).join(String.raw`\\`) + String.raw`\end{pmatrix}`;
    const vtex = (a) => `(${a.map((x) => L.fmt(Math.abs(x) < 1e-12 ? 0 : x, 4)).join(",\\ ")})`;
    const clean = (a) => a.map((x) => Math.round(x * 1e9) / 1e9 + 0);

    // ------------------------------------------------------------ 2D rotation
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>DTAM compares images taken from different places. To predict where a surface point shows up in another camera, we need to describe how cameras sit and move in space. This chapter builds that from a 2D rotation up to the 6-number "twist" that the tracker optimises.</p>
      <h3>Rotating in 2D</h3>
      <p>Rotating the point $(1,0)$ by angle $\theta$ gives $(\cos\theta, \sin\theta)$; rotating $(0,1)$ gives $(-\sin\theta, \cos\theta)$. Since rotation keeps sums and multiples intact, any point $(x,y) = x\,(1,0) + y\,(0,1)$ rotates to $x$ times the first plus $y$ times the second. That is a matrix whose <b>columns are the rotated axes</b>:</p>
      <div class="eq-card"><div class="eq-label">2D rotation</div>
      $$R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix},\qquad R(\theta)\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}x\cos\theta - y\sin\theta\\ x\sin\theta + y\cos\theta\end{pmatrix}$$
      <div class="parts">
        <span>column 1</span><span>where the $x$ axis $(1,0)$ goes</span>
        <span>column 2</span><span>where the $y$ axis $(0,1)$ goes</span>
        <span>$\theta > 0$</span><span>counter-clockwise when $y$ points up (as in this 2D widget)</span>
      </div></div>
      <p><b>Worked example.</b> $\theta = 90°$: $\cos = 0$, $\sin = 1$, so $(2, 1) \mapsto (2\cdot 0 - 1\cdot 1,\ 2\cdot 1 + 1\cdot 0) = (-1, 2)$.</p>
    `);
    {
      const fig = L.figure(root, "<b>2D rotation.</b> Drag the round handle around the circle. The coloured arrows are the columns of $R$; the shape and the point $(2,1)$ rotate with them.");
      const c = L.canvas(fig.el, { aspect: 0.72 });
      fig.add(c.el);
      const out = L.readout(fig.el); fig.add(out.el);
      let th = 35 * deg;
      const geo = () => { const sc = Math.min(c.w, c.h) / 6.4; return { sc, X: (x) => c.w / 2 + x * sc, Y: (y) => c.h / 2 - y * sc }; };
      L.drag(c, () => { const g = geo(); return [{ x: g.X(2.6 * Math.cos(th)), y: g.Y(2.6 * Math.sin(th)) }]; }, (_, p) => {
        const g = geo();
        th = Math.atan2(-(p.y - c.h / 2), p.x - c.w / 2);
      });
      const F = [[[0.4, 0.2], [0.4, 2.2], [1.6, 2.2]], [[0.4, 1.2], [1.2, 1.2]]];
      c.draw = (ctx) => {
        const t = L.theme(), g = geo(), co = Math.cos(th), si = Math.sin(th);
        const rot = ([x, y]) => [co * x - si * y, si * x + co * y];
        L.draw.line(ctx, 0, g.Y(0), c.w, g.Y(0), t.line, 1);
        L.draw.line(ctx, g.X(0), 0, g.X(0), c.h, t.line, 1);
        ctx.save(); ctx.strokeStyle = t.line; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.arc(g.X(0), g.Y(0), 2.6 * g.sc, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
        for (const s of F) L.draw.path(ctx, s.map(([x, y]) => [g.X(x), g.Y(y)]), t.faint, 3);
        for (const s of F) L.draw.path(ctx, s.map(rot).map(([x, y]) => [g.X(x), g.Y(y)]), t.accent3, 4);
        L.draw.arrow(ctx, g.X(0), g.Y(0), g.X(1.5 * co), g.Y(1.5 * si), t.bad, 3);
        L.draw.arrow(ctx, g.X(0), g.Y(0), g.X(-1.5 * si), g.Y(1.5 * co), t.good, 3);
        L.draw.text(ctx, "col 1", g.X(1.7 * co) - 14, g.Y(1.7 * si) + 4, t.bad, { size: 12, bold: true });
        L.draw.text(ctx, "col 2", g.X(-1.75 * si) - 14, g.Y(1.75 * co) + 4, t.good, { size: 12, bold: true });
        const p = [2, 1], q = rot(p);
        L.draw.dot(ctx, g.X(p[0]), g.Y(p[1]), 4, t.faint);
        L.draw.dot(ctx, g.X(q[0]), g.Y(q[1]), 6, t.accent2);
        L.draw.handle(ctx, g.X(2.6 * co), g.Y(2.6 * si), t.accent);
        out.html = `θ = <b>${f1(th / deg)}°</b> · R = [[${f2(co)}, ${f2(-si)}], [${f2(si)}, ${f2(co)}]] · R·(2, 1) = (<b>${f2(q[0])}</b>, <b>${f2(q[1])}</b>)`;
      };
    }

    // ------------------------------------------------------------ 3D rotations
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Rotating in 3D</h3>
      <p>In 3D the simplest rotations turn about one coordinate axis and leave that axis alone. Each is the 2D rotation applied to the other two coordinates:</p>
      <div class="eq-card"><div class="eq-label">Rotations about the axes</div>
      $$R_x(\alpha)=\begin{pmatrix}1&0&0\\0&\cos\alpha&-\sin\alpha\\0&\sin\alpha&\cos\alpha\end{pmatrix}\quad R_y(\alpha)=\begin{pmatrix}\cos\alpha&0&\sin\alpha\\0&1&0\\-\sin\alpha&0&\cos\alpha\end{pmatrix}\quad R_z(\alpha)=\begin{pmatrix}\cos\alpha&-\sin\alpha&0\\\sin\alpha&\cos\alpha&0\\0&0&1\end{pmatrix}$$
      <div class="parts">
        <span>$R_x$</span><span>turns $y$ toward $z$ ($x$ fixed)</span>
        <span>$R_y$</span><span>turns $z$ toward $x$ ($y$ fixed). Note the sign pattern differs: this keeps the same right-hand rule for all three</span>
        <span>$R_z$</span><span>turns $x$ toward $y$ ($z$ fixed)</span>
        <span>right-hand rule</span><span>thumb along the axis, fingers curl in the direction of positive angle</span>
      </div></div>
      <p>Every rotation matrix $R$ (any combination of these) has the same three properties. Its columns are the rotated axes: unit length and mutually perpendicular.</p>
      <div class="eq-card"><div class="eq-label">Properties of a rotation (the set SO(3))</div>
      $$R^\top R = I,\qquad \det R = +1,\qquad R^{-1} = R^\top$$
      <div class="parts">
        <span>$R^\top R = I$</span><span>columns have length 1 and are perpendicular (their dot products are the entries of $R^\top R$), so lengths and angles are preserved</span>
        <span>$\det R = +1$</span><span>no mirror image (a reflection also satisfies $R^\top R = I$ but has $\det = -1$)</span>
        <span>$R^{-1} = R^\top$</span><span>undoing a rotation is free: just transpose</span>
      </div></div>
      <p><b>Order matters.</b> $R_a R_b\,\mathbf x$ means "apply $R_b$ first, then $R_a$" (the matrix nearest $\mathbf x$ acts first). In 3D, $R_a R_b \ne R_b R_a$ in general. Example: $\mathbf x = (0,0,1)$. $R_x(90°)$ sends it to $(0,-1,0)$, then $R_z(90°)$ sends that to $(1,0,0)$. The other order: $R_z(90°)$ leaves $(0,0,1)$ alone, then $R_x(90°)$ gives $(0,-1,0)$. Different results.</p>
    `);
    {
      const fig = L.figure(root, "<b>Order matters.</b> Left object: rotate about $x$ by α first, then about $z$ by β ($R_z R_x$). Right object: the other order ($R_x R_z$). Change the angles; drag to orbit. They only agree when one angle is 0.");
      const c = L.canvas(fig.el, { aspect: 0.62 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const sa = L.slider(ctl, { label: "α about $x$ (°)", min: -180, max: 180, step: 5, value: 90, oninput: () => c.redraw() });
      const sb = L.slider(ctl, { label: "β about $z$ (°)", min: -180, max: 180, step: 5, value: 90, oninput: () => c.redraw() });
      const view = orbit(c, { target: [0, 0, 0], dist: 7, zoom: 1.3, yaw: -0.55, pitch: 0.42 });
      const Fsegs = [[[0, 0.4, 0], [0, -0.5, 0]], [[0, -0.5, 0], [0.45, -0.5, 0]], [[0, -0.05, 0], [0.3, -0.05, 0]]];
      c.draw = (ctx) => {
        const t = L.theme(), a = sa.value * deg, b = sb.value * deg;
        const RA = la.matMul(la.rotZ(b), la.rotX(a)), RB = la.matMul(la.rotX(a), la.rotZ(b));
        [[RA, -1.3, "x then z"], [RB, 1.3, "z then x"]].forEach(([R, ox, name]) => {
          const T = { R, t: [ox, 0, 0] }, T0 = { R: la.eye(3), t: [ox, 0, 0] };
          for (const [p, q] of Fsegs) view.seg(ctx, apply(T0, p), apply(T0, q), t.line, 3);
          triad(ctx, view, T, t, 0.75);
          for (const [p, q] of Fsegs) view.seg(ctx, apply(T, p), apply(T, q), t.accent3, 4);
          view.label(ctx, [ox, 1.05, 0], name, t.fg, -24, 0);
        });
        const row = (R) => R.map((r) => `[${r.map(f2).join(", ")}]`).join(" ");
        out.html = `R<sub>z</sub>R<sub>x</sub> = ${row(RA)}<br>R<sub>x</sub>R<sub>z</sub> = ${row(RB)}`;
      };
    }

    // ------------------------------------------------------------ poses (eq 1)
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Poses: rotation + translation</h3>
      <p>A camera has a position and an orientation. A <b>rigid transform</b> rotates then shifts: $\mathbf x' = R\mathbf x + \mathbf t$. Appending a 1 to the point ($\mathbf x \to (x,y,z,1)$) turns it into a single $4\times 4$ matrix multiply, so transforms can be chained by matrix products.</p>
      <div class="eq-card"><div class="eq-label">Paper eq. (1) · camera pose</div>
      $$T_{wc} = \begin{pmatrix} R_{wc} & \mathbf c_w\\ \mathbf 0^\top & 1\end{pmatrix},\qquad \mathbf x_w = T_{wc}\,\mathbf x_c \;\;\text{i.e.}\;\; \mathbf x_w = R_{wc}\,\mathbf x_c + \mathbf c_w$$
      <div class="parts">
        <span>$T_{wc}$</span><span>pose of camera $c$ in world $w$: turns <b>camera</b> coordinates into <b>world</b> coordinates (read subscripts right to left: from $c$ to $w$)</span>
        <span>$R_{wc}$</span><span>orientation. Its columns are the camera's $x$, $y$, $z$ axes written in world coordinates; column 3 is the viewing direction</span>
        <span>$\mathbf c_w$</span><span>the camera centre in world coordinates (put $\mathbf x_c = \mathbf 0$ and you get $\mathbf x_w = \mathbf c_w$)</span>
        <span>$\mathbf 0^\top\ 1$</span><span>bottom row that makes the $4\times4$ trick work</span>
      </div></div>
      <div class="eq-card"><div class="eq-label">Composition and inverse</div>
      $$T_{ab}\,T_{bc} = T_{ac},\qquad T^{-1} = \begin{pmatrix}R^\top & -R^\top\mathbf t\\ \mathbf 0^\top & 1\end{pmatrix},\qquad T_{ab}^{-1} = T_{ba}$$
      <div class="parts">
        <span>$T_{ab}T_{bc}$</span><span>"c to b, then b to a". Inner subscripts cancel. Rotation $R_{ab}R_{bc}$, translation $R_{ab}\mathbf t_{bc} + \mathbf t_{ab}$</span>
        <span>$T^{-1}$</span><span>solve $\mathbf x' = R\mathbf x + \mathbf t$ for $\mathbf x$: $\mathbf x = R^\top(\mathbf x' - \mathbf t) = R^\top\mathbf x' - R^\top\mathbf t$</span>
        <span>$T_{cw} = T_{wc}^{-1}$</span><span>world → camera: $\mathbf x_c = R_{wc}^\top(\mathbf x_w - \mathbf c_w)$. This is what you use to see a world point from a camera</span>
      </div></div>
      <p><b>Worked example.</b> Camera at $\mathbf c_w = (1,0,2)$ with $R_{wc} = R_y(90°) = \left(\begin{smallmatrix}0&0&1\\0&1&0\\-1&0&0\end{smallmatrix}\right)$. Its viewing direction (column 3) is $(1,0,0)$: it looks along world $+x$. A point 3 m straight ahead, $\mathbf x_c = (0,0,3)$, is at $\mathbf x_w = (3,0,0) + (1,0,2) = (4,0,2)$. Back: $\mathbf x_w - \mathbf c_w = (3,0,0)$, and $R^\top(3,0,0) = (0,0,3)$ ✓.</p>
    `);
    {
      const fig = L.figure(root, "<b>Top-down view of the world</b> ($x$ right, $z$ up the page, $y$ into the page). Drag the camera (blue), its heading knob, and the point (orange). The readout converts the point into camera coordinates with $T_{cw}=T_{wc}^{-1}$.");
      const c = L.canvas(fig.el, { aspect: 0.75 });
      fig.add(c.el);
      const out = L.readout(fig.el); fig.add(out.el);
      const S = { cx: -1.2, cz: 0.5, yaw: 35 * deg, px: 1.4, pz: 3.2 };
      const geo = () => { const sc = c.w / 9; return { sc, X: (x) => c.w / 2 + x * sc, Y: (z) => c.h * 0.85 - z * sc, ix: (px) => (px - c.w / 2) / sc, iz: (py) => (c.h * 0.85 - py) / sc }; };
      const knob = () => [S.cx + 1.3 * Math.sin(S.yaw), S.cz + 1.3 * Math.cos(S.yaw)];
      L.drag(c, () => { const g = geo(), k = knob(); return [{ x: g.X(S.cx), y: g.Y(S.cz) }, { x: g.X(k[0]), y: g.Y(k[1]) }, { x: g.X(S.px), y: g.Y(S.pz) }]; }, (i, p) => {
        const g = geo(), x = g.ix(p.x), z = g.iz(p.y);
        if (i === 0) { S.cx = x; S.cz = z; } else if (i === 1) S.yaw = Math.atan2(x - S.cx, z - S.cz); else { S.px = x; S.pz = z; }
      });
      c.draw = (ctx) => {
        const t = L.theme(), g = geo();
        const R = la.rotY(S.yaw), cw = [S.cx, 0, S.cz], xw = [S.px, 0, S.pz];
        const xc = la.matVec(la.T(R), la.sub(xw, cw));
        // world axes
        L.draw.arrow(ctx, g.X(0), g.Y(0), g.X(1), g.Y(0), t.faint, 1.5);
        L.draw.arrow(ctx, g.X(0), g.Y(0), g.X(0), g.Y(1), t.faint, 1.5);
        L.draw.text(ctx, "world x", g.X(1) + 4, g.Y(0) + 4, t.faint, { size: 11 });
        L.draw.text(ctx, "world z", g.X(0) + 4, g.Y(1) - 4, t.faint, { size: 11 });
        // FOV wedge
        for (const s of [-1, 1]) {
          const a = S.yaw + s * 30 * deg;
          L.draw.line(ctx, g.X(S.cx), g.Y(S.cz), g.X(S.cx + 2.2 * Math.sin(a)), g.Y(S.cz + 2.2 * Math.cos(a)), t.accent, 1, [4, 4]);
        }
        // camera axes = columns of R (x: (cos,0,-sin), z: (sin,0,cos))
        L.draw.arrow(ctx, g.X(S.cx), g.Y(S.cz), g.X(S.cx + 0.8 * R[0][0]), g.Y(S.cz + 0.8 * R[2][0]), t.bad, 2.5);
        L.draw.text(ctx, "x_c", g.X(S.cx + 0.95 * R[0][0]), g.Y(S.cz + 0.95 * R[2][0]), t.bad, { size: 12, bold: true });
        const k = knob();
        L.draw.line(ctx, g.X(S.cx), g.Y(S.cz), g.X(k[0]), g.Y(k[1]), t.accent, 2.5);
        L.draw.text(ctx, "z_c", g.X(k[0]) + 10, g.Y(k[1]), t.accent, { size: 12, bold: true });
        L.draw.line(ctx, g.X(S.cx), g.Y(S.cz), g.X(S.px), g.Y(S.pz), xc[2] > 0 ? t.accent3 : t.bad, 1, [2, 4]);
        L.draw.handle(ctx, g.X(S.cx), g.Y(S.cz), t.accent);
        L.draw.handle(ctx, g.X(k[0]), g.Y(k[1]), t.accent);
        L.draw.handle(ctx, g.X(S.px), g.Y(S.pz), t.accent3);
        out.html = `c_w = ${v3(cw)}, R_wc = R_y(${f1(S.yaw / deg)}°)<br>x_w = ${v3(xw)}<br>x_c = R_wcᵀ(x_w − c_w) = <b>${v3(xc)}</b> → ${xc[2] > 0 ? "in front (z_c &gt; 0)" : "<b>behind</b> the camera (z_c &lt; 0)"}`;
      };
    }

    // ------------------------------------------------------------ pixel transfer
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Transferring a pixel to another camera</h3>
      <p>This is the core operation of DTAM. Take pixel $\mathbf u$ of the reference camera $r$ and guess its inverse depth $d$. Where does that 3D point appear in another camera $m$? Three steps: back-project (chapter 3), change coordinates, project.</p>
      <div class="eq-card"><div class="eq-label">Pixel transfer (inside paper eq. 3)</div>
      $$\mathbf u_m = \pi\!\left(K\,T_{mr}\,\pi^{-1}(\mathbf u, d)\right),\qquad T_{mr} = T_{mw}T_{wr} = T_{wm}^{-1}\,T_{wr}$$
      <div class="parts">
        <span>$\pi^{-1}(\mathbf u, d)$</span><span>the 3D point in $r$'s coordinates: $\mathbf x_r = \frac1d K^{-1}\dot{\mathbf u}$ (a 1 is appended to multiply by the $4\times4$ $T$)</span>
        <span>$T_{mr}$</span><span>moves it into $m$'s coordinates: $\mathbf x_m = R_{mr}\mathbf x_r + \mathbf t_{mr}$. Built from the two world poses</span>
        <span>$\pi(K\,\cdot)$</span><span>projects into $m$'s image (chapter 3)</span>
      </div></div>
      <p><b>Worked example.</b> $f=100$, $(c_x,c_y)=(50,50)$. Camera $m$ is 0.2 m to the right of $r$, same orientation, so $R_{mr}=I$, $\mathbf t_{mr} = (-0.2, 0, 0)$. Pixel $\mathbf u = (60, 50)$, $d = 0.5$:
      $\mathbf x_r = 2\cdot(0.1, 0, 1) = (0.2, 0, 2)$; $\mathbf x_m = (0, 0, 2)$; $\mathbf u_m = (100\cdot 0/2 + 50,\ 50) = (50, 50)$.</p>
      <p><b>How the GPU does it</b> (<code>cost_update.wgsl</code>). Multiply $K\mathbf x_m$ by $d$ (allowed: homogeneous scale does not change the pixel):</p>
      <div class="eq-card"><div class="eq-label">Transfer as a straight line in $d$</div>
      $$d\,K\mathbf x_m = \underbrace{K R_{mr} K^{-1}\dot{\mathbf u}}_{\mathbf a} + d\,\underbrace{K\mathbf t_{mr}}_{\mathbf b},\qquad \mathbf u_m = \pi(\mathbf a + d\,\mathbf b)$$
      <div class="parts">
        <span>$\mathbf a$</span><span>computed once per pixel (a $3\times3$ matrix times $\dot{\mathbf u}$); independent of $d$</span>
        <span>$\mathbf b$</span><span>the same for every pixel of this frame</span>
        <span>$d = 0$</span><span>point at infinity: $\mathbf u_m = \pi(\mathbf a)$ depends only on the rotation</span>
        <span>$d \to \infty$</span><span>$\mathbf u_m \to \pi(\mathbf b) = \pi(K\mathbf t_{mr})$, the <b>epipole</b>: where camera $r$'s centre appears in $m$</span>
      </div></div>
      <p>Straight lines through the origin stay straight lines after $\pi$, so as $d$ varies $\mathbf u_m$ slides along a straight line in image $m$: the <b>epipolar line</b> of $\mathbf u$. Worked example again: $\mathbf a = (60, 50, 1)$, $\mathbf b = K(-0.2,0,0) = (-20, 0, 0)$, $d=0.5$: $\mathbf a + d\mathbf b = (50, 50, 1)$ ✓.</p>
    `);
    {
      const K = { f: 140, cx: 79.5, cy: 59.5, w: 160, h: 120 };
      const fig = L.figure(root, "<b>Pixel transfer.</b> Left image: reference $r$ — drag pixel $\\mathbf u$. Right image: camera $m$ — drag the orange point along the epipolar line (or use the $d$ slider). The sliders also move camera $m$. When $d$ matches the box corner under $\\mathbf u$, the point lands on that corner in $m$. Drag the 3D view to orbit.");
      const c3 = L.canvas(fig.el, { aspect: 0.62 });
      fig.add(c3.el);
      const ci = L.canvas(fig.el, { aspect: 0.46 });
      fig.add(ci.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const redraw = () => { c3.redraw(); ci.redraw(); };
      // scene: a box
      const box = [];
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
        const x = sx * 0.4, z = sz * 0.4, a = 0.5;
        box.push([0.15 + Math.cos(a) * x + Math.sin(a) * z, 0.1 + sy * 0.4, 3 - Math.sin(a) * x + Math.cos(a) * z]);
      }
      const edges = [];
      for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) if ([1, 2, 4].includes(i ^ j)) edges.push([i, j]);
      const CORNER = 0; // front-top-left corner
      const proj = (x) => [K.f * x[0] / x[2] + K.cx, K.f * x[1] / x[2] + K.cy];
      const u0 = proj(box[CORNER]);
      const st = { u: u0[0], v: u0[1] };
      const sd = L.slider(ctl, { label: "inverse depth $d$", min: 0.05, max: 2.5, step: 0.005, value: 0.6, fmt: (x) => `${x.toFixed(3)}  (z = ${(1 / x).toFixed(2)} m)`, oninput: redraw });
      const sbx = L.slider(ctl, { label: "camera $m$: $x$ (m)", min: -0.8, max: 0.8, step: 0.01, value: 0.5, oninput: redraw });
      const sbz = L.slider(ctl, { label: "camera $m$: $z$ (m)", min: -0.6, max: 0.6, step: 0.01, value: 0, oninput: redraw });
      const syw = L.slider(ctl, { label: "camera $m$: turn (°)", min: -30, max: 30, step: 1, value: -8, oninput: redraw });
      const Twm = () => ({ R: la.rotY(syw.value * deg), t: [sbx.value, 0, sbz.value] });
      const Tmr = () => inv(Twm()); // T_wr = identity: the world is r's frame
      const Kmat = [[K.f, 0, K.cx], [0, K.f, K.cy], [0, 0, 1]];
      const Kinv = [[1 / K.f, 0, -K.cx / K.f], [0, 1 / K.f, -K.cy / K.f], [0, 0, 1]];
      const transfer = (d) => {
        const T = Tmr();
        const xr = la.scale(la.matVec(Kinv, [st.u, st.v, 1]), 1 / d);
        const xm = apply(T, xr);
        return { xr, xm, um: xm[2] > 1e-6 ? proj(xm) : null };
      };
      const lineAB = () => {
        const T = Tmr();
        return { a: la.matVec(la.matMul(la.matMul(Kmat, T.R), Kinv), [st.u, st.v, 1]), b: la.matVec(Kmat, T.t) };
      };
      const layout = () => {
        const pad = 8, iw = (ci.w - 3 * pad) / 2, s = iw / K.w, oy = 22;
        const mk = (ox) => ({ X: (u) => ox + (u + 0.5) * s, Y: (v) => oy + (v + 0.5) * s, iu: (x) => (x - ox) / s - 0.5, iv: (y) => (y - oy) / s - 0.5, ox, oy, W: iw, H: K.h * s });
        return { r: mk(pad), m: mk(2 * pad + iw) };
      };
      L.drag(ci, () => {
        const Ly = layout(), tr = transfer(sd.value);
        const pts = [{ x: Ly.r.X(st.u), y: Ly.r.Y(st.v) }];
        pts.push(tr.um ? { x: Ly.m.X(tr.um[0]), y: Ly.m.Y(tr.um[1]) } : { x: -999, y: -999 });
        return pts;
      }, (i, p) => {
        const Ly = layout();
        if (i === 0) {
          st.u = Math.max(0, Math.min(K.w - 1, Ly.r.iu(p.x)));
          st.v = Math.max(0, Math.min(K.h - 1, Ly.r.iv(p.y)));
        } else {
          const { a, b } = lineAB();
          let best = sd.value, bd = Infinity;
          for (let k = 0; k <= 490; k++) {
            const d = 0.05 + k * 0.005, q = la.add(a, la.scale(b, d));
            if (q[2] <= 1e-6) continue;
            const e = Math.hypot(Ly.m.X(q[0] / q[2]) - p.x, Ly.m.Y(q[1] / q[2]) - p.y);
            if (e < bd) { bd = e; best = d; }
          }
          sd.value = best;
        }
        c3.redraw();
      });
      const view = orbit(c3, { target: [0.2, 0, 1.7], dist: 8, zoom: 1.25, yaw: -0.9, pitch: 0.5 });
      c3.draw = (ctx) => {
        const t = L.theme(), T = Twm(), tr = transfer(sd.value), I = { R: la.eye(3), t: [0, 0, 0] };
        for (const [i, j] of edges) view.seg(ctx, box[i], box[j], t.muted, 1.5);
        view.dot(ctx, box[CORNER], 4, t.muted);
        drawFrustum(ctx, view, K, 0.45, t.accent, I);
        drawFrustum(ctx, view, K, 0.45, t.accent2, T);
        view.label(ctx, [0, 0, 0], "r", t.accent, -14, 14);
        view.label(ctx, T.t, "m", t.accent2, 6, 16);
        const dir = la.matVec(Kinv, [st.u, st.v, 1]);
        view.seg(ctx, [0, 0, 0], la.scale(dir, 7), t.accent, 1.2, [5, 4]);
        view.poly(ctx, [[0, 0, 0], T.t, tr.xr], t.accent4, 0.13);
        view.seg(ctx, tr.xr, T.t, t.accent2, 1.5);
        view.dot(ctx, tr.xr, 7, t.accent3, "white");
        view.label(ctx, tr.xr, "x", t.accent3, 9, -8);
      };
      ci.draw = (ctx) => {
        const t = L.theme(), Ly = layout(), T = Tmr(), tr = transfer(sd.value);
        const frame = (M, name, col) => {
          ctx.save(); ctx.fillStyle = t.panel2; ctx.fillRect(M.ox, M.oy, M.W, M.H); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.strokeRect(M.ox, M.oy, M.W, M.H); ctx.restore();
          L.draw.text(ctx, name, M.ox, M.oy - 7, col, { size: 12, bold: true });
        };
        const clip = (M, fn) => { ctx.save(); ctx.beginPath(); ctx.rect(M.ox, M.oy, M.W, M.H); ctx.clip(); fn(); ctx.restore(); };
        frame(Ly.r, "reference r", t.accent);
        frame(Ly.m, "camera m", t.accent2);
        const boxIn = (M, P) => {
          const q = box.map(P);
          clip(M, () => {
            for (const [i, j] of edges) if (q[i] && q[j]) L.draw.line(ctx, M.X(q[i][0]), M.Y(q[i][1]), M.X(q[j][0]), M.Y(q[j][1]), t.muted, 1.5);
            if (q[CORNER]) L.draw.dot(ctx, M.X(q[CORNER][0]), M.Y(q[CORNER][1]), 3, t.muted);
          });
          return q[CORNER];
        };
        boxIn(Ly.r, (x) => proj(x));
        const cm = boxIn(Ly.m, (x) => { const y = apply(T, x); return y[2] > 1e-3 ? proj(y) : null; });
        // epipolar line: pi(a + d b), d from 0 to "infinity"
        const { a, b } = lineAB();
        const pts = [];
        for (let k = 0; k <= 160; k++) {
          const d = 60 * (k / 160) ** 3, q = la.add(a, la.scale(b, d));
          if (q[2] > 1e-4) pts.push([Ly.m.X(q[0] / q[2]), Ly.m.Y(q[1] / q[2])]);
        }
        clip(Ly.m, () => {
          L.draw.path(ctx, pts, t.accent4, 2);
          if (a[2] > 1e-4) {
            const p0 = [Ly.m.X(a[0] / a[2]), Ly.m.Y(a[1] / a[2])];
            L.draw.dot(ctx, p0[0], p0[1], 3.5, t.accent4);
            L.draw.text(ctx, "d=0 (∞)", p0[0] + 5, p0[1] - 5, t.accent4, { size: 11 });
          }
          if (b[2] > 1e-6) {
            const e = [Ly.m.X(b[0] / b[2]), Ly.m.Y(b[1] / b[2])];
            L.draw.dot(ctx, e[0], e[1], 3.5, t.accent);
            L.draw.text(ctx, "epipole", e[0] + 5, e[1] + 13, t.accent, { size: 11 });
          }
        });
        L.draw.handle(ctx, Ly.r.X(st.u), Ly.r.Y(st.v), t.accent);
        if (tr.um) {
          const inside = tr.um[0] >= -0.5 && tr.um[0] <= K.w - 0.5 && tr.um[1] >= -0.5 && tr.um[1] <= K.h - 0.5;
          if (inside) L.draw.handle(ctx, Ly.m.X(tr.um[0]), Ly.m.Y(tr.um[1]), t.accent3);
        }
        const hit = cm && tr.um && Math.hypot(tr.um[0] - cm[0], tr.um[1] - cm[1]) < 1.5 && Math.hypot(st.u - u0[0], st.v - u0[1]) < 1.5;
        const um = tr.um ? `(<b>${f1(tr.um[0])}</b>, <b>${f1(tr.um[1])}</b>)` : "behind camera m";
        out.html = `u = (${f1(st.u)}, ${f1(st.v)}), d = ${sd.value.toFixed(3)}<br>x_r = π⁻¹(u, d) = ${v3(tr.xr)}<br>x_m = R_mr x_r + t_mr = ${v3(tr.xm)}<br>u_m = π(K x_m) = ${um}` +
          (hit ? ` <b style="color:${t.good}">✓ lands on the corner: this d is the true depth</b>` : "");
      };
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <div class="key">For one reference pixel, every candidate inverse depth $d$ is a point on one line in image $m$. DTAM's cost volume (chapter 8) compares $I_r(\mathbf u)$ with $I_m$ at each of those points; the right $d$ is where the colours agree.</div>
<pre><code>transfer(u, d, K, T_mr):                 // pixel of r + inverse depth -> pixel of m
    x_r = (1/d) * ((u.u - cx)/fx, (u.v - cy)/fy, 1)     // π⁻¹(u, d)
    x_m = T_mr.R * x_r + T_mr.t
    if x_m.z <= 0: return NONE                           // behind camera m
    return (fx * x_m.x / x_m.z + cx,  fy * x_m.y / x_m.z + cy)

// GPU form, per pixel: a = (K R_mr K⁻¹) u̇ once, b = K t_mr once per frame,
// then for each layer d:  q = a + d*b;  u_m = (q.x/q.z, q.y/q.z)</code></pre>

      <h3>Small rotations and the skew matrix</h3>
      <p>Tracking (chapter 11) nudges a pose by tiny amounts. For a tiny rotation, a point moves perpendicular to both the rotation axis and itself, like a spinning wheel's rim. Describe the rotation by a vector $\boldsymbol\omega$: its direction is the axis, its length the angle in radians. To first order, $R\mathbf x \approx \mathbf x + \boldsymbol\omega\times\mathbf x$ (recall the cross product from chapter 2). A cross product with a fixed $\boldsymbol\omega$ is itself a matrix:</p>
      <div class="eq-card"><div class="eq-label">Skew (cross-product) matrix</div>
      $$[\boldsymbol\omega]_\times = \begin{pmatrix}0 & -\omega_3 & \omega_2\\ \omega_3 & 0 & -\omega_1\\ -\omega_2 & \omega_1 & 0\end{pmatrix},\qquad [\boldsymbol\omega]_\times\mathbf x = \boldsymbol\omega\times\mathbf x,\qquad R \approx I + [\boldsymbol\omega]_\times$$
      <div class="parts">
        <span>$[\boldsymbol\omega]_\times$</span><span>"skew-symmetric": $[\boldsymbol\omega]_\times^\top = -[\boldsymbol\omega]_\times$, zeros on the diagonal</span>
        <span>$\boldsymbol\omega\times\mathbf x$</span><span>velocity of point $\mathbf x$ when rotating about axis $\boldsymbol\omega$; zero for points on the axis</span>
        <span>$I + [\boldsymbol\omega]_\times$</span><span>first-order rotation; good only for small $|\boldsymbol\omega|$ (it is not exactly a rotation: it slightly stretches)</span>
      </div></div>
      <p><b>Worked example.</b> $\boldsymbol\omega = (0,0,0.1)$ (0.1 rad ≈ 5.7° about $z$), $\mathbf x = (1,0,0)$: $\boldsymbol\omega\times\mathbf x = (0\cdot0 - 0.1\cdot 0,\ 0.1\cdot 1 - 0\cdot 0,\ 0\cdot 0 - 0\cdot 1) = (0, 0.1, 0)$, so $R\mathbf x\approx(1, 0.1, 0)$. Exact: $(\cos 0.1, \sin 0.1, 0) = (0.995, 0.0998, 0)$.</p>

      <h3>The exponential map: many tiny rotations make a big one</h3>
      <p>To rotate by a large $\boldsymbol\omega$ exactly, split it into $n$ tiny steps $\boldsymbol\omega/n$, apply the first-order step $n$ times, and let $n$ grow. The limit is written $\exp([\boldsymbol\omega]_\times)$, just like $e^x = \lim (1 + x/n)^n$.</p>
    `);
    {
      const fig = L.figure(root, "<b>Exponential map as a limit.</b> Rotation about $z$ by angle θ, seen from the front ($y$ up here). The polygon applies $I + [\\boldsymbol\\omega]_\\times/n$ to $(1,0,0)$ $n$ times. Increase $n$: it converges to the exact rotation (dashed arc).");
      const c = L.canvas(fig.el, { aspect: 0.7, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const sth = L.slider(ctl, { label: "angle θ (rad)", min: 0, max: 6.28, step: 0.01, value: 2, oninput: () => c.redraw() });
      const sn = L.slider(ctl, { label: "steps $n$", min: 1, max: 64, step: 1, value: 3, oninput: () => c.redraw() });
      L.button(ctl, "n × 2", () => { sn.value = Math.min(64, sn.value * 2); c.redraw(); });
      L.button(ctl, "n = 1", () => { sn.value = 1; c.redraw(); });
      c.draw = (ctx) => {
        const t = L.theme(), th = sth.value, n = sn.value;
        const sc = Math.min(c.w, c.h) / 5.2, X = (x) => c.w / 2 + x * sc, Y = (y) => c.h / 2 - y * sc;
        L.draw.line(ctx, 0, Y(0), c.w, Y(0), t.line, 1);
        L.draw.line(ctx, X(0), 0, X(0), c.h, t.line, 1);
        ctx.save(); ctx.strokeStyle = t.muted; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(X(0), Y(0), sc, 0, -th, true); ctx.stroke(); ctx.restore();
        let p = [1, 0];
        const pts = [[X(1), Y(0)]];
        for (let k = 0; k < n; k++) {
          const h = th / n;
          p = [p[0] - h * p[1], p[1] + h * p[0]];
          pts.push([X(p[0]), Y(p[1])]);
        }
        L.draw.path(ctx, pts, t.accent2, 2);
        if (n <= 32) for (const q of pts) L.draw.dot(ctx, q[0], q[1], 3, t.accent2);
        const ex = [Math.cos(th), Math.sin(th)];
        L.draw.dot(ctx, X(ex[0]), Y(ex[1]), 6, t.good);
        L.draw.dot(ctx, pts[pts.length - 1][0], pts[pts.length - 1][1], 6, t.accent3);
        L.draw.dot(ctx, X(1), Y(0), 4, t.fg);
        const err = Math.hypot(p[0] - ex[0], p[1] - ex[1]);
        out.html = `(I + [ω]×/${n})^${n}·(1,0,0) = (<b>${f3(p[0])}</b>, <b>${f3(p[1])}</b>, 0) · exact (${f3(ex[0])}, ${f3(ex[1])}, 0) · error ${f3(err)} · length ${f3(Math.hypot(...p))}`;
      };
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>The limit has a closed form, found by Rodrigues. It needs only $[\boldsymbol\omega]_\times$ and its square:</p>
      <div class="eq-card"><div class="eq-label">Rodrigues' formula (<code>Se3::exp</code>, rotation part)</div>
      $$R = \exp([\boldsymbol\omega]_\times) = I + \frac{\sin\theta}{\theta}[\boldsymbol\omega]_\times + \frac{1-\cos\theta}{\theta^2}[\boldsymbol\omega]_\times^2,\qquad \theta = |\boldsymbol\omega|$$
      <div class="parts">
        <span>$\theta = |\boldsymbol\omega|$</span><span>rotation angle (radians)</span>
        <span>$\mathbf n = \boldsymbol\omega/\theta$</span><span>rotation axis (unit vector). $\boldsymbol\omega$ is called a <b>rotation vector</b> or axis–angle</span>
        <span>vector form</span><span>$R\mathbf x = \mathbf x\cos\theta + (\mathbf n\times\mathbf x)\sin\theta + \mathbf n\,(\mathbf n\cdot\mathbf x)(1-\cos\theta)$ — the same thing applied to one point</span>
        <span>$\theta\to 0$</span><span>use the series $\frac{\sin\theta}{\theta}\approx 1 - \frac{\theta^2}{6}$, $\frac{1-\cos\theta}{\theta^2}\approx\frac12 - \frac{\theta^2}{24}$ to avoid 0/0 (the code does this below $10^{-8}$)</span>
      </div></div>
      <p><b>Worked example.</b> $\boldsymbol\omega = (0, 0, \pi/2)$: $\theta = \pi/2$, $\frac{\sin\theta}{\theta} = \frac2\pi$, $\frac{1-\cos\theta}{\theta^2} = \frac4{\pi^2}$. $[\boldsymbol\omega]_\times$ has $-\pi/2$ at (1,2) and $\pi/2$ at (2,1); $[\boldsymbol\omega]_\times^2 = \text{diag}(-\pi^2/4, -\pi^2/4, 0)$. So $R = I + \frac2\pi[\boldsymbol\omega]_\times + \frac4{\pi^2}[\boldsymbol\omega]_\times^2 = \left(\begin{smallmatrix}0&-1&0\\1&0&0\\0&0&1\end{smallmatrix}\right) = R_z(90°)$ ✓.</p>

      <h3>Twists: the six ways to move</h3>
      <p>A full pose change has 3 rotation numbers and 3 translation numbers. Stack them into a <b>twist</b> $\psi = (\mathbf v, \boldsymbol\omega)\in\mathbb R^6$ (this implementation puts translation first). Each of the six numbers scales one <b>generator</b>, a $4\times4$ matrix describing one elementary motion:</p>
      <div class="eq-card"><div class="eq-label">SE(3) generators</div>
      $$\text{gen}_1 = \left(\begin{smallmatrix}0&0&0&1\\0&0&0&0\\0&0&0&0\\0&0&0&0\end{smallmatrix}\right)\ \cdots\quad \text{gen}_4 = \left(\begin{smallmatrix}0&0&0&0\\0&0&-1&0\\0&1&0&0\\0&0&0&0\end{smallmatrix}\right)\ \cdots\qquad \sum_{i=1}^6\psi_i\,\text{gen}_i = \begin{pmatrix}[\boldsymbol\omega]_\times & \mathbf v\\ \mathbf 0^\top & 0\end{pmatrix}$$
      <div class="parts">
        <span>$\text{gen}_1,\text{gen}_2,\text{gen}_3$</span><span>translate along $x$, $y$, $z$: a 1 in the last column, row 1, 2, 3</span>
        <span>$\text{gen}_4,\text{gen}_5,\text{gen}_6$</span><span>rotate about $x$, $y$, $z$: $[\mathbf e_1]_\times$, $[\mathbf e_2]_\times$, $[\mathbf e_3]_\times$ in the top-left block</span>
        <span>$\text{gen}_i\,(\mathbf x, 1)$</span><span>the velocity of point $\mathbf x$ under motion $i$: $\mathbf e_1, \mathbf e_2, \mathbf e_3$, then $\mathbf e_1\times\mathbf x$, $\mathbf e_2\times\mathbf x$, $\mathbf e_3\times\mathbf x$</span>
      </div></div>
      <div class="eq-card"><div class="eq-label">Paper eq. (21) · pose from a twist</div>
      $$T(\psi) = \exp\!\Big(\sum_{i=1}^6 \psi_i\,\text{gen}_i\Big) = \begin{pmatrix}R & V\mathbf v\\ \mathbf 0^\top & 1\end{pmatrix},\quad V = I + \frac{1-\cos\theta}{\theta^2}[\boldsymbol\omega]_\times + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol\omega]_\times^2$$
      <div class="parts">
        <span>$R$</span><span>Rodrigues of $\boldsymbol\omega$ (above)</span>
        <span>$V\mathbf v$</span><span>translation. If $\boldsymbol\omega = 0$ then $V = I$ and it is just $\mathbf v$; otherwise the motion is a screw (rotate while moving), and the end point is $V\mathbf v$</span>
        <span>small $\psi$</span><span>$T(\psi)\,\mathbf x \approx \mathbf x + \mathbf v + \boldsymbol\omega\times\mathbf x$</span>
      </div></div>
    `);
    {
      const fig = L.figure(root, "<b>Twist explorer.</b> Set the six numbers of ψ. Faint: the object before; solid: after $T(\\psi)$. The orange vertex's true path (solid, exp of $s\\psi$ for $s$ from 0 to 1) versus the first-order straight line $\\mathbf x + s(\\mathbf v + \\boldsymbol\\omega\\times\\mathbf x)$ (dashed). Tap a generator button to see its velocity arrows at every vertex. Drag to orbit.");
      const c = L.canvas(fig.el, { aspect: 0.66 });
      fig.add(c.el);
      const gbar = L.controls(fig.el); fig.add(gbar);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      let gen = -1;
      const names = ["ψ₁ move x", "ψ₂ move y", "ψ₃ move z", "ψ₄ turn x", "ψ₅ turn y", "ψ₆ turn z"];
      const gbtn = [];
      names.forEach((nm, i) => gbtn.push(L.button(gbar, `gen ${i + 1}`, () => { gen = gen === i ? -1 : i; gbtn.forEach((b, k) => b.classList.toggle("primary", k === gen)); c.redraw(); })));
      const sl = names.map((nm, i) => L.slider(ctl, { label: nm, min: -1, max: 1, step: 0.01, value: [0.3, 0, 0, 0, 0.6, 0][i], oninput: () => c.redraw() }));
      L.button(ctl, "reset ψ", () => { sl.forEach((s) => { s.value = 0; }); c.redraw(); });
      const view = orbit(c, { target: [0.3, 0, 1.6], dist: 8, zoom: 1.2, yaw: -0.85, pitch: 0.55 });
      const obj = [];
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) obj.push([0.4 + sx * 0.35, -0.1 + sy * 0.25, 2 + sz * 0.3]);
      const edges = [];
      for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) if ([1, 2, 4].includes(i ^ j)) edges.push([i, j]);
      c.draw = (ctx) => {
        const t = L.theme(), psi = sl.map((s) => s.value), T = expSE3(psi);
        const I = { R: la.eye(3), t: [0, 0, 0] };
        triad(ctx, view, I, t, 0.6, 1.8);
        view.label(ctx, [0, 0, 0], "origin", t.muted, -18, 18);
        for (const [i, j] of edges) view.seg(ctx, obj[i], obj[j], t.faint, 1.5);
        const moved = obj.map((p) => apply(T, p));
        for (const [i, j] of edges) view.seg(ctx, moved[i], moved[j], t.accent, 2.5);
        const P = obj[7];
        const path = [], lin = [];
        const vel = la.add(psi.slice(0, 3), la.cross(psi.slice(3), P));
        for (let k = 0; k <= 24; k++) {
          const s = k / 24;
          path.push(apply(expSE3(psi.map((x) => x * s)), P));
          lin.push(la.add(P, la.scale(vel, s)));
        }
        for (let k = 0; k < 24; k++) { view.seg(ctx, path[k], path[k + 1], t.accent3, 2.5); view.seg(ctx, lin[k], lin[k + 1], t.accent3, 1.2, [3, 3]); }
        view.dot(ctx, P, 4, t.accent3);
        view.dot(ctx, moved[7], 6, t.accent3, "white");
        if (gen >= 0) {
          for (const p of obj) {
            const e = [0, 0, 0]; e[gen % 3] = 1;
            const g = gen < 3 ? e : la.cross(e, p);
            view.arrow(ctx, p, la.add(p, la.scale(g, 0.35)), t.accent2, 2);
          }
        }
        const r = (i) => `[${T.R[i].map(f2).join(", ")} | ${f2(T.t[i])}]`;
        out.html = `T(ψ) = ${r(0)}<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${r(1)}<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${r(2)}` +
          (gen >= 0 ? `<br>arrows: gen ${gen + 1}·(x,1) = ${gen < 3 ? "e" + (gen + 1) + " (same for every point)" : "e" + (gen - 2) + " × x (grows with distance from the axis)"}` : "");
      };
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <p><b>Worked example (first order).</b> $\mathbf x = (1, 2, 4)$, $\psi = (0.01, 0, 0,\ 0, 0.02, 0)$: $\boldsymbol\omega\times\mathbf x = (0.02\cdot 4 - 0\cdot 2,\ 0\cdot 1 - 0\cdot 4,\ 0\cdot 2 - 0.02\cdot 1) = (0.08, 0, -0.02)$, so $T(\psi)\mathbf x \approx (1.09, 2, 3.98)$.</p>
      <h3>The derivative the tracker needs</h3>
      <p>How does a point move as each $\psi_i$ changes, starting from $\psi = 0$? From $\mathbf x + \mathbf v + \boldsymbol\omega\times\mathbf x$, and $\boldsymbol\omega\times\mathbf x = -\mathbf x\times\boldsymbol\omega = -[\mathbf x]_\times\boldsymbol\omega$:</p>
      <div class="eq-card"><div class="eq-label">Point derivative at ψ = 0 (used in chapter 11)</div>
      $$\frac{\partial\, T(\psi)\mathbf x}{\partial\psi}\Big|_{\psi=0} = \big[\, I_{3\times3} \;\big|\; -[\mathbf x]_\times \,\big] \qquad (3\times 6)$$
      <div class="parts">
        <span>columns 1–3</span><span>$\mathbf e_1, \mathbf e_2, \mathbf e_3$: translating moves every point equally</span>
        <span>columns 4–6</span><span>$\mathbf e_1\times\mathbf x$, $\mathbf e_2\times\mathbf x$, $\mathbf e_3\times\mathbf x$ (the columns of $-[\mathbf x]_\times$)</span>
      </div></div>
      <p>Example: $\mathbf x = (1,2,4)$, column 5 (turn about $y$) $= \mathbf e_2\times\mathbf x = (1\cdot4 - 0\cdot2,\ 0\cdot1 - 0\cdot4,\ 0\cdot2 - 1\cdot1) = (4, 0, -1)$.</p>

      <h3>Why optimise a small twist?</h3>
      <ul>
        <li>A rotation matrix has 9 entries but only 3 degrees of freedom. Nudging the entries directly breaks $R^\top R = I$.</li>
        <li>Every $\psi\in\mathbb R^6$ gives a valid pose through $\exp$, so an optimiser can step in any direction.</li>
        <li>Near $\psi = 0$, the effect on a point is the simple linear $\mathbf v + \boldsymbol\omega\times\mathbf x$, so derivatives are easy (above).</li>
      </ul>
      <p>So each Gauss–Newton step (chapter 5) solves for a small $\psi$ around the <b>current</b> estimate and then folds it in: the paper's forward-compositional update $\hat T_{lv}\leftarrow\hat T_{lv}\,T(\psi)$, which this implementation's tracker uses. $\psi$ is then reset to 0 for the next step.</p>
<pre><code>se3_exp(psi):                          // psi = (v1, v2, v3, w1, w2, w3), as geom.rs Se3::exp
    v = psi[0..3];  w = psi[3..6]
    th = |w|;  W = skew(w);  W2 = W * W
    if th < 1e-8:  A = 1 - th²/6;  B = 1/2 - th²/24;  C = 1/6 - th²/120
    else:          A = sin(th)/th;  B = (1 - cos th)/th²;  C = (th - sin th)/th³
    R = I + A*W + B*W2                 // Rodrigues
    t = (I + B*W + C*W2) * v           // V v
    return (R, t)

skew(w) = [[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]]
compose((Ra, ta), (Rb, tb)) = (Ra*Rb, Ra*tb + ta)      // apply b first
inverse((R, t)) = (Rᵀ, -Rᵀ t)</code></pre>
      <div class="note">After many compositions, floating-point error makes $R$ drift slightly away from a rotation. <code>Se3::exp</code> snaps its result back to the nearest rotation (<code>orthonormalize</code>).</div>
    `);

    // ------------------------------------------------------------ quiz
    const fmt = L.fmt;
    const rotBy = { x: la.rotX, y: la.rotY, z: la.rotZ };
    const nice = [
      ["R_y(90°)", la.rotY(Math.PI / 2)], ["R_y(-90°)", la.rotY(-Math.PI / 2)], ["R_z(90°)", la.rotZ(Math.PI / 2)],
      ["R_x(-90°)", la.rotX(-Math.PI / 2)], ["R_y(180°)", la.rotY(Math.PI)], ["R_x(90°)", la.rotX(Math.PI / 2)],
    ].map(([n, R]) => [n, R.map((row) => row.map((x) => Math.round(x)))]);
    const vi = (r, a = 3) => [r.int(-a, a), r.int(-a, a), r.int(-a, a)];
    L.quiz(root, "motion", [
      { id: "rot2d", type: "num",
        gen: (r) => {
          const th = r.pick([30, 45, 60, 90, 120, 135, 150, 180, -30, -60, -90]), x = r.nz(4), y = r.int(-4, 4);
          const c = Math.cos(th * deg), s = Math.sin(th * deg);
          return {
            q: String.raw`Rotate the 2D point $(${x}, ${y})$ by $\theta = ${th}°$ (counter-clockwise, $y$ up). Give the result to 2 decimals.`,
            answer: [c * x - s * y, s * x + c * y], labels: ["$x'$", "$y'$"], tol: 0.011,
            explain: String.raw`$\cos ${th}° = ${fmt(c, 4)}$, $\sin ${th}° = ${fmt(s, 4)}$. $x' = x\cos\theta - y\sin\theta = ${fmt(c * x - s * y, 4)}$, $y' = x\sin\theta + y\cos\theta = ${fmt(s * x + c * y, 4)}$.`,
          };
        } },
      { id: "rot_order", type: "num",
        gen: (r) => {
          const [A, B] = r.pick([["x", "z"], ["z", "x"], ["x", "y"], ["y", "x"], ["y", "z"], ["z", "y"]]);
          const a = r.pick([90, -90]), b = r.pick([90, -90]);
          const x = vi(r);
          if (x.every((k) => k === 0)) x[2] = 2;
          const RA = rotBy[A](a * deg), RB = rotBy[B](b * deg);
          const y1 = clean(la.matVec(RA, x)), y2 = clean(la.matVec(RB, y1));
          return {
            q: String.raw`Start from $\mathbf x = ${vtex(x)}$. First rotate by $${a}°$ about the $${A}$ axis, then by $${b}°$ about the $${B}$ axis (use $R_x, R_y, R_z$ exactly as defined in this chapter). Where does the point end up?`,
            answer: y2, labels: ["$x$", "$y$", "$z$"], tol: 0.01,
            explain: String.raw`The result is $R_${B}(${b}°)\,R_${A}(${a}°)\,\mathbf x$ (the first rotation sits next to $\mathbf x$). $R_${A}(${a}°) = ${mtex(RA.map(clean))}$ gives $${vtex(y1)}$; then $R_${B}(${b}°) = ${mtex(RB.map(clean))}$ gives $${vtex(y2)}$.`,
          };
        } },
      { id: "is_rot", type: "multi",
        q: "Which of these matrices are rotation matrices? (select all)",
        choices: [
          String.raw`$\begin{pmatrix}0&-1&0\\1&0&0\\0&0&1\end{pmatrix}$`,
          String.raw`$\begin{pmatrix}0.6&0.8&0\\-0.8&0.6&0\\0&0&1\end{pmatrix}$`,
          String.raw`$\begin{pmatrix}0&0&1\\1&0&0\\0&1&0\end{pmatrix}$`,
          String.raw`$\begin{pmatrix}1&0&0\\0&1&0\\0&0&-1\end{pmatrix}$`,
          String.raw`$\begin{pmatrix}2&0&0\\0&0.5&0\\0&0&1\end{pmatrix}$`,
          String.raw`$\begin{pmatrix}1&0&0\\0&0&1\\0&1&0\end{pmatrix}$`,
        ],
        answer: [0, 1, 2],
        explain: String.raw`A rotation needs orthonormal columns ($R^\top R=I$) <b>and</b> $\det R = +1$. The first three pass. $\text{diag}(1,1,-1)$ and the $y$/$z$ swap are mirrors ($\det=-1$). $\text{diag}(2, 0.5, 1)$ has $\det = 1$ but stretches: its columns are not unit length.` },
      { id: "pose_to_world", type: "num",
        gen: (r) => {
          const [nm, R] = r.pick(nice), cw = vi(r, 4), xc = [r.int(-2, 2), r.int(-2, 2), r.int(1, 5)];
          const xw = clean(la.add(la.matVec(R, xc), cw));
          return {
            q: String.raw`A camera has pose $T_{wc}$ with $R_{wc} = ${mtex(R)}$ and centre $\mathbf c_w = ${vtex(cw)}$. A point has camera coordinates $\mathbf x_c = ${vtex(xc)}$. What are its world coordinates $\mathbf x_w$?`,
            answer: xw, labels: ["$x_w$", "$y_w$", "$z_w$"], tol: 0.01,
            explain: String.raw`$\mathbf x_w = R_{wc}\mathbf x_c + \mathbf c_w = ${vtex(clean(la.matVec(R, xc)))} + ${vtex(cw)} = ${vtex(xw)}$. (This $R_{wc}$ is $${nm}$.)`,
          };
        } },
      { id: "world_to_cam", type: "num",
        gen: (r) => {
          const [nm, R] = r.pick(nice), cw = vi(r, 4), xw = vi(r, 5);
          const diff = la.sub(xw, cw), xc = clean(la.matVec(la.T(R), diff));
          return {
            q: String.raw`Camera pose: $R_{wc} = ${mtex(R)}$, $\mathbf c_w = ${vtex(cw)}$. A world point is $\mathbf x_w = ${vtex(xw)}$. What are its coordinates $\mathbf x_c$ in this camera's frame? Is it in front of the camera?`,
            answer: xc, labels: ["$x_c$", "$y_c$", "$z_c$"], tol: 0.01,
            explain: String.raw`$\mathbf x_c = R_{wc}^\top(\mathbf x_w - \mathbf c_w)$. $\mathbf x_w - \mathbf c_w = ${vtex(diff)}$; $R_{wc}^\top = ${mtex(la.T(R))}$; product $= ${vtex(xc)}$. It is ${xc[2] > 0 ? "in front ($z_c > 0$)" : "not in front ($z_c \\le 0$)"}.`,
          };
        } },
      { id: "cw_meaning", type: "mc",
        q: String.raw`In $T_{wc} = \begin{pmatrix}R_{wc}&\mathbf c_w\\\mathbf 0^\top&1\end{pmatrix}$, which statement is right?`,
        choices: [
          String.raw`$\mathbf c_w$ is the camera centre in world coordinates, and column 3 of $R_{wc}$ is the camera's viewing direction in world coordinates`,
          String.raw`$\mathbf c_w$ is the world origin expressed in camera coordinates, and row 3 of $R_{wc}$ is the viewing direction`,
          String.raw`$\mathbf c_w$ is the point the camera is looking at, and column 3 of $R_{wc}$ is the world's $z$ axis in camera coordinates`,
          String.raw`$\mathbf c_w$ is the camera centre in world coordinates, and row 3 of $R_{wc}$ is the camera's viewing direction in world coordinates`,
        ],
        answer: 0,
        explain: String.raw`$\mathbf x_c = \mathbf 0 \Rightarrow \mathbf x_w = \mathbf c_w$. $\mathbf x_c = (0,0,1)$ (one step along the camera's $z$) maps to $\mathbf c_w + $ column 3. The world origin in camera coordinates is $-R_{wc}^\top\mathbf c_w$ (the translation of $T_{cw}$).` },
      { id: "t_mr", type: "num",
        gen: (r) => {
          const [nm, R] = r.pick(nice), cr = vi(r, 3), cm = vi(r, 3);
          const tmr = clean(la.matVec(la.T(R), la.sub(cr, cm)));
          return {
            q: String.raw`Reference camera: $R_{wr} = I$, $\mathbf c_r = ${vtex(cr)}$. Camera $m$: $R_{wm} = ${mtex(R)}$, $\mathbf c_m = ${vtex(cm)}$. Find the translation $\mathbf t_{mr}$ of $T_{mr}$ (the transform taking $r$-coordinates to $m$-coordinates).`,
            answer: tmr, labels: [String.raw`$t_1$`, String.raw`$t_2$`, String.raw`$t_3$`], tol: 0.01,
            explain: String.raw`$T_{mr} = T_{wm}^{-1}T_{wr}$. $T_{wm}^{-1} = (R_{wm}^\top,\ -R_{wm}^\top\mathbf c_m)$; composing with $(I, \mathbf c_r)$ gives rotation $R_{wm}^\top$ and translation $R_{wm}^\top\mathbf c_r - R_{wm}^\top\mathbf c_m = R_{wm}^\top(\mathbf c_r - \mathbf c_m) = ${mtex(la.T(R))}${vtex(la.sub(cr, cm))} = ${vtex(tmr)}$. (Check: $\mathbf t_{mr}$ is $r$'s centre seen from $m$.)`,
          };
        } },
      { id: "transfer", type: "num",
        gen: (r) => {
          const f = 500, cx = 319.5, cy = 239.5;
          const du = r.pick([-100, -50, 0, 50, 100, 150]), dv = r.pick([-100, -50, 0, 50, 100]), d = r.pick([0.5, 0.25, 1]);
          const t = [r.pick([-0.2, 0.1, 0.3, -0.4]), r.pick([0, 0.1, -0.1]), r.pick([0, 0.5, -0.5, 1])];
          const xr = [du / f / d, dv / f / d, 1 / d], xm = la.add(xr, t);
          const um = [f * xm[0] / xm[2] + cx, f * xm[1] / xm[2] + cy];
          return {
            q: String.raw`$f = 500$, $(c_x, c_y) = (319.5, 239.5)$ for both cameras. $R_{mr} = I$, $\mathbf t_{mr} = ${vtex(t)}$. Transfer pixel $\mathbf u = (${cx + du}, ${cy + dv})$ of $r$ at inverse depth $d = ${d}$ into camera $m$ (1 decimal).`,
            answer: um, labels: ["$u_m$", "$v_m$"], tol: 0.06,
            explain: String.raw`$\mathbf x_r = \frac1d K^{-1}\dot{\mathbf u} = \frac1{${d}}(${fmt(du / f)}, ${fmt(dv / f)}, 1) = ${vtex(xr)}$. $\mathbf x_m = \mathbf x_r + \mathbf t_{mr} = ${vtex(xm)}$. $u_m = 500\cdot ${fmt(xm[0], 4)}/${fmt(xm[2], 4)} + 319.5 = ${fmt(um[0], 3)}$, $v_m = 500\cdot ${fmt(xm[1], 4)}/${fmt(xm[2], 4)} + 239.5 = ${fmt(um[1], 3)}$.`,
          };
        } },
      { id: "epipole", type: "num",
        gen: (r) => {
          const f = r.pick([400, 500]), cx = 319.5, cy = 239.5;
          const t = [r.pick([-0.3, -0.2, 0.1, 0.2, 0.4]), r.pick([-0.1, 0, 0.05, 0.2]), r.pick([0.5, 1, 2])];
          const e = [f * t[0] / t[2] + cx, f * t[1] / t[2] + cy];
          return {
            q: String.raw`$f = ${f}$, $(c_x,c_y)=(319.5, 239.5)$, $\mathbf t_{mr} = ${vtex(t)}$ (any $R_{mr}$). As the inverse depth $d$ of a reference pixel grows without bound, where in image $m$ does the transferred point end up? (the epipole, 1 decimal)`,
            answer: e, labels: ["$u$", "$v$"], tol: 0.06,
            explain: String.raw`$\pi(\mathbf a + d\,\mathbf b)\to\pi(\mathbf b)$ as $d\to\infty$, with $\mathbf b = K\mathbf t_{mr}$. So $u = f\,t_1/t_3 + c_x = ${f}\cdot ${t[0]}/${t[2]} + 319.5 = ${fmt(e[0], 3)}$, $v = ${f}\cdot ${t[1]}/${t[2]} + 239.5 = ${fmt(e[1], 3)}$. It is where camera $r$'s centre appears in $m$ (independent of the pixel).`,
          };
        } },
      { id: "infinity", type: "num",
        gen: (r) => {
          const f = r.pick([400, 500, 600]), th = r.pick([10, 15, 20, 30, -10, -20, -25]), t = [r.pick([0.1, -0.2]), 0, r.pick([0, 0.3])];
          const u = f * Math.tan(th * deg) + 319.5;
          return {
            q: String.raw`$f=${f}$, $(c_x,c_y)=(319.5, 239.5)$. Camera $m$ has $R_{mr} = R_y(${th}°)$ and $\mathbf t_{mr} = ${vtex(t)}$. Where does the <b>centre pixel</b> $(319.5, 239.5)$ of $r$ land in $m$ for a point infinitely far away ($d = 0$)? (1 decimal)`,
            answer: [u, 239.5], labels: ["$u_m$", "$v_m$"], tol: 0.06,
            explain: String.raw`At $d=0$, $\mathbf u_m = \pi(K R_{mr} K^{-1}\dot{\mathbf u})$: the translation does not matter. $K^{-1}\dot{\mathbf u} = (0,0,1)$; $R_y(\theta)(0,0,1) = (\sin\theta, 0, \cos\theta)$; projecting: $u_m = f\tan\theta + c_x = ${f}\tan(${th}°) + 319.5 = ${fmt(u, 3)}$, $v_m = 239.5$.`,
          };
        } },
      { id: "epiline", type: "mc",
        q: String.raw`Fix a reference pixel $\mathbf u$ and let its inverse depth $d$ run from 0 upward. What does the transferred pixel $\mathbf u_m$ do (camera $m$ has moved, not only rotated)?`,
        choices: [
          String.raw`It moves along a straight line, starting at $\pi(K R_{mr}K^{-1}\dot{\mathbf u})$ for $d=0$ and heading toward the epipole $\pi(K\mathbf t_{mr})$`,
          String.raw`It moves along a straight line, starting at the epipole for $d=0$ and heading toward $\pi(K R_{mr}K^{-1}\dot{\mathbf u})$`,
          String.raw`It moves along a curve, because projection divides by depth`,
          String.raw`It stays at one pixel, because all points on a ray project to the same pixel`,
        ],
        answer: 0,
        explain: String.raw`$\mathbf u_m = \pi(\mathbf a + d\mathbf b)$: the homogeneous point moves on a straight line, and $\pi$ maps lines to lines. $d=0$ gives $\pi(\mathbf a)$; large $d$ approaches $\pi(\mathbf b)$. All points on the ray share a pixel in $r$, not in $m$ — that is why a second view reveals depth.` },
      { id: "skew", type: "num",
        gen: (r) => {
          const w = [r.int(-3, 3), r.int(-3, 3), r.nz(3)], x = vi(r, 4);
          const y = la.cross(w, x);
          return {
            q: String.raw`$\boldsymbol\omega = ${vtex(w)}$, $\mathbf x = ${vtex(x)}$. Compute $[\boldsymbol\omega]_\times\mathbf x$.`,
            answer: y, labels: ["1", "2", "3"], tol: 0.001,
            explain: String.raw`$[\boldsymbol\omega]_\times = ${mtex(la.skew(w))}$. Multiplying by $\mathbf x$ gives $${vtex(y)}$, the same as $\boldsymbol\omega\times\mathbf x = (\omega_2x_3-\omega_3x_2,\ \omega_3x_1-\omega_1x_3,\ \omega_1x_2-\omega_2x_1)$.`,
          };
        } },
      { id: "rodrigues", type: "num",
        gen: (r) => {
          const n = r.pick([[0, 0.6, 0.8], [0.6, 0, 0.8], [0.8, 0.6, 0], [0, 0.8, 0.6], [0.6, 0.8, 0], [0.8, 0, -0.6], [0, -0.6, 0.8]]);
          const thd = r.pick([90, 60, 180, 120, 45]), th = thd * deg;
          const w = n.map((k) => +(k * th).toFixed(4));
          const x = vi(r, 3);
          if (x.every((k) => k === 0)) x[0] = 1;
          const y = la.matVec(la.expSO3(w), x);
          const tq = Math.hypot(...w), nn = w.map((k) => k / tq);
          const nx = la.cross(nn, x), nd = la.dot(nn, x);
          return {
            q: String.raw`Rotate $\mathbf x = ${vtex(x)}$ by the rotation vector $\boldsymbol\omega = ${vtex(w)}$ (radians). Find the angle $\theta=|\boldsymbol\omega|$ and axis $\mathbf n$ first, then use Rodrigues. Give $R\mathbf x$ to 2 decimals.`,
            answer: y, labels: ["1", "2", "3"], tol: 0.012,
            explain: String.raw`$\theta = |\boldsymbol\omega| = ${fmt(tq, 4)}$ rad $= ${fmt(tq / deg, 2)}°$, $\mathbf n = \boldsymbol\omega/\theta ≈ ${vtex(nn.map((k) => +k.toFixed(3)))}$. $\cos\theta = ${fmt(Math.cos(tq), 4)}$, $\sin\theta = ${fmt(Math.sin(tq), 4)}$, $\mathbf n\times\mathbf x = ${vtex(nx.map((k) => +k.toFixed(3)))}$, $\mathbf n\cdot\mathbf x = ${fmt(nd, 3)}$. $R\mathbf x = \mathbf x\cos\theta + (\mathbf n\times\mathbf x)\sin\theta + \mathbf n(\mathbf n\cdot\mathbf x)(1-\cos\theta) = ${vtex(y.map((k) => +k.toFixed(3)))}$.`,
          };
        } },
      { id: "twist_first", type: "num",
        gen: (r) => {
          const x = [r.int(-3, 3), r.int(-3, 3), r.int(1, 5)];
          const v = [r.pick([0, 0.01, -0.02, 0.03]), r.pick([0, 0.01, -0.01]), r.pick([0, 0.02, -0.01])];
          const w = [r.pick([0, 0.01, -0.01]), r.pick([0, 0.02, -0.02, 0.01]), r.pick([0, 0.01, -0.03])];
          if (w.every((k) => k === 0)) w[1] = 0.01;
          const wx = la.cross(w, x), y = la.add(la.add(x, v), wx);
          return {
            q: String.raw`Small twist $\psi = (\mathbf v, \boldsymbol\omega)$ with $\mathbf v = ${vtex(v)}$, $\boldsymbol\omega = ${vtex(w)}$. Using the first-order approximation, where does $\mathbf x = ${vtex(x)}$ go under $T(\psi)$? (4 decimals)`,
            answer: y, labels: ["1", "2", "3"], tol: 0.00051,
            explain: String.raw`$T(\psi)\mathbf x \approx \mathbf x + \mathbf v + \boldsymbol\omega\times\mathbf x$. $\boldsymbol\omega\times\mathbf x = ${vtex(wx.map((k) => +k.toFixed(6)))}$, so the result is $${vtex(y.map((k) => +k.toFixed(6)))}$.`,
          };
        } },
      { id: "jac_col", type: "num",
        gen: (r) => {
          const x = [r.int(-3, 3), r.int(-3, 3), r.int(1, 5)], i = r.int(1, 6);
          const e = [0, 0, 0]; e[(i - 1) % 3] = 1;
          const col = i <= 3 ? e : la.cross(e, x);
          const names = ["translation along x", "translation along y", "translation along z", "rotation about x", "rotation about y", "rotation about z"];
          return {
            q: String.raw`For the point $\mathbf x = ${vtex(x)}$, give column ${i} of $\partial\,(T(\psi)\mathbf x)/\partial\psi$ at $\psi = 0$ (how fast the point moves per unit of $\psi_{${i}}$, ${names[i - 1]}; translation first).`,
            answer: col, labels: ["1", "2", "3"], tol: 0.001,
            explain: i <= 3
              ? String.raw`Columns 1–3 are the identity: translating by $\psi_${i}$ moves every point by $\mathbf e_${i} = ${vtex(e)}$.`
              : String.raw`Column ${i} is $\text{gen}_${i}$ applied to $\mathbf x$: $\mathbf e_${i - 3}\times\mathbf x = ${vtex(e)}\times${vtex(x)} = ${vtex(col)}$ (equivalently column ${i - 3} of $-[\mathbf x]_\times$).`,
          };
        } },
      { id: "hat", type: "num",
        gen: (r) => {
          const psi = [r.int(-3, 3), r.int(-3, 3), r.int(-3, 3), r.nz(3), r.nz(3), r.nz(3)];
          const W = la.skew(psi.slice(3));
          const M = [[...W[0], psi[0]], [...W[1], psi[1]], [...W[2], psi[2]], [0, 0, 0, 0]];
          return {
            q: String.raw`The matrix $\sum_i\psi_i\,\text{gen}_i$ equals $${mtex(M)}$. Read off the twist $\psi$ (translation first).`,
            answer: psi, labels: ["$\\psi_1$", "$\\psi_2$", "$\\psi_3$", "$\\psi_4$", "$\\psi_5$", "$\\psi_6$"], tol: 0.001,
            explain: String.raw`The last column is $\mathbf v = ${vtex(psi.slice(0, 3))}$. The top-left block is $[\boldsymbol\omega]_\times$: $\omega_1$ = entry (3,2) $= ${psi[3]}$, $\omega_2$ = entry (1,3) $= ${psi[4]}$, $\omega_3$ = entry (2,1) $= ${psi[5]}$. So $\psi = ${vtex(psi)}$.`,
          };
        } },
      { id: "exp_limit", type: "mc",
        q: String.raw`What happens to $\left(I + [\boldsymbol\omega]_\times/n\right)^n$ as $n$ grows?`,
        choices: [
          String.raw`It converges to $\exp([\boldsymbol\omega]_\times)$: a rotation by angle $|\boldsymbol\omega|$ about the axis $\boldsymbol\omega/|\boldsymbol\omega|$`,
          String.raw`It converges to $I + [\boldsymbol\omega]_\times$, the first-order rotation`,
          String.raw`It converges to a rotation by angle $|\boldsymbol\omega|/n$`,
          String.raw`Its entries grow without bound, since each factor slightly stretches vectors`,
        ],
        answer: 0,
        explain: String.raw`Each factor stretches by $\sqrt{1 + (\theta/n)^2}$, but $n$ of them stretch by $(1+\theta^2/n^2)^{n/2}\to 1$. The limit is the exact rotation, given in closed form by Rodrigues' formula.` },
      { id: "why_psi", type: "multi",
        q: "Why does the tracker solve for a small twist ψ ∈ ℝ⁶ around the current pose rather than for the 12 entries of the 3×4 pose matrix? (select all)",
        choices: [
          "A rotation matrix has 9 entries but only 3 degrees of freedom; changing entries freely breaks RᵀR = I",
          "Any ψ gives a valid pose through the exponential map, so the optimiser can step freely",
          "Near ψ = 0 a point moves by v + ω × x, so the derivatives are simple",
          "The exponential map makes the photometric cost convex, so there is only one minimum",
          "ψ stores the absolute pose of the camera in the world, which avoids drift",
        ],
        answer: [0, 1, 2],
        explain: "ψ is a small <i>update</i> composed onto the current estimate (T ← T·T(ψ)) and reset to 0 each step. It does not make the cost convex (that is why coarse-to-fine and a good initial guess are needed), and it is not an absolute pose." },
    ]);
  },
});
