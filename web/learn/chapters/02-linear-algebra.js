// Chapter 2: vectors, matrices, solving systems, least squares, quadratic
// forms (min of nᵀMn over unit n) and the null-space problem.
DTAM.chapter({
  id: "linalg",
  order: 2,
  title: "Vectors, matrices and least squares",
  subtitle: "The linear algebra DTAM needs, from arrows to null spaces",
  minutes: 60,
  render(root, L) {
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const fx = (v, d = 3) => L.fmt(v, d);
    const snap = (v, s) => Math.round(v / s) * s;
    const deg = (r) => (r * 180) / Math.PI;
    const n2 = (v) => fx(v, 2);

    /** Math-style view (y up) centred in a box of the canvas. */
    const view = (c, span, box) => {
      const b = box || { x: 0, y: 0, w: c.w, h: c.h };
      const s = Math.min(b.w, b.h) / (2 * span);
      const ox = b.x + b.w / 2, oy = b.y + b.h / 2;
      return {
        s, b,
        X: (x) => ox + x * s, Y: (y) => oy - y * s,
        inv: (p) => ({ x: (p.x - ox) / s, y: (oy - p.y) / s }),
        grid(ctx, t, step = 1) {
          const xr = b.w / 2 / s, yr = b.h / 2 / s;
          ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
          for (let k = -Math.ceil(xr); k <= xr; k += step) L.draw.line(ctx, ox + k * s, b.y, ox + k * s, b.y + b.h, t.line, 1);
          for (let k = -Math.ceil(yr); k <= yr; k += step) L.draw.line(ctx, b.x, oy - k * s, b.x + b.w, oy - k * s, t.line, 1);
          L.draw.line(ctx, b.x, oy, b.x + b.w, oy, t.faint, 1.2);
          L.draw.line(ctx, ox, b.y, ox, b.y + b.h, t.faint, 1.2);
          ctx.restore();
        },
      };
    };
    /** HTML matrix table from rows of numbers. */
    const mat = (rows, cls) => `<table class="mat">${rows.map((r, i) => `<tr>${r.map((v, j) => `<td${cls ? ` class="${cls(i, j) || ""}"` : ""}>${typeof v === "number" ? fx(v) : v}</td>`).join("")}</tr>`).join("")}</table>`;
    const texM = (rows) => String.raw`\begin{pmatrix}` + rows.map((r) => r.join(" & ")).join(String.raw` \\ `) + String.raw`\end{pmatrix}`;

    root.insertAdjacentHTML("beforeend", String.raw`
      <style>
        #linalg .mt { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; font: 14px var(--mono); }
        #linalg .mt .op { color: var(--muted); font-size: 18px; }
        #linalg .mt .lab { display: block; font: 12px var(--font); color: var(--muted); margin-bottom: 2px; }
        #linalg table.mat td.hl { background: var(--accent-soft); }
        #linalg table.mat td.hl2 { background: color-mix(in srgb, var(--accent-2) 22%, transparent); }
        #linalg table.mat td.click { cursor: pointer; }
        #linalg table.mat td.sel { outline: 2px solid var(--accent); outline-offset: -2px; font-weight: 700; }
        #linalg .steptext { font: 14.5px/1.45 var(--font); margin: 8px 0 4px; min-height: 2.9em; }
      </style>
      <p>Cameras, poses and optimisers are all written in the language of vectors and matrices. This chapter builds exactly the pieces later chapters use, and nothing more.</p>

      <h3>Vectors</h3>
      <p>A <b>vector</b> is a list of numbers, e.g. $\mathbf a = (3, 1)$ or a 3D point $\mathbf x = (x, y, z)$. Picture it as an arrow from the origin. We write vectors as columns; $\mathbf a^\top$ ("a transposed") is the same numbers as a row.</p>
      <ul>
        <li><b>Add</b>: component by component, $(3,1) + (1,2) = (4,3)$: put the arrows head to tail.</li>
        <li><b>Scale</b>: $2\,(3,1) = (6,2)$: same direction, twice as long ($-1$ flips it).</li>
      </ul>
      <div class="eq-card"><div class="eq-label">Length and dot product</div>
        $$|\mathbf a| = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2}, \qquad \mathbf a \cdot \mathbf b = \mathbf a^\top\mathbf b = a_1 b_1 + a_2 b_2 + \dots + a_n b_n = |\mathbf a|\,|\mathbf b|\cos\theta$$
        <div class="parts">
          <span>$|\mathbf a|$</span><span>length (Pythagoras in $n$ dimensions); $\mathbf a\cdot\mathbf a = |\mathbf a|^2$</span>
          <span>$\mathbf a\cdot\mathbf b$</span><span>multiply matching entries and add: one number</span>
          <span>$\theta$</span><span>angle between the arrows: $\cos\theta = \frac{\mathbf a\cdot\mathbf b}{|\mathbf a||\mathbf b|}$</span>
          <span>$= 0$</span><span>the vectors are <b>perpendicular</b></span>
          <span>$\frac{\mathbf a\cdot\mathbf b}{|\mathbf a|}$</span><span>the (signed) length of $\mathbf b$'s shadow on the line of $\mathbf a$: its <b>projection</b></span>
        </div>
      </div>
      <p><b>Example.</b> $\mathbf a = (3, 4)$, $\mathbf b = (2, -1)$: $|\mathbf a| = 5$, $|\mathbf b| = \sqrt5$, $\mathbf a\cdot\mathbf b = 6 - 4 = 2$, $\cos\theta = 2/(5\sqrt5) = 0.179$, $\theta = 79.7°$. Projection of $\mathbf b$ on $\mathbf a$: $2/5 = 0.4$. A <b>unit vector</b> has length 1: $\mathbf a/|\mathbf a| = (0.6, 0.8)$.</p>
    `);

    // ================================================================ W1 vectors
    {
      const fig = L.figure(root, "<b>Vectors in 2D.</b> Drag the tips of $\\mathbf a$ (blue) and $\\mathbf b$ (orange). <i>Sum</i> shows head-to-tail addition; <i>Dot</i> shows the angle and the projection of $\\mathbf b$ onto $\\mathbf a$ (green).");
      const c = L.canvas(fig.el, { aspect: 0.62 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const a = { x: 3, y: 1 }, b = { x: 1, y: 2.5 };
      let mode = "dot";
      const V = () => view(c, 4.2);
      const info = () => {
        const d = a.x * b.x + a.y * b.y, la = Math.hypot(a.x, a.y), lb = Math.hypot(b.x, b.y);
        const th = la && lb ? deg(Math.acos(clamp(d / (la * lb), -1, 1))) : 0;
        out.html = mode === "sum"
          ? `a = (${a.x}, ${a.y}), b = (${b.x}, ${b.y}) · a + b = <b>(${a.x + b.x}, ${a.y + b.y})</b> · |a| = ${n2(la)}, |b| = ${n2(lb)}, |a + b| = ${n2(Math.hypot(a.x + b.x, a.y + b.y))}`
          : `a·b = ${a.x}·${b.x} + ${a.y}·${b.y} = <b>${fx(d)}</b> · |a| = ${n2(la)}, |b| = ${n2(lb)} · θ = <b>${th.toFixed(1)}°</b> · projection of b on a = a·b/|a| = <b>${la ? n2(d / la) : "–"}</b>`;
      };
      L.drag(c, () => { const v = V(); return [{ x: v.X(a.x), y: v.Y(a.y) }, { x: v.X(b.x), y: v.Y(b.y) }]; }, (i, p) => {
        const q = V().inv(p), tgt = i ? b : a;
        tgt.x = clamp(snap(q.x, 0.5), -6, 6); tgt.y = clamp(snap(q.y, 0.5), -4, 4);
        info();
      });
      L.button(ctl, "Sum", () => { mode = "sum"; info(); c.redraw(); });
      L.button(ctl, "Dot &amp; projection", () => { mode = "dot"; info(); c.redraw(); });
      c.draw = (ctx) => {
        const t = L.theme(), v = V();
        v.grid(ctx, t);
        const O = [v.X(0), v.Y(0)];
        if (mode === "sum") {
          L.draw.line(ctx, v.X(a.x), v.Y(a.y), v.X(a.x + b.x), v.Y(a.y + b.y), t.accent2, 1.5, [5, 4]);
          L.draw.line(ctx, v.X(b.x), v.Y(b.y), v.X(a.x + b.x), v.Y(a.y + b.y), t.accent, 1.5, [5, 4]);
          L.draw.arrow(ctx, ...O, v.X(a.x + b.x), v.Y(a.y + b.y), t.accent3, 3);
          L.draw.text(ctx, "a + b", v.X(a.x + b.x) + 8, v.Y(a.y + b.y) - 6, t.accent3, { size: 13, bold: true });
        } else {
          const aa = a.x * a.x + a.y * a.y;
          if (aa > 0) {
            const k = (a.x * b.x + a.y * b.y) / aa, px = k * a.x, py = k * a.y;
            L.draw.line(ctx, v.X(-a.x * 3), v.Y(-a.y * 3), v.X(a.x * 3), v.Y(a.y * 3), t.faint, 1, [2, 4]);
            L.draw.line(ctx, v.X(b.x), v.Y(b.y), v.X(px), v.Y(py), t.muted, 1.2, [4, 3]);
            L.draw.line(ctx, ...O, v.X(px), v.Y(py), t.accent3, 6);
            const a0 = Math.atan2(-a.y, a.x), a1 = Math.atan2(-b.y, b.x);
            let da = a1 - a0; while (da > Math.PI) da -= 2 * Math.PI; while (da < -Math.PI) da += 2 * Math.PI;
            ctx.save(); ctx.strokeStyle = t.accent4; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(O[0], O[1], 26, a0, a0 + da, da < 0); ctx.stroke(); ctx.restore();
            L.draw.text(ctx, "θ", O[0] + 34 * Math.cos(a0 + da / 2), O[1] + 34 * Math.sin(a0 + da / 2) + 5, t.accent4, { size: 14, align: "center", bold: true });
          }
        }
        L.draw.arrow(ctx, ...O, v.X(a.x), v.Y(a.y), t.accent, 3);
        L.draw.arrow(ctx, ...O, v.X(b.x), v.Y(b.y), t.accent2, 3);
        L.draw.handle(ctx, v.X(a.x), v.Y(a.y), t.accent);
        L.draw.handle(ctx, v.X(b.x), v.Y(b.y), t.accent2);
        L.draw.text(ctx, "a", v.X(a.x) + 12, v.Y(a.y) + 16, t.accent, { size: 14, bold: true });
        L.draw.text(ctx, "b", v.X(b.x) + 12, v.Y(b.y) + 16, t.accent2, { size: 14, bold: true });
      };
      info();
    }

    // ================================================================ cross product
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>The cross product (3D only)</h3>
      <div class="eq-card"><div class="eq-label">Cross product</div>
        $$\mathbf a \times \mathbf b = \begin{pmatrix} a_2 b_3 - a_3 b_2 \\ a_3 b_1 - a_1 b_3 \\ a_1 b_2 - a_2 b_1 \end{pmatrix}$$
        <div class="parts">
          <span>direction</span><span>perpendicular to both $\mathbf a$ and $\mathbf b$; which way is given by the <b>right-hand rule</b> (fingers along $\mathbf a$, curl towards $\mathbf b$, thumb = $\mathbf a\times\mathbf b$)</span>
          <span>length</span><span>$|\mathbf a||\mathbf b|\sin\theta$ = area of the parallelogram they span</span>
          <span>order</span><span>$\mathbf b\times\mathbf a = -\,\mathbf a\times\mathbf b$</span>
          <span>parallel</span><span>$\mathbf a\times\mathbf a = \mathbf 0$ (zero area)</span>
          <span>pattern</span><span>entry 1 uses indices 2,3; entry 2 uses 3,1; entry 3 uses 1,2 ("cycle")</span>
        </div>
      </div>
      <p><b>Example.</b> $(1, 2, 3)\times(4, 5, 6) = (2\cdot6 - 3\cdot5,\ 3\cdot4 - 1\cdot6,\ 1\cdot5 - 2\cdot4) = (-3, 6, -3)$. Check: $(1,2,3)\cdot(-3,6,-3) = -3+12-9 = 0$ ✓.</p>
      <p>DTAM uses it for rotations (chapter 4), epipolar geometry and triangulation (chapter 7), and the tracking Jacobian (chapter 11). The axes $x, y, z$ used here form a <b>right-handed</b> set: $\mathbf e_x \times \mathbf e_y = \mathbf e_z$.</p>
    `);
    {
      const fig = L.figure(root, "<b>Cross product in 3D.</b> Drag the tips of $\\mathbf a$ and $\\mathbf b$ across the floor; lift $\\mathbf b$ with its slider; turn the view. Green arrow: $\\mathbf a\\times\\mathbf b$, drawn at half length. Its length is the shaded area.");
      const c = L.canvas(fig.el, { aspect: 0.66 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const a = [2, 0, 0], b = [0.5, 2, 0];
      let swap = false;
      const el = (25 * Math.PI) / 180;
      const P = () => {
        const az = (saz.value * Math.PI) / 180, s = Math.min(c.w / 8.5, c.h / 6.8), cx = c.w / 2, cy = c.h * 0.6;
        const proj = (x, y, z) => {
          const xr = x * Math.cos(az) - y * Math.sin(az), yr = x * Math.sin(az) + y * Math.cos(az);
          return [cx + s * xr, cy - s * (z * Math.cos(el) + yr * Math.sin(el))];
        };
        const unproj = (X, Y, z) => {
          const xr = (X - cx) / s, yr = ((cy - Y) / s - z * Math.cos(el)) / Math.sin(el);
          return [xr * Math.cos(az) + yr * Math.sin(az), -xr * Math.sin(az) + yr * Math.cos(az)];
        };
        return { proj, unproj };
      };
      const cr = () => (swap ? L.la.cross(b, a) : L.la.cross(a, b));
      const info = () => {
        const [u, w] = swap ? [b, a] : [a, b], x = cr();
        const f = (v) => `(${v.map((q) => fx(q, 2)).join(", ")})`;
        out.html = `${swap ? "b × a" : "a × b"} = (${n2(u[1])}·${n2(w[2])} − ${n2(u[2])}·${n2(w[1])}, ${n2(u[2])}·${n2(w[0])} − ${n2(u[0])}·${n2(w[2])}, ${n2(u[0])}·${n2(w[1])} − ${n2(u[1])}·${n2(w[0])}) = <b>${f(x)}</b> · length = area = <b>${n2(L.la.norm(x))}</b> · a·(a×b) = ${n2(L.la.dot(a, x))}, b·(a×b) = ${n2(L.la.dot(b, x))}`;
      };
      const saz = L.slider(ctl, { label: "view angle", min: 0, max: 360, step: 1, value: 30, fmt: (v) => v + "°", oninput: () => c.redraw() });
      const sbz = L.slider(ctl, { label: "height of $\\mathbf b$", min: -2, max: 2, step: 0.1, value: 0, oninput: (v) => { b[2] = v; info(); c.redraw(); } });
      L.toggle(ctl, "show $\\mathbf b\\times\\mathbf a$ instead", false, (v) => { swap = v; info(); c.redraw(); });
      L.drag(c, () => { const { proj } = P(); return [a, b].map((v) => { const [x, y] = proj(...v); return { x, y }; }); }, (i, p) => {
        const { unproj } = P(), v = i ? b : a;
        const [x, y] = unproj(p.x, p.y, v[2]);
        v[0] = clamp(snap(x, 0.25), -3, 3); v[1] = clamp(snap(y, 0.25), -3, 3);
        info();
      });
      c.draw = (ctx) => {
        const t = L.theme(), { proj } = P();
        const ln = (p, q, col, w, dash) => { const A = proj(...p), B = proj(...q); L.draw.line(ctx, A[0], A[1], B[0], B[1], col, w, dash); };
        const ar = (p, q, col, w) => { const A = proj(...p), B = proj(...q); L.draw.arrow(ctx, A[0], A[1], B[0], B[1], col, w); };
        for (let k = -3; k <= 3; k++) { ln([k, -3, 0], [k, 3, 0], t.line, 1); ln([-3, k, 0], [3, k, 0], t.line, 1); }
        ar([0, 0, 0], [3.4, 0, 0], t.faint, 1.5); ar([0, 0, 0], [0, 3.4, 0], t.faint, 1.5); ar([0, 0, 0], [0, 0, 2.6], t.faint, 1.5);
        for (const [lab, p] of [["x", [3.7, 0, 0]], ["y", [0, 3.7, 0]], ["z", [0, 0, 2.9]]]) { const q = proj(...p); L.draw.text(ctx, lab, q[0], q[1] + 4, t.muted, { size: 13, align: "center" }); }
        // parallelogram
        const pts = [[0, 0, 0], a, L.la.add(a, b), b].map((p) => proj(...p));
        ctx.save(); ctx.fillStyle = t.accent3; ctx.globalAlpha = 0.18; ctx.beginPath();
        pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath(); ctx.fill(); ctx.restore();
        if (Math.abs(b[2]) > 1e-9) ln(b, [b[0], b[1], 0], t.accent2, 1, [3, 3]);
        ar([0, 0, 0], a, t.accent, 3);
        ar([0, 0, 0], b, t.accent2, 3);
        const x = cr();
        if (L.la.norm(x) > 1e-6) ar([0, 0, 0], L.la.scale(x, 0.5), t.accent3, 3.5);
        for (const [v, col, lab] of [[a, t.accent, "a"], [b, t.accent2, "b"]]) {
          const q = proj(...v); L.draw.handle(ctx, q[0], q[1], col);
          L.draw.text(ctx, lab, q[0] + 11, q[1] - 9, col, { size: 14, bold: true });
        }
      };
      info();
    }

    // ================================================================ matrices
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Matrices: machines that transform vectors</h3>
      <p>A <b>matrix</b> is a grid of numbers; an $m\times n$ matrix has $m$ rows and $n$ columns. Multiplying it with a vector gives a new vector:</p>
      <div class="eq-card"><div class="eq-label">Matrix × vector</div>
        $$\begin{pmatrix} a & b \\ c & d \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} a x_1 + b x_2 \\ c x_1 + d x_2 \end{pmatrix} = x_1\begin{pmatrix} a \\ c \end{pmatrix} + x_2\begin{pmatrix} b \\ d \end{pmatrix}$$
        <div class="parts">
          <span>row view</span><span>entry $i$ of the result = (row $i$ of $M$) · $\mathbf x$</span>
          <span>column view</span><span>the result mixes $M$'s columns with weights $x_1, x_2$</span>
          <span>columns</span><span>column 1 is where $(1, 0)$ lands, column 2 is where $(0, 1)$ lands</span>
          <span>sizes</span><span>$(m\times n)$ matrix times $n$-vector gives an $m$-vector</span>
        </div>
      </div>
      <p><b>Example.</b> $\begin{pmatrix} 2 & 1 \\ 0 & 3\end{pmatrix}\begin{pmatrix} 4 \\ -1\end{pmatrix} = \begin{pmatrix} 8 - 1 \\ 0 - 3\end{pmatrix} = \begin{pmatrix} 7 \\ -3\end{pmatrix}$.</p>
    `);
    {
      const fig = L.figure(root, "<b>A 2×2 matrix as a transformation.</b> Drag the blue and orange tips: they are the matrix's columns (where $(1,0)$ and $(0,1)$ go). The whole grid follows. Drag the purple $\\mathbf x$; the solid purple arrow is $M\\mathbf x$. Shaded: the image of the unit square (area = $|\\det M|$).");
      const c = L.canvas(fig.el, { aspect: 0.66 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const c1 = { x: 1.5, y: 0.5 }, c2 = { x: -0.5, y: 1 }, xv = { x: 1, y: 1 };
      const V = () => view(c, 3.3);
      const info = () => {
        const [A, B, C, D] = [c1.x, c2.x, c1.y, c2.y], det = A * D - B * C;
        const mx = A * xv.x + B * xv.y, my = C * xv.x + D * xv.y;
        const inv = Math.abs(det) > 1e-9 ? `M⁻¹ = (1/${n2(det)})·[${n2(D)}, ${n2(-B)}; ${n2(-C)}, ${n2(A)}]` : "<b>singular</b>: no inverse (the plane is squashed onto a line)";
        out.html = `M = [${n2(A)}, ${n2(B)}; ${n2(C)}, ${n2(D)}] · Mx = ${n2(xv.x)}·(${n2(A)}, ${n2(C)}) + ${n2(xv.y)}·(${n2(B)}, ${n2(D)}) = <b>(${n2(mx)}, ${n2(my)})</b> · det M = ${n2(A)}·${n2(D)} − ${n2(B)}·${n2(C)} = <b>${n2(det)}</b> · ${inv}`;
      };
      L.drag(c, () => { const v = V(); return [c1, c2, xv].map((p) => ({ x: v.X(p.x), y: v.Y(p.y) })); }, (i, p) => {
        const q = V().inv(p), tgt = [c1, c2, xv][i];
        tgt.x = clamp(snap(q.x, 0.25), -4, 4); tgt.y = clamp(snap(q.y, 0.25), -3, 3);
        info();
      });
      const preset = (m) => { [c1.x, c2.x, c1.y, c2.y] = m; info(); c.redraw(); };
      const r30 = Math.PI / 6;
      L.button(ctl, "Identity", () => preset([1, 0, 0, 1]));
      L.button(ctl, "Rotate 30°", () => preset([Math.cos(r30), -Math.sin(r30), Math.sin(r30), Math.cos(r30)]));
      L.button(ctl, "Stretch", () => preset([2, 0, 0, 0.5]));
      L.button(ctl, "Shear", () => preset([1, 1, 0, 1]));
      L.button(ctl, "Squash (singular)", () => preset([1, 2, 0.5, 1]));
      c.draw = (ctx) => {
        const t = L.theme(), v = V();
        v.grid(ctx, t);
        const M = (x, y) => [v.X(c1.x * x + c2.x * y), v.Y(c1.y * x + c2.y * y)];
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, c.w, c.h); ctx.clip();
        for (let k = -8; k <= 8; k++) {
          L.draw.line(ctx, ...M(k, -8), ...M(k, 8), t.accent, k ? 0.8 : 1.6);
          L.draw.line(ctx, ...M(-8, k), ...M(8, k), t.accent2, k ? 0.8 : 1.6);
        }
        ctx.restore();
        const det = c1.x * c2.y - c2.x * c1.y;
        ctx.save(); ctx.globalAlpha = 0.22; ctx.fillStyle = det >= 0 ? t.accent3 : t.bad; ctx.beginPath();
        [[0, 0], [1, 0], [1, 1], [0, 1]].forEach(([x, y], i) => { const [X, Y] = M(x, y); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
        ctx.closePath(); ctx.fill(); ctx.restore();
        const O = [v.X(0), v.Y(0)];
        L.draw.arrow(ctx, ...O, v.X(xv.x), v.Y(xv.y), t.accent4, 1.5);
        L.draw.arrow(ctx, ...O, ...M(xv.x, xv.y), t.accent4, 3.5);
        L.draw.arrow(ctx, ...O, v.X(c1.x), v.Y(c1.y), t.accent, 3);
        L.draw.arrow(ctx, ...O, v.X(c2.x), v.Y(c2.y), t.accent2, 3);
        L.draw.handle(ctx, v.X(c1.x), v.Y(c1.y), t.accent);
        L.draw.handle(ctx, v.X(c2.x), v.Y(c2.y), t.accent2);
        L.draw.handle(ctx, v.X(xv.x), v.Y(xv.y), t.accent4);
        const [mx, my] = M(xv.x, xv.y);
        L.draw.text(ctx, "Mx", mx + 8, my - 8, t.accent4, { size: 13, bold: true });
        L.draw.text(ctx, "x", v.X(xv.x) + 11, v.Y(xv.y) + 16, t.accent4, { size: 13, bold: true });
      };
      info();
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <h4>Matrix × matrix, transpose, identity, inverse</h4>
      <div class="eq-card"><div class="eq-label">Matrix product</div>
        $$(AB)_{ij} = (\text{row } i \text{ of } A)\cdot(\text{column } j \text{ of } B) = \sum_k A_{ik}B_{kj}$$
        <div class="parts">
          <span>sizes</span><span>$(m\times n)(n\times p) = (m\times p)$: inner sizes must match</span>
          <span>meaning</span><span>$AB\mathbf x = A(B\mathbf x)$: first apply $B$, then $A$</span>
          <span>order matters</span><span>usually $AB \ne BA$ (rotate-then-stretch ≠ stretch-then-rotate)</span>
        </div>
      </div>
      <ul>
        <li><b>Transpose</b> $A^\top$: rows become columns, $(A^\top)_{ij} = A_{ji}$. Rule: $(AB)^\top = B^\top A^\top$ (order flips). A dot product is a 1×1 matrix product: $\mathbf a\cdot\mathbf b = \mathbf a^\top\mathbf b$.</li>
        <li><b>Identity</b> $I$: 1 on the diagonal, 0 elsewhere. $I\mathbf x = \mathbf x$, $AI = IA = A$.</li>
        <li><b>Inverse</b> $A^{-1}$ undoes $A$: $A^{-1}A = I$. Only square matrices with $\det \ne 0$ have one. $(AB)^{-1} = B^{-1}A^{-1}$.</li>
      </ul>
      <div class="eq-card"><div class="eq-label">2×2 inverse</div>
        $$\begin{pmatrix} a & b \\ c & d\end{pmatrix}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a\end{pmatrix}$$
        <div class="parts">
          <span>$ad - bc$</span><span>the <b>determinant</b> $\det$: the area scale factor of the transformation</span>
          <span>swap, negate</span><span>swap the diagonal, negate the off-diagonal</span>
          <span>$\det = 0$</span><span>the plane is squashed flat; information is lost; no inverse</span>
        </div>
      </div>
      <p><b>Example.</b> $\begin{pmatrix} 2 & 1 \\ 5 & 3\end{pmatrix}^{-1}$: $\det = 6 - 5 = 1$, inverse $\begin{pmatrix} 3 & -1 \\ -5 & 2\end{pmatrix}$. Check the top-left of the product: $2\cdot3 + 1\cdot(-5) = 1$ ✓.</p>
    `);

    // ================================================================ W4 product stepper
    {
      const fig = L.figure(root, "<b>Matrix product, entry by entry.</b> Tap any entry of the result to see which row and column produce it. Switch to $BA$: different size, different numbers.");
      const box = L.el("div", { class: "mt" });
      fig.add(box);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const R = L.rng(21);
      let A, B, rev = false, sel = [0, 0];
      const fresh = () => {
        A = Array.from({ length: 2 }, () => Array.from({ length: 3 }, () => R.int(-3, 4)));
        B = Array.from({ length: 3 }, () => Array.from({ length: 2 }, () => R.int(-3, 4)));
        sel = [0, 0];
      };
      const render = () => {
        const [X, Y, nx, ny] = rev ? [B, A, "B", "A"] : [A, B, "A", "B"];
        const C = L.la.matMul(X, Y);
        const [i, j] = sel;
        box.innerHTML = `<div><span class="lab">${nx} (${X.length}×${X[0].length})</span>${mat(X, (r) => (r === i ? "hl" : ""))}</div><span class="op">×</span>` +
          `<div><span class="lab">${ny} (${Y.length}×${Y[0].length})</span>${mat(Y, (r, cc) => (cc === j ? "hl2" : ""))}</div><span class="op">=</span>` +
          `<div><span class="lab">${nx}${ny} (${C.length}×${C[0].length}) · tap an entry</span>${mat(C, (r, cc) => "click" + (r === i && cc === j ? " sel" : ""))}</div>`;
        box.querySelectorAll("table")[2].querySelectorAll("td").forEach((td, k) => {
          td.addEventListener("click", () => { sel = [Math.floor(k / C[0].length), k % C[0].length]; render(); });
        });
        const row = X[i], col = Y.map((r) => r[j]);
        out.html = `(${nx}${ny})<sub>${i + 1}${j + 1}</sub> = row ${i + 1} of ${nx} · column ${j + 1} of ${ny} = ${row.map((v, k) => `(${v})(${col[k]})`).join(" + ")} = <b>${C[i][j]}</b>`;
      };
      L.button(ctl, "Show AB / BA", () => { rev = !rev; sel = [0, 0]; render(); });
      L.button(ctl, "New matrices", () => { fresh(); render(); });
      fresh(); render();
    }

    // ================================================================ Gaussian elimination
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Solving $A\mathbf x = \mathbf b$: Gaussian elimination</h3>
      <p>Many problems end with "find $\mathbf x$ such that $A\mathbf x = \mathbf b$" ($n$ equations, $n$ unknowns). You could compute $A^{-1}\mathbf b$, but <b>elimination</b> is faster and more accurate. Two allowed moves never change the solution:</p>
      <ul>
        <li>swap two equations (rows);</li>
        <li>subtract a multiple of one row from another.</li>
      </ul>
      <p>Use them to make zeros below the diagonal (column by column), then solve from the last equation upwards (<b>back substitution</b>).</p>
      <p><b>Example.</b> $x + 2y = 5$, $3x + 4y = 11$. Row 2 − 3·row 1: $-2y = -4$, so $y = 2$; then $x = 5 - 2\cdot2 = 1$.</p>
    `);
    {
      const fig = L.figure(root, "<b>Gaussian elimination, step by step</b> (with partial pivoting, as in this guide's <code>la.solve</code>). Press <i>Step</i>. Highlighted: the pivot row (blue) and the row being changed (orange). The last column is $\\mathbf b$.");
      const box = L.el("div", { class: "mt" });
      const txt = L.el("div", { class: "steptext" });
      fig.add(box, txt);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const R = L.rng(5);
      let steps = [], k = 0;
      const build = () => {
        let A, xs;
        do {
          A = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => R.int(-3, 4)));
        } while (Math.abs(A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) < 1);
        xs = [R.int(-3, 3), R.int(-3, 3), R.int(-3, 3)];
        const M = A.map((r) => [...r, L.la.dot(r, xs)]);
        const snapM = () => M.map((r) => r.slice());
        const nm = ["x", "y", "z"];
        steps = [{ M: snapM(), text: `The system as an augmented matrix: each row is one equation, e.g. row 1 means ${M[0][0]}x + ${M[0][1]}y + ${M[0][2]}z = ${M[0][3]}.` }];
        for (let col = 0; col < 3; col++) {
          let p = col;
          for (let r = col + 1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[p][col])) p = r;
          if (p !== col) {
            [M[col], M[p]] = [M[p], M[col]];
            steps.push({ M: snapM(), piv: col, text: `Pivot: swap rows ${col + 1} and ${p + 1} so the entry with the largest size (${fx(M[col][col])}) is on the diagonal. This avoids dividing by tiny numbers.` });
          }
          for (let r = col + 1; r < 3; r++) {
            const f = M[r][col] / M[col][col];
            if (Math.abs(f) < 1e-12) continue;
            for (let q = col; q < 4; q++) M[r][q] -= f * M[col][q];
            M[r][col] = 0;
            steps.push({ M: snapM(), piv: col, ch: r, text: `Row ${r + 1} −= (${fx(f)}) × row ${col + 1}, making a zero in column ${col + 1}.` });
          }
        }
        const x = [0, 0, 0];
        for (let r = 2; r >= 0; r--) {
          let s = M[r][3]; const terms = [];
          for (let q = r + 1; q < 3; q++) { s -= M[r][q] * x[q]; if (Math.abs(M[r][q]) > 1e-12) terms.push(`${fx(M[r][q])}·${fx(x[q])}`); }
          x[r] = s / M[r][r];
          steps.push({ M: snapM(), ch: r, x: x.slice(), done: r, text: `Back substitution, row ${r + 1}: ${nm[r]} = (${fx(M[r][3])}${terms.length ? " − " + terms.join(" − ") : ""}) / ${fx(M[r][r])} = <b>${fx(x[r])}</b>.` });
        }
        steps.push({ M: snapM(), x: x.slice(), done: 0, text: `Solution <b>(x, y, z) = (${x.map((v) => fx(v)).join(", ")})</b>. Check by putting it into the original equations.` });
        k = 0;
      };
      const show = () => {
        const s = steps[k];
        box.innerHTML = mat(s.M, (i, j) => (i === s.piv ? "hl" : i === s.ch ? "hl2" : "")) +
          (s.x ? `<div>${["x", "y", "z"].map((n, i) => (i >= s.done ? `${n} = ${fx(s.x[i])}` : `${n} = ?`)).join("<br>")}</div>` : "");
        txt.innerHTML = `<b>Step ${k} / ${steps.length - 1}.</b> ${s.text}`;
      };
      L.button(ctl, "Step", () => { k = Math.min(k + 1, steps.length - 1); show(); }, "btn primary");
      L.button(ctl, "Back", () => { k = Math.max(k - 1, 0); show(); });
      L.button(ctl, "New system", () => { build(); show(); });
      build(); show();
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <pre><code>function solve(A, b):                   # n×n system A x = b
    M = [A | b]                          # augmented: n rows, n+1 columns
    for col = 0 .. n-1:
        p = the row ≥ col with the largest |M[row][col]|    # pivot
        if |M[p][col]| ≈ 0: return "singular"
        swap rows p and col
        for r = col+1 .. n-1:                 # zeros below the pivot
            f = M[r][col] / M[col][col]
            M[r] = M[r] - f * M[col]
    for r = n-1 down to 0:                    # back substitution
        x[r] = (M[r][n] - sum_{k>r} M[r][k] * x[k]) / M[r][r]
    return x</code></pre>
      <p class="note">If a pivot is zero even after swapping, $\det A = 0$: the equations are dependent (none or infinitely many solutions). For 2×2: $ad - bc = 0$.</p>
    `);

    // ================================================================ least squares
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Too many equations: least squares</h3>
      <p>Measurements are noisy, so we usually have <b>more equations than unknowns</b> and no exact solution. Example: fit a line $y = m x + c$ through 5 points. Each point gives one equation $m x_i + c = y_i$:</p>
      $$\underbrace{\begin{pmatrix} x_1 & 1 \\ x_2 & 1 \\ \vdots & \vdots \\ x_5 & 1\end{pmatrix}}_{A\ (5\times2)} \underbrace{\begin{pmatrix} m \\ c \end{pmatrix}}_{\mathbf x} \approx \underbrace{\begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_5\end{pmatrix}}_{\mathbf b}$$
      <p>Pick $\mathbf x$ to make the <b>residuals</b> $\mathbf r = A\mathbf x - \mathbf b$ small in total: minimise $|\mathbf r|^2 = \sum_i r_i^2$.</p>
      <p><b>One unknown first.</b> Minimise $S(x) = \sum_i (a_i x - b_i)^2 = x^2\sum a_i^2 - 2x\sum a_i b_i + \sum b_i^2$. That is a parabola in $x$; its bottom is where the slope $2x\sum a_i^2 - 2\sum a_ib_i$ is zero: $x = \frac{\mathbf a\cdot\mathbf b}{\mathbf a\cdot\mathbf a}$.</p>
      <p><b>Several unknowns.</b> Do the same for each unknown $x_j$ (slope zero with the others held fixed): $\sum_i A_{ij}\,r_i = 0$ for every $j$, i.e. $A^\top\mathbf r = \mathbf 0$. Substituting $\mathbf r = A\mathbf x - \mathbf b$:</p>
      <div class="eq-card"><div class="eq-label">Normal equations</div>
        $$A^\top A\,\mathbf x = A^\top\mathbf b$$
        <div class="parts">
          <span>$A$ ($N\times n$)</span><span>one row per measurement, one column per unknown ($N &gt; n$)</span>
          <span>$A^\top A$ ($n\times n$)</span><span>small, square and symmetric; solve with elimination</span>
          <span>$A^\top \mathbf b$ ($n$)</span><span>the right-hand side</span>
          <span>meaning</span><span>the best residual is <b>perpendicular</b> to every column of $A$: no change of $\mathbf x$ can shrink it further</span>
        </div>
      </div>
      <p><b>Example.</b> Points $(0,1), (1,3), (2,4)$. Rows of $A$: $(0,1), (1,1), (2,1)$; $\mathbf b = (1,3,4)$.
      $A^\top A = \begin{pmatrix}\sum x_i^2 & \sum x_i \\ \sum x_i & N\end{pmatrix} = \begin{pmatrix} 5 & 3 \\ 3 & 3\end{pmatrix}$, $A^\top\mathbf b = \begin{pmatrix}\sum x_iy_i \\ \sum y_i\end{pmatrix} = \begin{pmatrix} 11 \\ 8\end{pmatrix}$.
      $\det = 6$, so $m = (3\cdot11 - 3\cdot8)/6 = 1.5$, $c = (5\cdot8 - 3\cdot11)/6 = 7/6 \approx 1.167$.</p>
    `);
    {
      const fig = L.figure(root, "<b>Least-squares line fit.</b> Drag the points. Solid line: the normal-equations solution; vertical segments are its residuals. Use the sliders to try to beat it with your own (dashed) line: you can't.");
      const c = L.canvas(fig.el, { aspect: 0.62 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const pts = [[1, 2.2], [3, 3.1], [4.5, 5.4], [6.5, 5.6], [8.5, 8.3]];
      const fit = () => {
        let sxx = 0, sx = 0, sxy = 0, sy = 0; const n = pts.length;
        for (const [x, y] of pts) { sxx += x * x; sx += x; sxy += x * y; sy += y; }
        const det = sxx * n - sx * sx;
        if (Math.abs(det) < 1e-9) return null;
        return { m: (n * sxy - sx * sy) / det, c: (sxx * sy - sx * sxy) / det, sxx, sx, sxy, sy, n };
      };
      const sse = (m, cc) => pts.reduce((s, [x, y]) => s + (m * x + cc - y) ** 2, 0);
      const P = () => L.plot(c, { x0: 0, x1: 10, y0: 0, y1: 10, pad: [10, 10, 24, 30] });
      const info = () => {
        const F = fit();
        if (!F) { out.html = "All points have the same x: the line is not determined (AᵀA is singular)."; return; }
        out.html = `AᵀA = [${n2(F.sxx)}, ${n2(F.sx)}; ${n2(F.sx)}, ${F.n}] · Aᵀb = (${n2(F.sxy)}, ${n2(F.sy)}) · solve → m = <b>${n2(F.m)}</b>, c = <b>${n2(F.c)}</b> · Σr² = <b>${sse(F.m, F.c).toFixed(3)}</b> · your line: Σr² = <b>${sse(sm.value, sc.value).toFixed(3)}</b>`;
      };
      const sm = L.slider(ctl, { label: "your $m$", min: -1, max: 2, step: 0.01, value: 0.5, oninput: () => { info(); c.redraw(); } });
      const sc = L.slider(ctl, { label: "your $c$", min: -3, max: 8, step: 0.05, value: 2, oninput: () => { info(); c.redraw(); } });
      L.drag(c, () => { const p = P(); return pts.map(([x, y]) => ({ x: p.X(x), y: p.Y(y) })); }, (i, q) => {
        const p = P(); pts[i] = [clamp(p.invX(q.x), 0, 10), clamp(p.invY(q.y), 0, 10)]; info();
      });
      c.draw = (ctx) => {
        const t = L.theme(), p = P();
        p.axes(ctx, { xticks: 5, yticks: 5, fmt: (v) => Math.round(v) });
        ctx.save(); ctx.beginPath(); ctx.rect(p.X(0), p.Y(10), p.X(10) - p.X(0), p.Y(0) - p.Y(10)); ctx.clip();
        L.draw.line(ctx, p.X(0), p.Y(sc.value), p.X(10), p.Y(sm.value * 10 + sc.value), t.accent2, 2, [6, 4]);
        const F = fit();
        if (F) {
          for (const [x, y] of pts) L.draw.line(ctx, p.X(x), p.Y(y), p.X(x), p.Y(F.m * x + F.c), t.bad, 2);
          L.draw.line(ctx, p.X(0), p.Y(F.c), p.X(10), p.Y(F.m * 10 + F.c), t.accent, 2.5);
        }
        ctx.restore();
        for (const [x, y] of pts) L.draw.handle(ctx, p.X(x), p.Y(y), t.accent3);
      };
      info();
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>In practice we never build the tall matrix $A$. Each measurement row $\mathbf a_i$ just <b>adds</b> its contribution, which is how the GPU does it later (chapter 11: every pixel adds its own term):</p>
      <pre><code>H = zeros(n, n);  g = zeros(n)          # will hold AᵀA and Aᵀb
for each measurement i with row a_i and target b_i:
    for j, k in 0..n-1:  H[j][k] += a_i[j] * a_i[k]    # H += a_i a_iᵀ
    for j in 0..n-1:     g[j]    += a_i[j] * b_i       # g += a_i b_i
x = solve(H, g)</code></pre>
      <p class="note">$A^\top A$ is symmetric with $\mathbf x^\top A^\top A\mathbf x = |A\mathbf x|^2 \ge 0$. For such matrices there is a faster elimination variant called <b>Cholesky</b> factorisation; this implementation uses it (<code>h.cholesky().solve(g)</code> in sfm.rs). You don't need its details: it gives the same $\mathbf x$.</p>
    `);

    // ================================================================ quadratic forms
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Symmetric matrices and quadratic forms</h3>
      <p>A matrix is <b>symmetric</b> if $M^\top = M$. For 2×2: $M = \begin{pmatrix} a & b \\ b & c\end{pmatrix}$. Examples: $A^\top A$ above, and the corner detector's matrix in chapter 6.</p>
      <div class="eq-card"><div class="eq-label">Quadratic form</div>
        $$\mathbf n^\top M\,\mathbf n = a\,n_1^2 + 2b\,n_1n_2 + c\,n_2^2$$
        <div class="parts">
          <span>$\mathbf n$</span><span>a direction; below always a <b>unit</b> vector, $\mathbf n = (\cos\theta, \sin\theta)$</span>
          <span>result</span><span>one number: how strongly $M$ "responds" in direction $\mathbf n$</span>
          <span>$M = A^\top A$</span><span>then $\mathbf n^\top M\mathbf n = |A\mathbf n|^2$: how much $A$ stretches $\mathbf n$ (squared)</span>
        </div>
      </div>
      <p><b>Example.</b> $M = \begin{pmatrix} 3 & 1 \\ 1 & 1\end{pmatrix}$, $\mathbf n = (0.6, 0.8)$: $3(0.36) + 2(1)(0.48) + 1(0.64) = 1.08 + 0.96 + 0.64 = 2.68$.</p>
      <p><b>Why we care.</b> Chapter 6 builds, for a small image patch, $M = \sum \begin{pmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2\end{pmatrix}$. Then $\mathbf n^\top M\mathbf n = \sum (I_x n_1 + I_y n_2)^2$ = how much the patch changes when shifted in direction $\mathbf n$. A good corner changes a lot in <b>every</b> direction, so we want the <b>smallest</b> value over all directions to be large.</p>
      <h4>The weakest direction, in closed form</h4>
      <p>Put $\mathbf n = (\cos\theta, \sin\theta)$ and use $\cos^2\theta = \frac{1+\cos2\theta}{2}$, $\sin^2\theta = \frac{1-\cos2\theta}{2}$, $2\sin\theta\cos\theta = \sin2\theta$:</p>
      $$q(\theta) = \mathbf n^\top M\mathbf n = \frac{a+c}{2} + \frac{a-c}{2}\cos2\theta + b\sin2\theta.$$
      <p>A sum $p\cos\varphi + q\sin\varphi$ swings between $-\sqrt{p^2+q^2}$ and $+\sqrt{p^2+q^2}$ (it equals $\sqrt{p^2+q^2}\cos(\varphi - \alpha)$ for some angle $\alpha$). So:</p>
      <div class="eq-card"><div class="eq-label">Min and max of nᵀMn over unit n (2×2 symmetric)</div>
        $$\lambda_{\min} = \frac{a+c}{2} - \sqrt{\Big(\frac{a-c}{2}\Big)^2 + b^2}, \qquad \lambda_{\max} = \frac{a+c}{2} + \sqrt{\Big(\frac{a-c}{2}\Big)^2 + b^2}$$
        <div class="parts">
          <span>$\frac{a+c}{2}$</span><span>the average response over all directions</span>
          <span>$\sqrt{\cdots}$</span><span>how unequal the directions are (0 means the same in every direction)</span>
          <span>directions</span><span>let $\varphi$ = angle of the point $\big(\frac{a-c}{2}, b\big)$ (i.e. $\operatorname{atan2}(2b, a-c)$). Strongest: $\theta_{\max} = \varphi/2$. Weakest: $\theta_{\min} = \varphi/2 + 90°$. They are perpendicular.</span>
          <span>name</span><span>$\lambda_{\min}, \lambda_{\max}$ are called the <b>eigenvalues</b> of $M$, and the two directions its <b>eigenvectors</b></span>
        </div>
      </div>
      <p><b>Example.</b> $M = \begin{pmatrix} 3 & 1 \\ 1 & 1\end{pmatrix}$: $\frac{a+c}{2} = 2$, $\sqrt{1^2 + 1^2} = 1.414$, so $\lambda_{\min} = 0.586$, $\lambda_{\max} = 3.414$. $\varphi = \operatorname{atan2}(2, 2) = 45°$: strongest at $22.5°$, weakest at $112.5°$. Check $\mathbf n = (\cos112.5°, \sin112.5°) = (-0.383, 0.924)$: $3(0.146) + 2(-0.354) + 0.854 = 0.586$ ✓.</p>
      <p class="note">The corner shaders compute exactly this: <code>0.5 * (tr - sqrt((gxx - gyy)² + 4 gxy²))</code> with <code>tr = gxx + gyy</code> is $\frac{a+c}{2} - \sqrt{(\frac{a-c}{2})^2 + b^2}$ written differently.</p>
    `);
    {
      const fig = L.figure(root, "<b>Quadratic form explorer.</b> Set $M$ with the sliders; drag the purple direction $\\mathbf n$ around the unit circle. Left: level curves of $\\mathbf x^\\top M\\mathbf x$ (blue: positive, orange: negative) with the weakest (green) and strongest (dashed) directions from the formula. Right/below: $q(\\theta)$ for every direction; the formula's $\\lambda_{\\min}$ and $\\lambda_{\\max}$ are its lowest and highest points.");
      const c = L.canvas(fig.el, { aspect: 0.95, maxHeight: 480 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      let th = 0.6;
      const upd = () => { info(); c.redraw(); };
      const sa = L.slider(ctl, { label: "$a$", min: -3, max: 4, step: 0.1, value: 3, oninput: upd });
      const sb = L.slider(ctl, { label: "$b$", min: -3, max: 3, step: 0.1, value: 1, oninput: upd });
      const sc = L.slider(ctl, { label: "$c$", min: -3, max: 4, step: 0.1, value: 1, oninput: upd });
      const eig = () => {
        const a = sa.value, b = sb.value, cc = sc.value;
        const m = (a + cc) / 2, R = Math.hypot((a - cc) / 2, b);
        const phi = Math.atan2(2 * b, a - cc);
        return { a, b, cc, lo: m - R, hi: m + R, tmax: phi / 2, tmin: phi / 2 + Math.PI / 2, R };
      };
      const q = (t) => { const a = sa.value, b = sb.value, cc = sc.value, x = Math.cos(t), y = Math.sin(t); return a * x * x + 2 * b * x * y + cc * y * y; };
      const lay = () => {
        if (c.w >= 560) { const s = Math.min(c.h, c.w * 0.55); return { left: { x: 0, y: 0, w: s, h: s }, right: { x: s + 10, y: 0, w: c.w - s - 10, h: c.h } }; }
        const s = c.h * 0.6;
        return { left: { x: (c.w - s) / 2, y: 0, w: s, h: s }, right: { x: 0, y: s + 6, w: c.w, h: c.h - s - 6 } };
      };
      const info = () => {
        const E = eig(), n = [Math.cos(th), Math.sin(th)];
        const pd = E.lo > 1e-9 ? "positive-definite: a bowl (every direction goes up)" : E.lo > -1e-9 ? "semi-definite: a flat valley in the weakest direction" : "not positive-definite: a saddle (some directions go down)";
        out.html = `n = (${n2(n[0])}, ${n2(n[1])}), θ = ${deg(th).toFixed(0)}° · nᵀMn = ${n2(E.a)}·${n2(n[0] * n[0])} + 2·${n2(E.b)}·${n2(n[0] * n[1])} + ${n2(E.cc)}·${n2(n[1] * n[1])} = <b>${q(th).toFixed(3)}</b> · λ<sub>min</sub> = <b>${E.lo.toFixed(3)}</b> at ${((deg(E.tmin) % 180) + 180) % 180 | 0}°, λ<sub>max</sub> = <b>${E.hi.toFixed(3)}</b> · ${pd}`;
      };
      L.drag(c, () => { const { left } = lay(), v = view(c, 1.6, left); return [{ x: v.X(Math.cos(th)), y: v.Y(Math.sin(th)) }]; }, (_, p) => {
        const { left } = lay(), v = view(c, 1.6, left), w = v.inv(p);
        th = Math.atan2(w.y, w.x); info();
      }, { radius: 22 });
      c.draw = (ctx) => {
        const t = L.theme(), { left, right } = lay(), v = view(c, 1.6, left), E = eig();
        v.grid(ctx, t, 0.5);
        // level curves x = r n(θ), r = sqrt(k / q(θ))
        ctx.save(); ctx.beginPath(); ctx.rect(left.x, left.y, left.w, left.h); ctx.clip();
        for (const k of [0.25, 0.5, 1, 2, 4, -0.25, -0.5, -1, -2, -4]) {
          let seg = [];
          const flush = () => { if (seg.length > 1) L.draw.path(ctx, seg, k > 0 ? t.accent : t.accent2, k === 1 || k === -1 ? 2 : 1); seg = []; };
          for (let i = 0; i <= 360; i++) {
            const a = (i / 360) * 2 * Math.PI, qq = q(a);
            if (qq * k <= 1e-6) { flush(); continue; }
            const r = Math.sqrt(k / qq);
            if (r > 4) { flush(); continue; }
            seg.push([v.X(r * Math.cos(a)), v.Y(r * Math.sin(a))]);
          }
          flush();
        }
        // unit circle and eigen-directions
        ctx.strokeStyle = t.muted; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.arc(v.X(0), v.Y(0), v.s, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
        const dir = (a, col, dash) => L.draw.line(ctx, v.X(-2 * Math.cos(a)), v.Y(-2 * Math.sin(a)), v.X(2 * Math.cos(a)), v.Y(2 * Math.sin(a)), col, 2, dash);
        if (E.R > 1e-9) { dir(E.tmax, t.muted, [6, 4]); dir(E.tmin, t.accent3); }
        ctx.restore();
        L.draw.arrow(ctx, v.X(0), v.Y(0), v.X(Math.cos(th)), v.Y(Math.sin(th)), t.accent4, 3);
        L.draw.handle(ctx, v.X(Math.cos(th)), v.Y(Math.sin(th)), t.accent4);
        // q(θ) plot
        const lo = Math.min(E.lo, 0) - 0.3, hi = Math.max(E.hi, 0) + 0.3;
        const pad = [14, 8, 26, 34];
        const X = (a) => right.x + pad[3] + (a / 180) * (right.w - pad[1] - pad[3]);
        const Y = (y) => right.y + right.h - pad[2] - ((y - lo) / (hi - lo)) * (right.h - pad[0] - pad[2]);
        for (const d of [0, 45, 90, 135, 180]) { L.draw.line(ctx, X(d), Y(hi), X(d), Y(lo), t.line, 1); L.draw.text(ctx, d + "°", X(d), right.y + right.h - 10, t.faint, { size: 11, align: "center" }); }
        L.draw.line(ctx, X(0), Y(0), X(180), Y(0), t.faint, 1.2);
        L.draw.line(ctx, X(0), Y(E.lo), X(180), Y(E.lo), t.accent3, 1.5, [5, 3]);
        L.draw.line(ctx, X(0), Y(E.hi), X(180), Y(E.hi), t.muted, 1.5, [5, 3]);
        L.draw.text(ctx, "λmin " + E.lo.toFixed(2), X(180), Y(E.lo) + 14, t.accent3, { size: 11, align: "right" });
        L.draw.text(ctx, "λmax " + E.hi.toFixed(2), X(180), Y(E.hi) - 5, t.muted, { size: 11, align: "right" });
        L.draw.text(ctx, "0", X(0) - 5, Y(0) + 4, t.faint, { size: 11, align: "right" });
        const pts = [];
        for (let i = 0; i <= 180; i++) pts.push([X(i), Y(q((i * Math.PI) / 180))]);
        L.draw.path(ctx, pts, t.accent4, 2.5);
        const thd = ((deg(th) % 180) + 180) % 180;
        L.draw.dot(ctx, X(thd), Y(q(th)), 6, t.accent4, t.panel);
        L.draw.text(ctx, "q(θ) = nᵀMn", X(90), right.y + 12, t.muted, { size: 12, align: "center" });
      };
      info();
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <div class="key">$M$ is <b>positive-definite</b> when $\mathbf n^\top M\mathbf n &gt; 0$ for every direction, i.e. $\lambda_{\min} &gt; 0$. Then $f(\mathbf x) = \mathbf x^\top M\mathbf x$ is a <b>bowl</b> with a single lowest point, and the level curves are ellipses. Least-squares problems have $M = A^\top A$ (always $\ge 0$), and a bowl means a unique best answer.</div>
    `);

    // ================================================================ null space
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>The null-space problem: $A\mathbf x \approx \mathbf 0$ with $|\mathbf x| = 1$</h3>
      <p>Some problems (the 8-point algorithm and triangulation in chapter 7) give equations with <b>zero</b> on the right: $A\mathbf x = \mathbf 0$. Then $\mathbf x = \mathbf 0$ always works and is useless, and if $\mathbf x$ works so does $5\mathbf x$. Only the <b>direction</b> of $\mathbf x$ matters, so we fix its length:</p>
      <div class="eq-card"><div class="eq-label">Homogeneous least squares</div>
        $$\min_{|\mathbf x| = 1} |A\mathbf x|^2 = \min_{|\mathbf x| = 1} \mathbf x^\top (A^\top A)\,\mathbf x$$
        <div class="parts">
          <span>$|\mathbf x| = 1$</span><span>rules out $\mathbf x = \mathbf 0$ and removes the free scale</span>
          <span>$|A\mathbf x|^2$</span><span>sum of squared equation errors (0 if all are satisfied exactly)</span>
          <span>$\mathbf x^\top A^\top A\,\mathbf x$</span><span>a quadratic form! The answer is the <b>weakest direction</b> of $A^\top A$, and the minimum value is $\lambda_{\min}$</span>
          <span>$\pm\mathbf x$</span><span>both are equally good; pick a sign by convention</span>
        </div>
      </div>
      <p>With noise-free data some $\mathbf x$ gives exactly $A\mathbf x = \mathbf 0$ (the <b>null space</b> of $A$) and $\lambda_{\min} = 0$. With noise, it is the best compromise.</p>
      <p><b>Small exact case.</b> 3 unknowns, 2 equations with rows $\mathbf r_1, \mathbf r_2$: $\mathbf x$ must be perpendicular to both rows, so $\mathbf x \propto \mathbf r_1\times\mathbf r_2$. E.g. $\mathbf r_1 = (1,2,3)$, $\mathbf r_2 = (0,1,-1)$: $\mathbf r_1\times\mathbf r_2 = (-5, 1, 1)$, length $\sqrt{27}$, so $\mathbf x = (-0.962, 0.192, 0.192)$.</p>
    `);
    {
      const fig = L.figure(root, "<b>Fit a line through the origin as a null-space problem.</b> Each point $\\mathbf p_i$ is a row of $A$; a line with unit normal $\\mathbf n$ fits when $\\mathbf p_i\\cdot\\mathbf n = 0$. Drag the points or the purple normal. The bottom plot is $|A\\mathbf n|^2$ for every direction; green is the closed-form weakest direction of $A^\\top A$.");
      const c = L.canvas(fig.el, { aspect: 0.9, maxHeight: 520 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      let P = [[-2.2, -1.1], [-1.2, -0.3], [0.6, 0.5], [1.5, 0.5], [2.4, 1.4]];
      let th = 1.2;
      const H = () => {
        let a = 0, b = 0, cc = 0;
        for (const [x, y] of P) { a += x * x; b += x * y; cc += y * y; }
        const R = Math.hypot((a - cc) / 2, b);
        return { a, b, cc, lo: (a + cc) / 2 - R, tmin: Math.atan2(2 * b, a - cc) / 2 + Math.PI / 2 };
      };
      const cost = (t) => P.reduce((s, [x, y]) => s + (x * Math.cos(t) + y * Math.sin(t)) ** 2, 0);
      const lay = () => { const top = c.h * 0.68; return { top, V: view(c, 3.2, { x: 0, y: 0, w: c.w, h: top }) }; };
      const info = () => {
        const h = H();
        out.html = `AᵀA = Σ p pᵀ = [${n2(h.a)}, ${n2(h.b)}; ${n2(h.b)}, ${n2(h.cc)}] · your n = (${n2(Math.cos(th))}, ${n2(Math.sin(th))}): |An|² = <b>${cost(th).toFixed(3)}</b> · best n = (${n2(Math.cos(h.tmin))}, ${n2(Math.sin(h.tmin))}): |An|² = λ<sub>min</sub> = <b>${h.lo.toFixed(3)}</b>`;
      };
      L.drag(c, () => { const { V } = lay(); return [...P.map(([x, y]) => ({ x: V.X(x), y: V.Y(y) })), { x: V.X(1.6 * Math.cos(th)), y: V.Y(1.6 * Math.sin(th)) }]; }, (i, p) => {
        const { V, top } = lay(), w = V.inv({ x: p.x, y: Math.min(p.y, top) });
        if (i < P.length) P[i] = [clamp(w.x, -4, 4), clamp(w.y, -3, 3)];
        else th = Math.atan2(w.y, w.x);
        info();
      });
      L.button(ctl, "Snap n to the best", () => { th = H().tmin; info(); c.redraw(); });
      L.button(ctl, "Make exactly collinear", () => {
        const d = [Math.cos(th + Math.PI / 2), Math.sin(th + Math.PI / 2)];
        P = P.map(([x, y]) => { const s = x * d[0] + y * d[1]; return [s * d[0], s * d[1]]; });
        info(); c.redraw();
      });
      L.button(ctl, "Scatter", () => { const r = L.rng((Math.random() * 1e9) | 0); const a = r.float(0, Math.PI); P = P.map(() => { const s = r.float(-3, 3); return [clamp(s * Math.cos(a) + r.float(-0.6, 0.6), -4, 4), clamp(s * Math.sin(a) + r.float(-0.6, 0.6), -3, 3)]; }); info(); c.redraw(); });
      c.draw = (ctx) => {
        const t = L.theme(), { top, V } = lay(), h = H();
        V.grid(ctx, t);
        const ln = (a, col, w, dash) => { const d = [-Math.sin(a), Math.cos(a)]; L.draw.line(ctx, V.X(-9 * d[0]), V.Y(-9 * d[1]), V.X(9 * d[0]), V.Y(9 * d[1]), col, w, dash); };
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, c.w, top); ctx.clip();
        ln(h.tmin, t.accent3, 2, [6, 4]);
        ln(th, t.accent4, 2.5);
        const n = [Math.cos(th), Math.sin(th)];
        for (const [x, y] of P) { const s = x * n[0] + y * n[1]; L.draw.line(ctx, V.X(x), V.Y(y), V.X(x - s * n[0]), V.Y(y - s * n[1]), t.bad, 2); }
        ctx.restore();
        L.draw.arrow(ctx, V.X(0), V.Y(0), V.X(1.6 * n[0]), V.Y(1.6 * n[1]), t.accent4, 3);
        L.draw.handle(ctx, V.X(1.6 * n[0]), V.Y(1.6 * n[1]), t.accent4);
        L.draw.text(ctx, "n", V.X(1.6 * n[0]) + 11, V.Y(1.6 * n[1]) - 9, t.accent4, { size: 14, bold: true });
        for (const [x, y] of P) L.draw.handle(ctx, V.X(x), V.Y(y), t.accent);
        // |An|^2 vs angle
        const pad = [6, 8, 22, 34], y0 = top + 6, hh = c.h - y0;
        const pts = [], N = 180; let mx = 0;
        for (let i = 0; i <= N; i++) { const v = cost((i / N) * Math.PI); pts.push(v); mx = Math.max(mx, v); }
        mx = mx || 1;
        const X = (d) => pad[3] + (d / 180) * (c.w - pad[1] - pad[3]), Y = (v) => y0 + hh - pad[2] - (v / mx) * (hh - pad[0] - pad[2]);
        for (const d of [0, 45, 90, 135, 180]) { L.draw.line(ctx, X(d), Y(0), X(d), Y(mx), t.line, 1); L.draw.text(ctx, d + "°", X(d), c.h - 6, t.faint, { size: 11, align: "center" }); }
        L.draw.line(ctx, X(0), Y(0), X(180), Y(0), t.faint, 1);
        L.draw.path(ctx, pts.map((v, i) => [X((i / N) * 180), Y(v)]), t.accent4, 2);
        const md = ((deg(h.tmin) % 180) + 180) % 180, cd = ((deg(th) % 180) + 180) % 180;
        L.draw.line(ctx, X(md), Y(0), X(md), Y(mx), t.accent3, 1.5, [4, 3]);
        L.draw.dot(ctx, X(cd), Y(cost(th)), 5, t.accent4, t.panel);
        L.draw.text(ctx, "|An|² vs direction of n", X(0) + 4, y0 + 10, t.muted, { size: 11 });
      };
      info();
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>For 2 unknowns the closed form above solves it. For 9 unknowns (chapter 7's fundamental matrix) we call a <b>library routine</b>:</p>
      <ul>
        <li><b>Symmetric eigen-decomposition</b> of $A^\top A$: returns every direction's $\lambda$ with its unit vector; take the one with the smallest $\lambda$. This implementation does exactly this (<code>null_vector</code> in calib.rs, <code>triangulate</code> in sfm.rs, via nalgebra's <code>SymmetricEigen</code>).</li>
        <li><b>SVD</b> (singular value decomposition), $A = U\Sigma V^\top$: the textbook route. The answer is the <b>last column of $V$</b> (the one with the smallest singular value $\sigma$, where $\sigma^2 = \lambda$). Same answer, slightly more accurate numerically.</li>
      </ul>
      <pre><code>H = zeros(n, n)
for each equation row a_i:  H += a_i a_iᵀ         # H = AᵀA, built row by row
(lambdas, vectors) = symmetric_eigen(H)            # library routine
x = vectors[:, argmin(lambdas)]                    # unit length; −x is just as good</code></pre>
      <div class="key">Two kinds of least squares, both built on $A^\top A$:
      <ul><li>$A\mathbf x \approx \mathbf b$ (right side not zero): solve the <b>normal equations</b> $A^\top A\mathbf x = A^\top\mathbf b$.</li>
      <li>$A\mathbf x \approx \mathbf 0$ with $|\mathbf x| = 1$: take the <b>weakest direction</b> of $A^\top A$.</li></ul></div>
    `);

    // ================================================================ quiz
    const vec = (v) => `(${v.join(", ")})`;
    L.quiz(root, "linalg", [
      { id: "dot", type: "num", gen: (r) => {
        const pick = () => { let v; do { v = [r.int(-4, 4), r.int(-4, 4), r.int(-4, 4)]; } while (!v.some((x) => x)); return v; };
        const a = pick(), b = pick();
        const d = L.la.dot(a, b), la = L.la.norm(a), lb = L.la.norm(b), th = deg(Math.acos(clamp(d / (la * lb), -1, 1)));
        return { q: String.raw`$\mathbf a = ${vec(a)}$, $\mathbf b = ${vec(b)}$. Find $\mathbf a\cdot\mathbf b$, the angle $\theta$ between them in degrees (1 decimal), and the projection length of $\mathbf b$ onto $\mathbf a$, $\mathbf a\cdot\mathbf b/|\mathbf a|$ (2 decimals).`,
          answer: [d, th, d / la], labels: ["$\\mathbf a\\cdot\\mathbf b$", "$\\theta$ (°)", "projection"], tol: 0.06,
          explain: String.raw`$\mathbf a\cdot\mathbf b = ${a.map((x, i) => `(${x})(${b[i]})`).join(" + ")} = ${d}$. $|\mathbf a| = \sqrt{${L.la.dot(a, a)}} = ${fx(la, 4)}$, $|\mathbf b| = \sqrt{${L.la.dot(b, b)}} = ${fx(lb, 4)}$. $\cos\theta = ${d}/(${fx(la, 4)}\cdot${fx(lb, 4)}) = ${fx(d / (la * lb), 4)}$, $\theta = ${fx(th, 2)}°$. Projection $= ${d}/${fx(la, 4)} = ${fx(d / la, 4)}$.` };
      } },
      { id: "cross", type: "num", gen: (r) => {
        const a = [r.int(-5, 5), r.int(-5, 5), r.int(-5, 5)], b = [r.int(-5, 5), r.int(-5, 5), r.int(-5, 5)];
        const x = L.la.cross(a, b);
        return { q: String.raw`Compute $\mathbf a\times\mathbf b$ for $\mathbf a = ${vec(a)}$, $\mathbf b = ${vec(b)}$.`,
          answer: x, labels: ["$x$", "$y$", "$z$"],
          explain: String.raw`$(a_2b_3 - a_3b_2,\ a_3b_1 - a_1b_3,\ a_1b_2 - a_2b_1) = ((${a[1]})(${b[2]}) - (${a[2]})(${b[1]}),\ (${a[2]})(${b[0]}) - (${a[0]})(${b[2]}),\ (${a[0]})(${b[1]}) - (${a[1]})(${b[0]})) = ${vec(x)}$. Check: $\mathbf a\cdot(\mathbf a\times\mathbf b) = ${L.la.dot(a, x)}$.` };
      } },
      { id: "crossprops", type: "multi",
        q: "Which statements about the cross product of 3D vectors are always true? (select all)",
        choices: ["$\\mathbf a\\times\\mathbf b$ is perpendicular to both $\\mathbf a$ and $\\mathbf b$",
          "$\\mathbf b\\times\\mathbf a = -\\,\\mathbf a\\times\\mathbf b$",
          "$|\\mathbf a\\times\\mathbf b|$ is the area of the parallelogram spanned by $\\mathbf a$ and $\\mathbf b$",
          "$\\mathbf a\\times\\mathbf b = \\mathbf b\\times\\mathbf a$",
          "$\\mathbf a\\times\\mathbf b$ is largest when $\\mathbf a$ and $\\mathbf b$ are parallel",
          "$\\mathbf a\\times\\mathbf b$ is a number, like the dot product"],
        answer: [0, 1, 2],
        explain: "The cross product is a vector perpendicular to both, flips sign when the order is swapped, and has length $|\\mathbf a||\\mathbf b|\\sin\\theta$: zero for parallel vectors, largest for perpendicular ones." },
      { id: "matvec", type: "num", gen: (r) => {
        const M = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => r.int(-3, 4))), x = [r.int(-3, 3), r.int(-3, 3), r.int(-3, 3)];
        const y = L.la.matVec(M, x);
        return { q: String.raw`Compute $M\mathbf x$ for $M = ${texM(M)}$ and $\mathbf x = ${vec(x)}$.`,
          answer: y, labels: ["1", "2", "3"],
          explain: String.raw`Each entry is a row of $M$ dotted with $\mathbf x$: ` + M.map((row, i) => `$${row.map((v, k) => `(${v})(${x[k]})`).join(" + ")} = ${y[i]}$`).join("; ") + "." };
      } },
      { id: "matmul", type: "num", gen: (r) => {
        const A = [[r.int(-4, 4), r.int(-4, 4)], [r.int(-4, 4), r.int(-4, 4)]], B = [[r.int(-4, 4), r.int(-4, 4)], [r.int(-4, 4), r.int(-4, 4)]];
        const C = L.la.matMul(A, B);
        return { q: String.raw`Compute $AB$ for $A = ${texM(A)}$, $B = ${texM(B)}$.`,
          answer: C.flat(), labels: ["$(AB)_{11}$", "$(AB)_{12}$", "$(AB)_{21}$", "$(AB)_{22}$"],
          explain: String.raw`$(AB)_{ij}$ = row $i$ of $A$ · column $j$ of $B$. ` + [0, 1].flatMap((i) => [0, 1].map((j) => String.raw`$(AB)_{${i + 1}${j + 1}} = (${A[i][0]})(${B[0][j]}) + (${A[i][1]})(${B[1][j]}) = ${C[i][j]}$`)).join(", ") + "." };
      } },
      { id: "algebra", type: "multi",
        q: "For square matrices $A$, $B$ (invertible where needed), which are <b>always</b> true? (select all)",
        choices: ["$(AB)^\\top = B^\\top A^\\top$", "$AI = A$", "$(AB)^{-1} = B^{-1}A^{-1}$", "$AB = BA$", "$(AB)^\\top = A^\\top B^\\top$", "$\\det A = 0$ means $A^{-1}$ exists"],
        answer: [0, 1, 2],
        explain: "Transposing or inverting a product reverses the order. Matrix products generally do not commute. $\\det A = 0$ means $A$ squashes space flat, so it has <i>no</i> inverse." },
      { id: "inv", type: "num", gen: (r) => {
        let a, b, c, d, det;
        do { a = r.int(-5, 5); b = r.int(-5, 5); c = r.int(-5, 5); d = r.int(-5, 5); det = a * d - b * c; } while (![1, 2, 4, 5, -1, -2, -4, -5].includes(det));
        const inv = [d / det, -b / det, -c / det, a / det];
        return { q: String.raw`Invert $M = ${texM([[a, b], [c, d]])}$. (Fractions like 3/4 are accepted.)`,
          answer: inv, labels: ["$(M^{-1})_{11}$", "$(M^{-1})_{12}$", "$(M^{-1})_{21}$", "$(M^{-1})_{22}$"], tol: 0.001,
          explain: String.raw`$\det M = (${a})(${d}) - (${b})(${c}) = ${det}$. $M^{-1} = \frac{1}{${det}}${texM([[d, -b], [-c, a]])} = ${texM([inv.slice(0, 2).map((v) => fx(v, 4)), inv.slice(2).map((v) => fx(v, 4))])}$.` };
      } },
      { id: "singular", type: "num", gen: (r) => {
        const p = r.nz(6), q = r.nz(6), s = r.nz(6);
        const k = (p * s) / q;
        return { q: String.raw`For which value of $k$ does $M = ${texM([[p, "k"], [q, s]])}$ have no inverse? (fractions OK)`,
          answer: k, tol: 0.001,
          explain: String.raw`No inverse $\iff \det M = 0$: $(${p})(${s}) - k(${q}) = 0$, so $k = ${p * s}/${q} = ${fx(k, 4)}$. Then the columns are parallel and the plane is squashed onto a line.` };
      } },
      { id: "gauss", type: "num", gen: (r) => {
        let A, det;
        do {
          A = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => r.int(-3, 3)));
          det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        } while (det === 0);
        const x = [r.int(-4, 4), r.int(-4, 4), r.int(-4, 4)], b = L.la.matVec(A, x);
        const eq = (row, rhs) => {
          const terms = row.map((v, i) => (v ? `${v === 1 ? "" : v === -1 ? "-" : v}${["x", "y", "z"][i]}` : "")).filter(Boolean);
          return (terms.join(" + ").replace(/\+ -/g, "- ") || "0") + ` = ${rhs}`;
        };
        return { q: String.raw`Solve by elimination: $$\begin{aligned} ${eq(A[0], b[0])} \\ ${eq(A[1], b[1])} \\ ${eq(A[2], b[2])} \end{aligned}$$`,
          answer: x, labels: ["$x$", "$y$", "$z$"], tol: 1e-6,
          explain: String.raw`Eliminate $x$ from two equations, then $y$, then back-substitute. The solution is $(x, y, z) = ${vec(x)}$. Check row 1: ${A[0].map((v, i) => `(${v})(${x[i]})`).join(" + ")} $= ${b[0]}$ ✓.` };
      } },
      { id: "ata", type: "num", gen: (r) => {
        const A = Array.from({ length: 3 }, () => [r.int(-3, 3), r.int(-3, 3)]), b = [r.int(-4, 4), r.int(-4, 4), r.int(-4, 4)];
        const H = L.la.matMul(L.la.T(A), A), g = L.la.matVec(L.la.T(A), b);
        return { q: String.raw`For the least-squares problem $A\mathbf x\approx\mathbf b$ with $A = ${texM(A)}$, $\mathbf b = ${vec(b)}$, compute the entries of the normal equations $A^\top A$ and $A^\top\mathbf b$.`,
          answer: [H[0][0], H[0][1], H[1][1], g[0], g[1]], labels: ["$(A^\\top A)_{11}$", "$(A^\\top A)_{12}$", "$(A^\\top A)_{22}$", "$(A^\\top\\mathbf b)_1$", "$(A^\\top\\mathbf b)_2$"],
          explain: String.raw`$(A^\top A)_{jk}$ = column $j$ · column $k$ of $A$. Columns: $${vec(A.map((x) => x[0]))}$ and $${vec(A.map((x) => x[1]))}$. $(A^\top A)_{11} = ${H[0][0]}$, $(A^\top A)_{12} = ${H[0][1]}$, $(A^\top A)_{22} = ${H[1][1]}$. $(A^\top\mathbf b)_j$ = column $j$ · $\mathbf b$: $${g[0]}$ and $${g[1]}$.` };
      } },
      { id: "linefit", type: "num", gen: (r) => {
        const xs = r.shuffle([0, 1, 2, 3, 4, 5]).slice(0, 4).sort((a, b) => a - b), ys = xs.map(() => r.int(0, 9));
        const n = 4, sx = xs.reduce((a, b) => a + b, 0), sxx = xs.reduce((a, b) => a + b * b, 0), sy = ys.reduce((a, b) => a + b, 0), sxy = xs.reduce((a, x, i) => a + x * ys[i], 0);
        const det = sxx * n - sx * sx, m = (n * sxy - sx * sy) / det, c = (sxx * sy - sx * sxy) / det;
        return { q: String.raw`Fit $y = mx + c$ by least squares to the points ${xs.map((x, i) => `$(${x}, ${ys[i]})$`).join(", ")}. (3 decimals or fractions)`,
          answer: [m, c], labels: ["$m$", "$c$"], tol: 0.0015,
          explain: String.raw`$\sum x_i^2 = ${sxx}$, $\sum x_i = ${sx}$, $N = 4$, $\sum x_iy_i = ${sxy}$, $\sum y_i = ${sy}$. Normal equations: $${texM([[sxx, sx], [sx, n]])}${texM([["m"], ["c"]])} = ${texM([[sxy], [sy]])}$. $\det = ${sxx}\cdot4 - ${sx}^2 = ${det}$. $m = (4\cdot${sxy} - ${sx}\cdot${sy})/${det} = ${fx(m, 4)}$, $c = (${sxx}\cdot${sy} - ${sx}\cdot${sxy})/${det} = ${fx(c, 4)}$.` };
      } },
      { id: "quad", type: "num", gen: (r) => {
        const a = r.int(1, 6), b = r.int(-3, 3), c = r.int(1, 6);
        const [n1, n2v] = r.pick([[0.6, 0.8], [0.8, 0.6], [-0.6, 0.8], [0.8, -0.6], [0.28, 0.96], [-0.96, 0.28]]);
        const v = a * n1 * n1 + 2 * b * n1 * n2v + c * n2v * n2v;
        return { q: String.raw`$M = ${texM([[a, b], [b, c]])}$ and $\mathbf n = (${n1}, ${n2v})$. Compute $\mathbf n^\top M\mathbf n$. (4 decimals)`,
          answer: v, tol: 0.0002,
          explain: String.raw`$a n_1^2 + 2b n_1n_2 + c n_2^2 = ${a}(${fx(n1 * n1, 4)}) + 2(${b})(${fx(n1 * n2v, 4)}) + ${c}(${fx(n2v * n2v, 4)}) = ${fx(v, 4)}$.` };
      } },
      { id: "minq", type: "num", gen: (r) => {
        const a = r.int(0, 9), b = r.int(-4, 4), c = r.int(0, 9);
        const m = (a + c) / 2, R = Math.hypot((a - c) / 2, b);
        return { q: String.raw`For $M = ${texM([[a, b], [b, c]])}$, find the smallest and largest values of $\mathbf n^\top M\mathbf n$ over all unit vectors $\mathbf n$. (3 decimals)`,
          answer: [m - R, m + R], labels: ["min", "max"], tol: 0.0015,
          explain: String.raw`$\frac{a+c}{2} = ${fx(m)}$, $\sqrt{(\frac{a-c}{2})^2 + b^2} = \sqrt{${fx(((a - c) / 2) ** 2)} + ${b * b}} = ${fx(R, 4)}$. Min $= ${fx(m - R, 4)}$, max $= ${fx(m + R, 4)}$.` };
      } },
      { id: "pd", type: "multi",
        q: String.raw`Which of these matrices are <b>positive-definite</b> ($\mathbf x^\top M\mathbf x$ is a bowl)? (select all)`,
        choices: [String.raw`$\begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$`, String.raw`$\begin{pmatrix} 1 & 2 \\ 2 & 1\end{pmatrix}$`, String.raw`$\begin{pmatrix} 4 & -2 \\ -2 & 1\end{pmatrix}$`,
          String.raw`$\begin{pmatrix} 3 & 0 \\ 0 & 0.5\end{pmatrix}$`, String.raw`$\begin{pmatrix} -1 & 0 \\ 0 & -2\end{pmatrix}$`],
        answer: [0, 3],
        explain: String.raw`Compute $\lambda_{\min} = \frac{a+c}{2} - \sqrt{(\frac{a-c}{2})^2+b^2}$: $2 - 1 = 1 &gt; 0$ ✓; $1 - 2 = -1$ (saddle); $2.5 - 2.5 = 0$ (flat valley along $(1, 2)$, not strictly positive); $0.5 &gt; 0$ ✓; $-2$ (upside-down bowl).` },
      { id: "weakdir", type: "num", gen: (r) => {
        const a = r.int(1, 9), b = r.nz(4), c = r.int(1, 9);
        const phi = deg(Math.atan2(2 * b, a - c)), tmin = (((phi / 2 + 90) % 180) + 180) % 180;
        return { q: String.raw`For $M = ${texM([[a, b], [b, c]])}$, in which direction $\mathbf n = (\cos\theta, \sin\theta)$ is $\mathbf n^\top M\mathbf n$ smallest? Give $\theta$ in degrees in $[0, 180)$, 1 decimal.`,
          answer: tmin, tol: 0.06,
          explain: String.raw`$\varphi$ = angle of $\big(\frac{a-c}{2}, b\big) = (${fx((a - c) / 2)}, ${b})$: $\varphi = \operatorname{atan2}(${2 * b}, ${a - c}) = ${fx(phi, 2)}°$. Strongest at $\varphi/2 = ${fx(phi / 2, 2)}°$; weakest $90°$ further: $${fx(phi / 2 + 90, 2)}°$${phi / 2 + 90 >= 180 ? String.raw`, i.e. $${fx(tmin, 2)}°$ (a direction and its opposite are the same line)` : ""}.` };
      } },
      { id: "null2", type: "num", gen: (r) => {
        let r1, r2, x;
        do {
          r1 = [r.int(-3, 3), r.int(-3, 3), r.int(-3, 3)]; r2 = [r.int(-3, 3), r.int(-3, 3), r.int(-3, 3)];
          x = L.la.cross(r1, r2);
        } while (x[2] === 0);
        const nx = L.la.norm(x), s = x[2] > 0 ? 1 : -1, u = x.map((v) => (s * v) / nx);
        return { q: String.raw`$A = ${texM([r1, r2])}$. Find the unit vector $\mathbf x$ with $A\mathbf x = \mathbf 0$ and $x_3 &gt; 0$. (3 decimals)`,
          answer: u, labels: ["$x_1$", "$x_2$", "$x_3$"], tol: 0.0015,
          explain: String.raw`$\mathbf x$ must be perpendicular to both rows, so it is along $\mathbf r_1\times\mathbf r_2 = ${vec(x)}$, length $\sqrt{${L.la.dot(x, x)}} = ${fx(nx, 4)}$. ${s < 0 ? "Flip the sign so that $x_3 &gt; 0$. " : ""}$\mathbf x = (${u.map((v) => fx(v, 4)).join(", ")})$.` };
      } },
      { id: "nullwhy", type: "mc",
        q: String.raw`When solving $A\mathbf x \approx \mathbf 0$, why do we add the constraint $|\mathbf x| = 1$?`,
        choices: ["Otherwise $\\mathbf x = \\mathbf 0$ is a perfect but useless answer, and any multiple of a solution is also a solution",
          "Because the normal equations $A^\\top A\\mathbf x = A^\\top\\mathbf b$ need a unit vector",
          "To make $A^\\top A$ invertible",
          "Because unit vectors are faster to compute with"],
        answer: 0,
        explain: "$|A\\mathbf x|$ can always be made smaller by shrinking $\\mathbf x$, down to $\\mathbf 0$. Fixing the length leaves only the direction free, which is what we want." },
      { id: "svd", type: "mc",
        q: String.raw`You have a 20×9 matrix $A$ and want the unit $\mathbf x$ minimising $|A\mathbf x|$. What do you compute?`,
        choices: ["The direction with the smallest $\\lambda$ of $A^\\top A$ (equivalently, the last column of $V$ in the SVD $A = U\\Sigma V^\\top$)",
          "The direction with the largest $\\lambda$ of $A^\\top A$",
          "$\\mathbf x = (A^\\top A)^{-1}A^\\top\\mathbf 0$",
          "The column of $A$ with the smallest length"],
        answer: 0,
        explain: "$|A\\mathbf x|^2 = \\mathbf x^\\top A^\\top A\\mathbf x$, a quadratic form, is smallest along the weakest direction. The normal equations with right side $\\mathbf 0$ just give $\\mathbf x = \\mathbf 0$." },
    ]);
  },
});
