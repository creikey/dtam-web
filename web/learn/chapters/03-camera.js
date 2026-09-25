// Chapter 3: the pinhole camera — projection, K, back-projection, FOV, inverse depth, pyramid intrinsics.
DTAM.chapter({
  id: "camera",
  order: 3,
  title: "The pinhole camera",
  subtitle: "From a 3D point to a pixel and back again",
  minutes: 40,
  render(root, L) {
    const { la } = L;

    // ------------------------------------------------------------ small 3D viewer (orbit by dragging)
    // World axes as in the camera: x right, y DOWN, z forward. The viewer looks
    // from a point on a sphere around `target`; drag to change yaw / pitch.
    function orbit(c, o = {}) {
      const v = { yaw: o.yaw ?? -0.75, pitch: o.pitch ?? 0.38, dist: o.dist ?? 8, target: o.target ?? [0, 0, 2.5], zoom: o.zoom ?? 1 };
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
      v.seg = (ctx, a, b, col, w = 1.5, dash) => {
        const A = v.P(a), B = v.P(b);
        if (A && B) L.draw.line(ctx, A[0], A[1], B[0], B[1], col, w, dash);
      };
      v.dot = (ctx, p, r, col, stroke) => {
        const A = v.P(p);
        if (A) L.draw.dot(ctx, A[0], A[1], r, col, stroke);
      };
      v.label = (ctx, p, s, col, dx = 6, dy = -6) => {
        const A = v.P(p);
        if (A) L.draw.text(ctx, s, A[0] + dx, A[1] + dy, col, { size: 12 });
      };
      let last = null;
      c.el.addEventListener("pointerdown", (e) => {
        last = { x: e.clientX, y: e.clientY };
        c.el.setPointerCapture(e.pointerId);
        e.preventDefault();
      });
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
    // Camera frustum at the origin (identity pose) with image plane at depth D.
    function drawFrustum(ctx, v, K, D, col) {
      const cs = [[-0.5, -0.5], [K.w - 0.5, -0.5], [K.w - 0.5, K.h - 0.5], [-0.5, K.h - 0.5]]
        .map(([u, vv]) => [((u - K.cx) / K.f) * D, ((vv - K.cy) / K.f) * D, D]);
      for (let i = 0; i < 4; i++) {
        v.seg(ctx, [0, 0, 0], cs[i], col, 1.2);
        v.seg(ctx, cs[i], cs[(i + 1) % 4], col, i === 0 ? 3 : 1.5);
      }
      v.dot(ctx, [0, 0, 0], 4, col);
    }
    function axes3(ctx, v, t) {
      v.seg(ctx, [0, 0, 0], [0.6, 0, 0], t.bad, 2); v.label(ctx, [0.6, 0, 0], "x", t.bad);
      v.seg(ctx, [0, 0, 0], [0, 0.6, 0], t.good, 2); v.label(ctx, [0, 0.6, 0], "y", t.good);
      v.seg(ctx, [0, 0, 0], [0, 0, 0.6], t.accent, 2); v.label(ctx, [0, 0, 0.6], "z", t.accent);
    }
    // Maps an image of size W x H (pixel centres at integers) into a canvas rect.
    function imgMap(c, W, H, pad = 10) {
      const s = Math.min((c.w - 2 * pad) / W, (c.h - 2 * pad) / H);
      const ox = (c.w - s * W) / 2, oy = (c.h - s * H) / 2;
      return {
        s, ox, oy,
        X: (u) => ox + (u + 0.5) * s,
        Y: (vv) => oy + (vv + 0.5) * s,
        iu: (px) => (px - ox) / s - 0.5,
        iv: (py) => (py - oy) / s - 0.5,
        frame(ctx, t) {
          ctx.save();
          ctx.fillStyle = t.panel2; ctx.fillRect(ox, oy, s * W, s * H);
          ctx.strokeStyle = t.line; ctx.lineWidth = 1;
          for (let g = 1; g < 8; g++) {
            L.draw.line(ctx, ox + (g * s * W) / 8, oy, ox + (g * s * W) / 8, oy + s * H, t.line, 1);
            L.draw.line(ctx, ox, oy + (g * s * H) / 8, ox + s * W, oy + (g * s * H) / 8, t.line, 1);
          }
          ctx.strokeStyle = t.muted; ctx.lineWidth = 1.5; ctx.strokeRect(ox, oy, s * W, s * H);
          ctx.restore();
        },
      };
    }
    const f1 = (x) => (+x).toFixed(1), f2 = (x) => (+x).toFixed(2), f3 = (x) => (+x).toFixed(3);

    // ------------------------------------------------------------ intro
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>DTAM constantly asks two questions: <i>"where in the image does this 3D point appear?"</i> and <i>"which 3D points could this pixel be looking at?"</i> This chapter answers both with one small matrix, $K$.</p>
      <div class="note">Camera coordinates (paper and this code): origin at the camera centre, $x$ to the right, $y$ <b>down</b>, $z$ forward (the viewing direction). $z$ is the <b>depth</b> of a point.</div>
      <h3>Similar triangles</h3>
      <p>A pinhole camera is a dark box with a tiny hole. Light from a point travels in a straight line through the hole and hits the back wall. The picture is upside down; mathematically it is simpler to use a <i>virtual</i> image plane the same distance <b>in front</b> of the hole, where the picture is upright.</p>
    `);

    // ------------------------------------------------------------ W1: side view similar triangles
    {
      const fig = L.figure(root, "<b>Side view (y–z plane).</b> Drag the tip of the arrow. Change the focal length. Turn on “same ray” to see many points land on one pixel.");
      const c = L.canvas(fig.el, { aspect: 0.58 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const P = { y: -0.8, z: 3 };
      let showRay = false;
      const sf = L.slider(ctl, { label: "focal length $f$ (px)", min: 200, max: 800, step: 10, value: 500, oninput: () => c.redraw() });
      L.toggle(ctl, "same ray", false, (v) => { showRay = v; c.redraw(); });
      const geo = () => {
        const sc = c.w / 7, X0 = c.w * 0.3, Y0 = c.h / 2;
        return { sc, X0, Y0, X: (z) => X0 + z * sc, Y: (y) => Y0 + y * sc, iz: (px) => (px - X0) / sc, iy: (py) => (py - Y0) / sc };
      };
      L.drag(c, () => { const g = geo(); return [{ x: g.X(P.z), y: g.Y(P.y) }]; }, (_, p) => {
        const g = geo();
        P.z = Math.max(0.4, Math.min(4.7, g.iz(p.x)));
        P.y = Math.max(-1.7, Math.min(1.7, g.iy(p.y)));
      });
      c.draw = (ctx) => {
        const t = L.theme(), g = geo(), f = sf.value;
        const zp = f / 500; // drawn plane distance (m) — 500 px per metre of plane distance
        const half = 240 / 500; // sensor half-height (240 px) in drawn metres
        // optical axis
        L.draw.line(ctx, 0, g.Y0, c.w, g.Y0, t.line, 1, [4, 4]);
        L.draw.text(ctx, "z (optical axis)", c.w - 6, g.Y0 + 16, t.faint, { size: 11, align: "right" });
        // planes
        L.draw.line(ctx, g.X(zp), g.Y(-half), g.X(zp), g.Y(half), t.accent, 3);
        L.draw.text(ctx, "virtual image", g.X(zp) + 4, g.Y(half) + 14, t.accent, { size: 11 });
        L.draw.line(ctx, g.X(-zp), g.Y(-half), g.X(-zp), g.Y(half), t.muted, 3);
        L.draw.text(ctx, "real (flipped)", g.X(-zp) + 2, g.Y(half) + 14, t.muted, { size: 11, align: "center" });
        // f bracket
        L.draw.line(ctx, g.X(0), g.Y(half) + 24, g.X(zp), g.Y(half) + 24, t.faint, 1);
        L.draw.text(ctx, "f", g.X(zp / 2), g.Y(half) + 38, t.muted, { size: 12, align: "center" });
        // object arrow
        L.draw.arrow(ctx, g.X(P.z), g.Y0, g.X(P.z), g.Y(P.y), t.accent3, 3);
        // rays
        const yv = (P.y * zp) / P.z, yr = -yv;
        L.draw.line(ctx, g.X(P.z), g.Y(P.y), g.X(-zp), g.Y(yr), t.accent2, 1.5);
        L.draw.arrow(ctx, g.X(zp), g.Y0, g.X(zp), g.Y(yv), t.accent2, 2.5, 7);
        L.draw.arrow(ctx, g.X(-zp), g.Y0, g.X(-zp), g.Y(yr), t.muted, 2.5, 7);
        if (showRay) {
          for (const k of [0.4, 0.7, 1.3, 1.55]) {
            const z = P.z * k, y = P.y * k;
            if (g.X(z) < c.w - 4 && Math.abs(g.Y(y) - g.Y0) < c.h / 2 - 4) L.draw.dot(ctx, g.X(z), g.Y(y), 5, t.accent4);
          }
          L.draw.line(ctx, g.X(0), g.Y0, g.X(6), g.Y((P.y / P.z) * 6), t.accent4, 1, [3, 4]);
        }
        L.draw.dot(ctx, g.X(0), g.Y0, 4, t.fg);
        L.draw.text(ctx, "pinhole", g.X(0) - 4, g.Y0 - 8, t.fg, { size: 11, align: "right" });
        L.draw.handle(ctx, g.X(P.z), g.Y(P.y), t.accent3);
        out.html = `y = <b>${f2(P.y)}</b> m, z = <b>${f2(P.z)}</b> m → v − c<sub>y</sub> = f·y/z = ${f}·${f2(P.y)}/${f2(P.z)} = <b>${f1((f * P.y) / P.z)} px</b>`;
      };
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <p>The triangle (pinhole, arrow base, arrow tip) and the small triangle (pinhole, plane centre, image tip) have the same shape, so $\frac{\text{image height}}{f} = \frac{y}{z}$. Measuring $f$ <b>in pixels</b> gives the answer directly in pixels.</p>
      <div class="eq-card"><div class="eq-label">Pinhole projection</div>
      $$u = f_x\,\frac{x}{z} + c_x,\qquad v = f_y\,\frac{y}{z} + c_y$$
      <div class="parts">
        <span>$(x, y, z)$</span><span>3D point in camera coordinates (metres)</span>
        <span>$x/z,\ y/z$</span><span>the point's direction: "how far sideways per metre forward"</span>
        <span>$f_x, f_y$</span><span>focal length in pixels (distance from pinhole to image plane, measured in pixel widths). Square pixels: $f_x = f_y = f$</span>
        <span>$c_x, c_y$</span><span>principal point: the pixel the optical axis hits (≈ the image centre)</span>
      </div></div>
      <p><b>Worked example.</b> $f = 500$, $(c_x, c_y) = (319.5, 239.5)$, point $\mathbf x = (0.2, -0.1, 2)$:
      $u = 500\cdot 0.1 + 319.5 = 369.5$, $v = 500\cdot(-0.05) + 239.5 = 214.5$.</p>
      <div class="note"><b>Why 319.5?</b> This implementation puts pixel <i>centres</i> at integer coordinates (as in chapter 1), so a 640-pixel-wide image spans $u\in[-0.5, 639.5]$ and its exact centre is $(640-1)/2 = 319.5$. <code>Intrinsics::centered</code> uses $c_x = (w-1)/2$, $c_y = (h-1)/2$.</div>
      <p>The division by $z$ is what makes far things small, and it throws information away: every point on a ray through the pinhole lands on the <b>same pixel</b>. One image alone cannot tell depth. DTAM's whole job is recovering that lost $z$.</p>

      <h3>Homogeneous coordinates and $K$</h3>
      <p>Division is not a matrix operation, so we split projection into a matrix step and a divide step. A 3-vector $(a, b, w)$ with $w\ne 0$ is used to <i>represent</i> the 2D point $(a/w,\ b/w)$. Scaling it by any non-zero number represents the same point: $(6, 4, 2)$, $(3, 2, 1)$ and $(-3,-2,-1)$ all mean $(3, 2)$. A pixel written as $\dot{\mathbf u} = (u, v, 1)^\top$ is its <b>homogeneous</b> form.</p>
      <div class="eq-card"><div class="eq-label">Intrinsic matrix and projection (paper §2.1)</div>
      $$K=\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix},\qquad K\mathbf x=\begin{pmatrix}f_x x + c_x z\\ f_y y + c_y z\\ z\end{pmatrix},\qquad \mathbf u = \pi(K\mathbf x)$$
      <div class="parts">
        <span>$K$</span><span>"intrinsics": everything about the camera's inside (lens + sensor), nothing about where it is</span>
        <span>$K\mathbf x$</span><span>a homogeneous pixel; its last entry is the depth $z$</span>
        <span>$\pi(a,b,w) = (a/w,\ b/w)$</span><span>dehomogenisation: divide by the last entry. The paper writes $\pi(\mathbf x)=(x/z,\,y/z)^\top$</span>
      </div></div>
      <p>Check: $\pi(K\mathbf x) = \left(\frac{f_x x + c_x z}{z}, \frac{f_y y + c_y z}{z}\right) = \left(f_x\frac xz + c_x,\ f_y\frac yz + c_y\right)$, exactly the pinhole formula.</p>
    `);

    // ------------------------------------------------------------ W2: project a cube (3D + image)
    {
      const K = { f: 500, cx: 319.5, cy: 239.5, w: 640, h: 480 };
      const fig = L.figure(root, "<b>Projection, live.</b> Drag the 3D view to orbit. Change $f$, the cube's depth and $c_x$; watch the image (below) and the frustum. The frustum's image plane is drawn at depth 1.");
      const grid = L.el("div", { class: "grid2" });
      fig.add(grid);
      const d1 = L.el("div"), d2 = L.el("div");
      grid.append(d1, d2);
      const c3 = L.canvas(d1, { aspect: 0.8 });
      const ci = L.canvas(d2, { aspect: 0.8, scroll: true });
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const redraw = () => { c3.redraw(); ci.redraw(); };
      const sf = L.slider(ctl, { label: "$f$ (px)", min: 200, max: 1200, step: 10, value: 500, oninput: redraw });
      const sz = L.slider(ctl, { label: "cube depth $z$ (m)", min: 1.5, max: 8, step: 0.1, value: 3, oninput: redraw });
      const sc = L.slider(ctl, { label: "$c_x$ (px)", min: 160, max: 480, step: 0.5, value: 319.5, oninput: redraw });
      const view = orbit(c3, { target: [0, 0, 2.2], dist: 9, zoom: 1.15 });
      const cube = () => {
        const z0 = sz.value, a = 0.6, h = 0.45;
        const pts = [];
        for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const s3 of [-1, 1]) {
          const x = sx * h, zz = s3 * h;
          pts.push([0.35 + Math.cos(a) * x + Math.sin(a) * zz, -0.15 + sy * h, z0 - Math.sin(a) * x + Math.cos(a) * zz]);
        }
        return pts;
      };
      const edges = [];
      for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
        const d = (i ^ j);
        if (d === 1 || d === 2 || d === 4) edges.push([i, j]);
      }
      const Kc = () => ({ ...K, f: sf.value, cx: sc.value });
      const proj = (k, p) => [k.f * p[0] / p[2] + k.cx, k.f * p[1] / p[2] + k.cy];
      c3.draw = (ctx) => {
        const t = L.theme(), k = Kc(), pts = cube();
        axes3(ctx, view, t);
        drawFrustum(ctx, view, k, 1, t.muted);
        for (const p of pts) view.seg(ctx, [0, 0, 0], p, t.faint, 1, [2, 4]);
        for (const [i, j] of edges) view.seg(ctx, pts[i], pts[j], t.accent3, 2);
        // cube drawn on the image plane at depth 1: (x/z, y/z, 1)
        const onPlane = pts.map((p) => [p[0] / p[2], p[1] / p[2], 1]);
        for (const [i, j] of edges) view.seg(ctx, onPlane[i], onPlane[j], t.accent2, 1.5);
        view.dot(ctx, pts[0], 4, t.accent4);
      };
      ci.draw = (ctx) => {
        const t = L.theme(), k = Kc(), pts = cube(), M = imgMap(ci, k.w, k.h);
        M.frame(ctx, t);
        ctx.save();
        ctx.beginPath(); ctx.rect(M.ox, M.oy, M.s * k.w, M.s * k.h); ctx.clip();
        const uv = pts.map((p) => proj(k, p));
        for (const [i, j] of edges) L.draw.line(ctx, M.X(uv[i][0]), M.Y(uv[i][1]), M.X(uv[j][0]), M.Y(uv[j][1]), t.accent2, 2);
        L.draw.dot(ctx, M.X(uv[0][0]), M.Y(uv[0][1]), 4, t.accent4);
        ctx.restore();
        const px = M.X(k.cx), py = M.Y(k.cy);
        L.draw.line(ctx, px - 7, py, px + 7, py, t.fg, 1.5);
        L.draw.line(ctx, px, py - 7, px, py + 7, t.fg, 1.5);
        L.draw.text(ctx, "(cx, cy)", px + 6, py - 6, t.muted, { size: 11 });
        L.draw.text(ctx, "image 640×480", M.ox + 4, M.oy + 14, t.faint, { size: 11 });
        const p = pts[0], Kx = [k.f * p[0] + k.cx * p[2], k.f * p[1] + k.cy * p[2], p[2]];
        out.html = `orange corner x = (${f2(p[0])}, ${f2(p[1])}, ${f2(p[2])})<br>Kx = (${f1(Kx[0])}, ${f1(Kx[1])}, ${f2(Kx[2])})<br>π(Kx) = (<b>${f1(Kx[0] / Kx[2])}</b>, <b>${f1(Kx[1] / Kx[2])}</b>) px`;
      };
    }

    // ------------------------------------------------------------ back-projection
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Back-projection: from a pixel to a ray</h3>
      <p>Going backwards, a pixel only fixes a <b>direction</b>. Undo $K$ to get the point on that ray at depth $z = 1$, then scale it out to the depth you want. DTAM stores <b>inverse depth</b> $d = \xi = 1/z$, so "scale to depth $z$" becomes "divide by $d$".</p>
      <div class="eq-card"><div class="eq-label">Back-projection (paper §2.1)</div>
      $$\pi^{-1}(\mathbf u, d) = \frac1d\,K^{-1}\dot{\mathbf u},\qquad K^{-1}\dot{\mathbf u} = \begin{pmatrix}(u-c_x)/f_x\\(v-c_y)/f_y\\1\end{pmatrix}$$
      <div class="parts">
        <span>$\dot{\mathbf u} = (u,v,1)^\top$</span><span>the pixel, homogeneous</span>
        <span>$K^{-1}\dot{\mathbf u}$</span><span>the point on pixel $\mathbf u$'s ray at depth exactly 1 ("normalised coordinates")</span>
        <span>$d$</span><span>inverse depth, $d = 1/z$; the paper uses $d$ for a candidate value and $\xi(\mathbf u)$ for the map</span>
        <span>$\frac1d$</span><span>scales the depth-1 point out to depth $z = 1/d$ along the same ray</span>
      </div></div>
      <p><b>Worked example.</b> $f=500$, $(c_x,c_y)=(319.5,239.5)$, pixel $(419.5, 139.5)$, $d = 0.5$: $K^{-1}\dot{\mathbf u} = (100/500,\,-100/500,\,1) = (0.2,-0.2,1)$, so $\mathbf x = (0.2,-0.2,1)/0.5 = (0.4,-0.4,2)$. Projecting it again: $u = 500\cdot 0.4/2 + 319.5 = 419.5$ ✓.</p>
    `);

    {
      const K = { f: 500, cx: 319.5, cy: 239.5, w: 640, h: 480 };
      const fig = L.figure(root, "<b>Back-projection.</b> Drag the pixel in the image (right/below). Slide the inverse depth $d$: the 3D point slides along the pixel's ray. Orbit the 3D view by dragging.");
      const grid = L.el("div", { class: "grid2" });
      fig.add(grid);
      const d1 = L.el("div"), d2 = L.el("div");
      grid.append(d1, d2);
      const c3 = L.canvas(d1, { aspect: 0.8 });
      const ci = L.canvas(d2, { aspect: 0.8 });
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const px = { u: 470, v: 160 };
      const redraw = () => { c3.redraw(); ci.redraw(); };
      const sd = L.slider(ctl, { label: "inverse depth $d$ (1/m)", min: 0.12, max: 2, step: 0.01, value: 0.4, oninput: redraw });
      const view = orbit(c3, { target: [0, 0, 3], dist: 11, zoom: 1.15, yaw: -1.0, pitch: 0.45 });
      L.drag(ci, () => { const M = imgMap(ci, K.w, K.h); return [{ x: M.X(px.u), y: M.Y(px.v) }]; }, (_, p) => {
        const M = imgMap(ci, K.w, K.h);
        px.u = Math.round(Math.max(0, Math.min(K.w - 1, M.iu(p.x))));
        px.v = Math.round(Math.max(0, Math.min(K.h - 1, M.iv(p.y))));
        c3.redraw();
      });
      c3.draw = (ctx) => {
        const t = L.theme(), d = sd.value;
        const n = [(px.u - K.cx) / K.f, (px.v - K.cy) / K.f, 1];
        axes3(ctx, view, t);
        drawFrustum(ctx, view, K, 1, t.muted);
        view.seg(ctx, [0, 0, 0], la.scale(n, 9), t.accent, 1.5, [5, 4]);
        view.dot(ctx, n, 4, t.accent2);
        view.label(ctx, n, "z = 1", t.accent2);
        const x = la.scale(n, 1 / d);
        view.seg(ctx, [x[0], 0, x[2]], x, t.faint, 1, [2, 3]);
        view.dot(ctx, x, 7, t.accent3, "white");
        view.label(ctx, x, `z = ${f2(1 / d)}`, t.accent3, 9, -8);
      };
      ci.draw = (ctx) => {
        const t = L.theme(), M = imgMap(ci, K.w, K.h), d = sd.value;
        M.frame(ctx, t);
        const cx = M.X(K.cx), cy = M.Y(K.cy);
        L.draw.line(ctx, cx - 7, cy, cx + 7, cy, t.fg, 1.5);
        L.draw.line(ctx, cx, cy - 7, cx, cy + 7, t.fg, 1.5);
        L.draw.handle(ctx, M.X(px.u), M.Y(px.v), t.accent);
        L.draw.text(ctx, "u", M.X(px.u) + 11, M.Y(px.v) - 9, t.accent, { size: 12, bold: true });
        const n = [(px.u - K.cx) / K.f, (px.v - K.cy) / K.f, 1];
        out.html = `u = (${px.u}, ${px.v})<br>K⁻¹u̇ = (${f3(n[0])}, ${f3(n[1])}, 1)<br>x = K⁻¹u̇ / d = (<b>${f3(n[0] / d)}</b>, <b>${f3(n[1] / d)}</b>, <b>${f3(1 / d)}</b>)`;
      };
    }

    // ------------------------------------------------------------ FOV
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Field of view ↔ focal length</h3>
      <p>The image edge is $w/2$ pixels from the centre. The ray to that edge makes angle $\text{hfov}/2$ with the optical axis, and the right triangle (pinhole, centre, edge) gives:</p>
      <div class="eq-card"><div class="eq-label">Horizontal field of view</div>
      $$\tan\frac{\text{hfov}}{2} = \frac{w/2}{f_x}\qquad\Longleftrightarrow\qquad f_x = \frac{w/2}{\tan(\text{hfov}/2)}$$
      <div class="parts">
        <span>$w$</span><span>image width in pixels</span>
        <span>$\text{hfov}$</span><span>horizontal angle the camera sees</span>
        <span>$f_x$</span><span>focal length in pixels. Larger $f$ = narrower view = "zoomed in"</span>
      </div></div>
      <p><b>Worked example.</b> $w=640$, $f=320$: $\tan(\text{hfov}/2) = 320/320 = 1$, so $\text{hfov}/2 = 45°$ and $\text{hfov} = 90°$. A phone camera at 640 px wide with hfov ≈ 65° has $f = 320/\tan 32.5° ≈ 502$ px. (This implementation estimates $f$ itself during bootstrap, chapter 7.)</p>
    `);
    {
      const fig = L.figure(root, "<b>Top-down view.</b> Change $f$ and the image width $w$. Posts inside the wedge appear in the image strip at the top, at $u = f\\,x/z + c_x$.");
      const c = L.canvas(fig.el, { aspect: 0.7, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const sf = L.slider(ctl, { label: "$f$ (px)", min: 100, max: 1400, step: 10, value: 500, oninput: () => c.redraw() });
      const sw = L.slider(ctl, { label: "width $w$ (px)", min: 320, max: 1280, step: 32, value: 640, oninput: () => c.redraw() });
      const posts = [[-3, 4], [-1.2, 2.2], [0.3, 6], [1.1, 3.4], [2.6, 5.2], [-0.6, 7.8], [3.9, 7], [-4.5, 8.5], [0.9, 1.3]];
      c.draw = (ctx) => {
        const t = L.theme(), f = sf.value, w = sw.value, cx = (w - 1) / 2;
        const stripY = 14, stripH = 24, top = stripY + stripH + 22;
        const sc = (c.h - top - 18) / 9, X0 = c.w / 2, Y0 = c.h - 14;
        const X = (x) => X0 + x * sc, Y = (z) => Y0 - z * sc;
        const half = Math.atan(w / 2 / f);
        // wedge
        const R = 20;
        ctx.save();
        ctx.fillStyle = t.accent; ctx.globalAlpha = 0.12;
        ctx.beginPath(); ctx.moveTo(X(0), Y(0));
        ctx.lineTo(X(Math.tan(half) * R), Y(R)); ctx.lineTo(X(-Math.tan(half) * R), Y(R)); ctx.closePath(); ctx.fill();
        ctx.restore();
        ctx.save(); ctx.beginPath(); ctx.rect(0, top - 4, c.w, c.h); ctx.clip();
        L.draw.line(ctx, X(0), Y(0), X(Math.tan(half) * R), Y(R), t.accent, 1.5);
        L.draw.line(ctx, X(0), Y(0), X(-Math.tan(half) * R), Y(R), t.accent, 1.5);
        ctx.restore();
        L.draw.dot(ctx, X(0), Y(0), 5, t.fg);
        L.draw.text(ctx, "camera (z up the page)", X(0) + 8, Y(0) - 2, t.muted, { size: 11 });
        // strip
        const sx0 = 12, sx1 = c.w - 12;
        const U = (u) => sx0 + ((u + 0.5) / w) * (sx1 - sx0);
        ctx.save(); ctx.fillStyle = t.panel2; ctx.fillRect(sx0, stripY, sx1 - sx0, stripH); ctx.strokeStyle = t.muted; ctx.strokeRect(sx0, stripY, sx1 - sx0, stripH); ctx.restore();
        L.draw.text(ctx, `image row, ${w} px`, sx0, stripY + stripH + 14, t.faint, { size: 11 });
        let inView = 0;
        posts.forEach(([x, z], i) => {
          const u = (f * x) / z + cx;
          const vis = u >= -0.5 && u <= w - 0.5;
          if (Y(z) > top) L.draw.dot(ctx, X(x), Y(z), 6, vis ? t.accent3 : t.faint);
          if (vis) {
            inView++;
            ctx.save(); ctx.fillStyle = t.accent3; ctx.fillRect(U(u) - 2.5, stripY + 3, 5, stripH - 6); ctx.restore();
            if (Y(z) > top) L.draw.line(ctx, X(x), Y(z), X(0), Y(0), t.accent3, 0.8, [2, 4]);
          }
        });
        const hf = (2 * half * 180) / Math.PI;
        out.html = `hfov = 2·atan(w/(2f)) = 2·atan(${w}/${2 * f}) = <b>${f1(hf)}°</b> · posts in view: ${inView}/${posts.length}`;
      };
    }

    // ------------------------------------------------------------ inverse depth
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Why DTAM uses inverse depth</h3>
      <p>Put a second camera $m$ a distance $b$ to the right of the reference camera $r$ (same orientation). A point at depth $z$ on the optical axis of $r$ has $x = 0$ in $r$ and $x = -b$ in $m$, so in $m$ it appears at $u_m = f\cdot(-b)/z + c_x$:</p>
      <div class="eq-card"><div class="eq-label">Disparity (sideways camera shift)</div>
      $$u_r - u_m = f\,b\,\frac1z = f\,b\,\xi$$
      <div class="parts">
        <span>$u_r - u_m$</span><span>how far the point's image shifts between the two cameras (disparity, pixels)</span>
        <span>$b$</span><span>baseline: distance between camera centres (m)</span>
        <span>$\xi = 1/z$</span><span>inverse depth. The shift is <b>proportional to $\xi$</b>, not to $z$</span>
      </div></div>
      <p>The image of a pixel's ray in another camera is a line (the <b>epipolar line</b>, chapter 4). To test depths for one pixel, DTAM walks along that line. Sampling equally in $\xi$ means equal pixel steps along the line; sampling equally in $z$ wastes most samples on far depths where the image hardly moves, and skips pixels near the camera.</p>
      <div class="eq-card"><div class="eq-label">Cost volume layers (paper §2.2.2)</div>
      $$\xi_k = \xi_{min} + k\,\Delta\xi,\qquad \Delta\xi=\frac{\xi_{max}-\xi_{min}}{S-1},\qquad k = 0,\dots,S-1$$
      <div class="parts">
        <span>$S$</span><span>number of layers (this implementation: 64 on desktop, 32 on mobile)</span>
        <span>$\xi_{min}$</span><span>farthest depth tested ($\xi_{min}=0$ would be infinitely far — no problem in inverse depth)</span>
        <span>$\xi_{max}$</span><span>nearest depth tested</span>
      </div></div>
      <p><b>Worked example.</b> $\xi_{min}=0.2$, $\xi_{max}=2$, $S=10$: $\Delta\xi = 1.8/9 = 0.2$, so layer $k=4$ has $\xi_4 = 1.0$ ($z = 1$ m). With $f=500$, $b=0.1$ each layer step moves $500\cdot 0.1\cdot 0.2 = 10$ px along the epipolar line.</p>
    `);
    {
      const fig = L.figure(root, "<b>Sampling a ray.</b> Top: camera $r$ (left, looking right) and camera $m$ shifted sideways by $b$. Dots are the depth samples on one ray of $r$. Bottom: where each sample appears in camera $m$'s image row. Switch between uniform-in-$z$ and uniform-in-$\\xi$.");
      const c = L.canvas(fig.el, { aspect: 0.62, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      let uniXi = true;
      const sS = L.slider(ctl, { label: "layers $S$", min: 4, max: 32, step: 1, value: 12, oninput: () => c.redraw() });
      const sb = L.slider(ctl, { label: "baseline $b$ (m)", min: 0.05, max: 0.5, step: 0.01, value: 0.2, oninput: () => c.redraw() });
      L.toggle(ctl, "uniform in $\\xi$ (off: uniform in $z$)", true, (v) => { uniXi = v; c.redraw(); });
      const f = 500, zn = 0.5, zf = 8;
      c.draw = (ctx) => {
        const t = L.theme(), S = sS.value, b = sb.value;
        const sceneH = c.h * 0.6, zs = (c.w - 40) / 8.6, xs = zs * 2.2;
        const Z = (z) => 28 + z * zs, Xr = (x) => sceneH * 0.3 + x * xs;
        const samples = [];
        for (let k = 0; k < S; k++) {
          const z = uniXi ? 1 / (1 / zf + (k * (1 / zn - 1 / zf)) / (S - 1)) : zf + (k * (zn - zf)) / (S - 1);
          samples.push(z);
        }
        // cameras
        L.draw.line(ctx, Z(0), Xr(0), Z(8.6), Xr(0), t.accent, 1.2, [5, 4]);
        L.draw.dot(ctx, Z(0), Xr(0), 6, t.accent);
        L.draw.text(ctx, "r", Z(0) - 14, Xr(0) + 4, t.accent, { size: 13, bold: true });
        L.draw.dot(ctx, Z(0), Xr(b), 6, t.accent2);
        L.draw.text(ctx, "m", Z(0) - 16, Xr(b) + 4, t.accent2, { size: 13, bold: true });
        for (const z of samples) {
          L.draw.line(ctx, Z(z), Xr(0), Z(0), Xr(b), t.faint, 0.7);
          L.draw.dot(ctx, Z(z), Xr(0), 3.5, t.accent3);
        }
        for (let z = 0; z <= 8; z += 2) L.draw.text(ctx, `${z} m`, Z(z), sceneH - 2, t.faint, { size: 11, align: "center" });
        L.draw.text(ctx, "sideways ×2.2", c.w - 6, 14, t.faint, { size: 11, align: "right" });
        // image row of m: offset u_m - c_x = -f b xi
        const maxShift = f * 0.5 * (1 / zn) * 1.02;
        const y0 = c.h * 0.74, hh = c.h * 0.14, x0 = 12, x1 = c.w - 12;
        const U = (s) => x1 - (s / maxShift) * (x1 - x0);
        ctx.save(); ctx.fillStyle = t.panel2; ctx.fillRect(x0, y0, x1 - x0, hh); ctx.strokeStyle = t.muted; ctx.strokeRect(x0, y0, x1 - x0, hh); ctx.restore();
        const us = samples.map((z) => (f * b) / z);
        for (const s of us) L.draw.line(ctx, U(s), y0 + 3, U(s), y0 + hh - 3, t.accent3, 2);
        L.draw.text(ctx, "m's image row: shift = f·b·ξ (∞ at right)", x0, y0 + hh + 15, t.faint, { size: 11 });
        const gaps = us.slice(1).map((s, i) => Math.abs(s - us[i]));
        out.html = `pixel gaps between neighbouring samples: smallest <b>${f1(Math.min(...gaps))}</b> px, largest <b>${f1(Math.max(...gaps))}</b> px` +
          ` · z from ${zn} to ${zf} m, f = ${f}`;
      };
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <div class="key">DTAM tests $S$ inverse depths per pixel, spaced evenly in $\xi$. Even $\xi$ steps are even pixel steps along the epipolar line (exactly for sideways motion, roughly in general), and $\xi = 0$ (infinity) is an ordinary number.</div>

      <h3>Intrinsics at pyramid levels</h3>
      <p>Recall from chapter 1: pyramid level $l$ halves the resolution $l$ times, and with pixel centres at integers a level-0 coordinate $u$ becomes $(u+0.5)/2^l - 0.5$. Apply that to the pinhole formula and you get the intrinsics of the smaller image:</p>
      <div class="eq-card"><div class="eq-label">Intrinsics at level l (<code>Intrinsics::level</code>)</div>
      $$f^{(l)} = \frac{f}{2^l},\qquad c^{(l)} = \frac{c + 0.5}{2^l} - 0.5$$
      <div class="parts">
        <span>$f^{(l)}$</span><span>focal length shrinks with the pixel count (same angle, fewer pixels)</span>
        <span>$c^{(l)}$</span><span>principal point, for $c_x$ and $c_y$ alike. Just halving $c$ would be off by a quarter pixel at level 1</span>
      </div></div>
      <p><b>Worked example.</b> $f=500$, $c_x=319.5$ (640 wide). Level 1: $f=250$, $c_x = 320/2 - 0.5 = 159.5$ (the centre of a 320-wide image ✓). Level 2: $f=125$, $c_x = 320/4-0.5 = 79.5$.</p>
      <h3>In code</h3>
<pre><code>project(K, x):               // 3D point (camera frame) -> pixel
    return (K.fx * x.x / x.z + K.cx,  K.fy * x.y / x.z + K.cy)

unproject(K, u, d):          // pixel + inverse depth -> 3D point
    n = ((u.u - K.cx) / K.fx, (u.v - K.cy) / K.fy, 1)   // K^-1 u̇ : depth 1
    return n / d

level(K, l):                 // intrinsics of pyramid level l
    s = 1 / 2^l
    return (fx*s, fy*s, (cx+0.5)*s - 0.5, (cy+0.5)*s - 0.5)

layer(k):                    // inverse depth of cost-volume layer k
    return xi_min + k * (xi_max - xi_min) / (S - 1)</code></pre>
    `);

    // ------------------------------------------------------------ quiz
    const imgs = [[640, 480], [320, 240], [1280, 720], [960, 540]];
    const fmt = L.fmt;
    L.quiz(root, "camera", [
      { id: "project", type: "num",
        gen: (r) => {
          const [w, h] = r.pick(imgs), f = r.pick([400, 500, 600, 800]);
          const cx = (w - 1) / 2, cy = (h - 1) / 2;
          const x = r.float(-1, 1, 1), y = r.float(-0.8, 0.8, 1), z = r.pick([2, 2.5, 4, 5]);
          const u = f * x / z + cx, v = f * y / z + cy;
          return {
            q: String.raw`A camera has $f_x=f_y=${f}$ and principal point $(c_x,c_y)=(${cx}, ${cy})$. Where does the point $\mathbf x = (${x}, ${y}, ${z})$ (camera coordinates) appear? Give $u, v$ to 1 decimal.`,
            answer: [u, v], labels: ["$u$", "$v$"], tol: 0.06,
            explain: String.raw`$u = f\,x/z + c_x = ${f}\cdot ${x}/${z} + ${cx} = ${fmt(f * x / z)} + ${cx} = ${fmt(u)}$; $v = ${f}\cdot ${y}/${z} + ${cy} = ${fmt(v)}$.`,
          };
        } },
      { id: "homog", type: "num",
        gen: (r) => {
          const u = r.int(10, 600), v = r.int(10, 400), s = r.pick([2, 4, 0.5, -2, 5, 2.5]);
          return {
            q: String.raw`A homogeneous pixel vector is $(${fmt(u * s)},\ ${fmt(v * s)},\ ${s})$. Which pixel $(u, v)$ does it represent?`,
            answer: [u, v], labels: ["$u$", "$v$"], tol: 0.01,
            explain: String.raw`Divide by the last entry: $(${fmt(u * s)}/${s},\ ${fmt(v * s)}/${s}) = (${u}, ${v})$. Scaling a homogeneous vector (even by a negative number) does not change the point it represents.`,
          };
        } },
      { id: "size", type: "num",
        gen: (r) => {
          const H = r.pick([1.5, 1.6, 1.8, 2, 2.4]), f = r.pick([400, 500, 600, 800, 1000]), hpx = r.pick([60, 80, 100, 120, 150, 200, 240]);
          const z = f * H / hpx;
          return {
            q: String.raw`A person ${H} m tall (standing upright, facing the camera) appears ${hpx} px tall in an image taken with $f = ${f}$ px. How far away (depth $z$, in m) are they? (2 decimals)`,
            answer: z, tol: 0.011,
            explain: String.raw`Similar triangles: image height $= f\,H/z$, so $z = f\,H/\text{height} = ${f}\cdot ${H}/${hpx} = ${fmt(z, 3)}$ m.`,
          };
        } },
      { id: "pp", type: "num",
        gen: (r) => {
          const [w, h] = r.pick([[640, 480], [320, 240], [1280, 720], [800, 600], [160, 120], [1920, 1080]]);
          return {
            q: String.raw`An image is ${w}×${h} pixels and pixel centres sit at integer coordinates (this implementation's convention). What is the exact image centre $(c_x, c_y)$?`,
            answer: [(w - 1) / 2, (h - 1) / 2], labels: ["$c_x$", "$c_y$"], tol: 0.001,
            explain: String.raw`Pixels span $u\in[-0.5,\ ${w - 0.5}]$, so the centre is $(${w}-1)/2 = ${(w - 1) / 2}$; likewise $c_y = (${h}-1)/2 = ${(h - 1) / 2}$.`,
          };
        } },
      { id: "backproject", type: "num",
        gen: (r) => {
          const f = r.pick([400, 500, 800]), cx = 319.5, cy = 239.5;
          const du = r.pick([-200, -160, -100, -80, 40, 80, 100, 200]), dv = r.pick([-160, -80, -40, 40, 80, 120]);
          const d = r.pick([0.25, 0.5, 2, 4, 0.2]);
          const u = cx + du, v = cy + dv;
          const X = [du / f / d, dv / f / d, 1 / d];
          return {
            q: String.raw`With $f_x=f_y=${f}$, $(c_x,c_y)=(319.5, 239.5)$, back-project pixel $\mathbf u = (${u}, ${v})$ at inverse depth $d = ${d}$: find $\mathbf x = \pi^{-1}(\mathbf u, d)$ (3 decimals).`,
            answer: X, labels: ["$x$", "$y$", "$z$"], tol: 0.002,
            explain: String.raw`$K^{-1}\dot{\mathbf u} = (${du}/${f},\ ${dv}/${f},\ 1) = (${fmt(du / f, 4)},\ ${fmt(dv / f, 4)},\ 1)$. Divide by $d=${d}$: $\mathbf x = (${fmt(X[0], 4)},\ ${fmt(X[1], 4)},\ ${fmt(X[2], 4)})$. Note $z = 1/d$.`,
          };
        } },
      { id: "normray", type: "mc",
        q: String.raw`What is $K^{-1}\dot{\mathbf u}$ geometrically?`,
        choices: [
          String.raw`The point on pixel $\mathbf u$'s ray whose depth is $z = 1$`,
          String.raw`The 3D point that pixel $\mathbf u$ actually sees`,
          String.raw`A unit-length vector (length 1) pointing along the ray`,
          String.raw`The pixel $\mathbf u$ measured from the image centre, in pixels`,
        ],
        answer: 0,
        explain: String.raw`Its last entry is 1, so it is the point at depth 1 on the ray; its length is generally more than 1. Any point on the ray is a multiple of it, which is why one pixel alone cannot give depth: we need $d$.` },
      { id: "ray", type: "mc",
        q: "Two different 3D points project to exactly the same pixel of one camera. What must be true?",
        choices: [
          "They lie on the same ray through the camera centre (one is a positive multiple of the other in camera coordinates)",
          "They have the same depth z",
          "They are the same distance from the camera centre",
          "They have the same x and y coordinates",
        ],
        answer: 0,
        explain: String.raw`Projection only keeps $x/z$ and $y/z$. Points $s\,\mathbf x$ for any $s>0$ give the same ratios, so they share the pixel. That is the depth DTAM has to recover from other views.` },
      { id: "fov", type: "num",
        gen: (r) => {
          const w = r.pick([640, 320, 1280, 800]), f = r.pick([300, 400, 500, 600, 800, 1000]);
          const hf = 2 * Math.atan(w / 2 / f) * 180 / Math.PI;
          return {
            q: String.raw`An image is ${w} px wide and $f_x = ${f}$ px. What is the horizontal field of view, in degrees? (1 decimal)`,
            answer: hf, tol: 0.15,
            explain: String.raw`$\text{hfov} = 2\arctan\frac{w/2}{f} = 2\arctan\frac{${w / 2}}{${f}} = 2\cdot ${fmt(hf / 2, 2)}° = ${fmt(hf, 2)}°$.`,
          };
        } },
      { id: "ffov", type: "num",
        gen: (r) => {
          const w = r.pick([640, 320, 1280]), hf = r.pick([60, 70, 90, 50, 80]);
          const f = w / 2 / Math.tan(hf / 2 * Math.PI / 180);
          return {
            q: String.raw`You want a virtual camera with a ${hf}° horizontal field of view and a ${w}-pixel-wide image. What focal length $f_x$ (px) do you need? (1 decimal)`,
            answer: f, tol: 0.15,
            explain: String.raw`$f = \frac{w/2}{\tan(\text{hfov}/2)} = \frac{${w / 2}}{\tan ${hf / 2}°} = \frac{${w / 2}}{${fmt(Math.tan(hf / 2 * Math.PI / 180), 4)}} = ${fmt(f, 2)}$ px.`,
          };
        } },
      { id: "zoom", type: "mc",
        q: String.raw`You double $f$ and keep the image size. What happens?`,
        choices: [
          String.raw`Objects appear twice as large and $\tan(\text{hfov}/2)$ halves`,
          String.raw`Objects appear twice as large and the field of view exactly halves`,
          String.raw`Objects appear four times as large (area) and the field of view is unchanged`,
          String.raw`The principal point moves twice as far from the corner`,
        ],
        answer: 0,
        explain: String.raw`Image offsets are $f\,x/z$, so they double. $\tan(\text{hfov}/2) = (w/2)/f$ halves; the angle itself halves only approximately (for small angles), because $\tan$ is not linear.` },
      { id: "disp", type: "num",
        gen: (r) => {
          const f = r.pick([400, 500, 600, 800]), b = r.pick([0.05, 0.1, 0.2, 0.3]), z = r.pick([0.5, 1, 2, 2.5, 4, 5, 8]);
          return {
            q: String.raw`Camera $m$ sits $b = ${b}$ m to the right of camera $r$, same orientation, $f=${f}$ px. A point is at depth $z=${z}$ m. By how many pixels does its image shift between $r$ and $m$?`,
            answer: f * b / z, tol: 0.01,
            explain: String.raw`Disparity $= f\,b\,\xi = f\,b/z = ${f}\cdot ${b}/${z} = ${fmt(f * b / z)}$ px. Halving the depth doubles the shift.`,
          };
        } },
      { id: "layers", type: "num",
        gen: (r) => {
          const xmin = r.pick([0, 0.1, 0.2, 0.25]), xmax = r.pick([1, 2, 4]), S = r.pick([5, 9, 11, 17, 21, 33]);
          const k = r.int(1, S - 1), step = (xmax - xmin) / (S - 1), xk = xmin + k * step;
          return {
            q: String.raw`A cost volume uses $S=${S}$ layers linearly spaced in inverse depth from $\xi_{min}=${xmin}$ to $\xi_{max}=${xmax}$ (layer 0 is $\xi_{min}$). What are $\xi_k$ and the depth $z_k$ of layer $k = ${k}$? (3 decimals)`,
            answer: [xk, 1 / xk], labels: [String.raw`$\xi_k$`, "$z_k$ (m)"], tol: 0.002,
            explain: String.raw`$\Delta\xi = (${xmax} - ${xmin})/(${S}-1) = ${fmt(step, 5)}$. $\xi_{${k}} = ${xmin} + ${k}\cdot ${fmt(step, 5)} = ${fmt(xk, 5)}$, so $z = 1/\xi = ${fmt(1 / xk, 4)}$ m.`,
          };
        } },
      { id: "step", type: "num",
        gen: (r) => {
          const f = r.pick([400, 500, 600]), b = r.pick([0.05, 0.1, 0.2]), S = r.pick([9, 17, 33, 65]), xmin = r.pick([0, 0.1, 0.2]), xmax = r.pick([2, 4]);
          const dxi = (xmax - xmin) / (S - 1);
          return {
            q: String.raw`Sideways baseline $b=${b}$ m, $f=${f}$ px, $S=${S}$ layers evenly spaced in inverse depth over $[${xmin}, ${xmax}]$. How many pixels apart along the epipolar line are the images of two neighbouring layers? (3 decimals)`,
            answer: f * b * dxi, tol: 0.002,
            explain: String.raw`Disparity is $f\,b\,\xi$, so a step $\Delta\xi$ moves $f\,b\,\Delta\xi$ px, the same for every layer. $\Delta\xi = ${fmt(xmax - xmin)}/${S - 1} = ${fmt(dxi, 5)}$, step $= ${f}\cdot ${b}\cdot ${fmt(dxi, 5)} = ${fmt(f * b * dxi, 4)}$ px.`,
          };
        } },
      { id: "whyxi", type: "multi",
        q: "Why does DTAM sample inverse depth ξ = 1/z evenly, rather than depth z? (select all)",
        choices: [
          "Equal steps in ξ move the projected point by equal pixel steps along the epipolar line, so no pixels are skipped and none is sampled many times",
          "A point at infinity has ξ = 0, so a finite range [0, ξ_max] covers everything from the nearest surface to the horizon",
          "Evenly spaced z would waste samples near the camera, where the image barely moves",
          "Inverse depth needs fewer bytes to store per layer",
        ],
        answer: [0, 1],
        explain: "Image motion is proportional to ξ. Evenly spaced z does the opposite of option 3: near the camera the image moves a lot per metre (samples too sparse), far away it barely moves (samples wasted). Storage per value is the same." },
      { id: "pyr", type: "num",
        gen: (r) => {
          const [w] = r.pick(imgs), f = r.pick([480, 500, 520, 640, 800]), l = r.int(1, 3);
          const cx = (w - 1) / 2 + r.pick([0, 3, -5, 7.5]);
          const s = 1 / 2 ** l;
          return {
            q: String.raw`Level 0 intrinsics: $f_x = ${f}$, $c_x = ${cx}$. Pyramid level $l = ${l}$ is made by halving the resolution ${l} time(s) with pixel-centred 2×2 averaging. Give $f_x$ and $c_x$ at that level (4 decimals).`,
            answer: [f * s, (cx + 0.5) * s - 0.5], labels: [String.raw`$f_x^{(${l})}$`, String.raw`$c_x^{(${l})}$`], tol: 0.0006,
            explain: String.raw`$f^{(l)} = f/2^l = ${f}/${2 ** l} = ${fmt(f * s, 4)}$. $c^{(l)} = (c+0.5)/2^l - 0.5 = ${fmt(cx + 0.5)}/${2 ** l} - 0.5 = ${fmt((cx + 0.5) * s - 0.5, 5)}$.`,
          };
        } },
    ]);
  },
});
