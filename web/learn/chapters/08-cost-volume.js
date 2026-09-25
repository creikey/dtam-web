// Chapter 8: the photometric cost volume (paper §2.2, §2.2.2, eqs 2-3).
DTAM.chapter({
  id: "costvolume",
  order: 8,
  title: "The cost volume",
  subtitle: "Test every depth of every pixel against many frames, and keep the average",
  minutes: 50,
  render(root, L) {
    // ================================================================ flatland world
    // A 2D world (x sideways, z forward) seen by 1D cameras that all face +z.
    // Reference camera r sits at the origin. Pixel u of a camera at (cx, cz)
    // looks along the ray (cx, cz) + t((u - CX)/F, 1).
    const W = 64, F = 64, CX = (W - 1) / 2;
    const S = 32, XI_MIN = 0.15, XI_MAX = 0.6, XI_STEP = (XI_MAX - XI_MIN) / (S - 1);
    const xiOf = (k) => XI_MIN + k * XI_STEP;
    const TAU = 2 * Math.PI;
    const FLAT_END = -0.72, TEX_END = 0.97;
    const texWall = (x) => x < FLAT_END ? 0.55
      : x < TEX_END ? 0.5 + 0.15 * Math.sin(9.3 * x + 1) + 0.13 * Math.sin(17.9 * x + 2.1) + 0.1 * Math.sin(26.3 * x + 0.4)
      : 0.5 + 0.3 * Math.sin(TAU * x / 0.3125);
    const texBox = (x) => 0.42 + 0.15 * Math.sin(19.2 * x + 0.3) + 0.13 * Math.sin(33.5 * x + 1.1) + 0.08 * Math.sin(47 * x + 2);
    const SEGS = [
      { x0: -3.5, z0: 4, x1: 3.5, z1: 4, tex: texWall, name: "wall" },
      { x0: -0.14, z0: 2.5, x1: 0.31, z1: 2.5, tex: texBox, name: "box" },
    ];
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    /** First surface hit by pixel u of a camera at (cx, cz): t = distance along z. */
    function hit(cx, cz, u) {
      const dx = (u - CX) / F, dz = 1;
      let best = Infinity, val = 0.3, seg = null;
      for (const s of SEGS) {
        const ex = s.x1 - s.x0, ez = s.z1 - s.z0, px = s.x0 - cx, pz = s.z0 - cz;
        const det = -dx * ez + ex * dz;
        if (Math.abs(det) < 1e-12) continue;
        const t = (-px * ez + ex * pz) / det, sp = (dx * pz - dz * px) / det;
        if (t > 1e-6 && sp >= 0 && sp <= 1 && t < best) { best = t; val = s.tex(cx + t * dx); seg = s; }
      }
      return { t: best, v: val, seg, x: cx + best * dx };
    }
    const gauss = (r) => Math.sqrt(-2 * Math.log(1 - r.next() * 0.999999)) * Math.cos(TAU * r.next());
    /** Renders a 1D image (3x supersampled) with optional sensor noise. */
    function renderImg(cx, cz, sigma, seed) {
      const r = L.rng(seed);
      const img = new Float32Array(W);
      for (let j = 0; j < W; j++) {
        const v = (hit(cx, cz, j - 1 / 3).v + hit(cx, cz, j).v + hit(cx, cz, j + 1 / 3).v) / 3;
        img[j] = clamp(v + sigma * gauss(r), 0, 1);
      }
      return img;
    }
    /** Linear interpolation with clamping, like the bilinear fetch in cost_update.wgsl. */
    function sample(img, x) {
      x = clamp(x, 0, W - 1);
      const i = Math.min(W - 2, Math.floor(x)), f = x - i;
      return img[i] * (1 - f) + img[i + 1] * f;
    }
    /** Eq. (3)'s warp in flatland: pixel u of r at inverse depth xi, seen by a camera at (bx, bz). */
    const um = (u, xi, cam) => (F * ((u - CX) / F - cam.x * xi)) / (1 - cam.z * xi) + CX;
    const trueXi = Array.from({ length: W }, (_, u) => 1 / hit(0, 0, u).t);
    const kindOf = (u) => {
      const h = hit(0, 0, u);
      if (h.seg.name === "box") return "textured box (nearer)";
      return h.x < FLAT_END ? "textureless wall" : h.x < TEX_END ? "textured wall" : "repeating stripes";
    };
    const frameCam = (k) => ({ x: 0.45 * Math.sin(0.9 * k + 0.3), z: 0.1 * Math.sin(1.7 * k) });

    // ================================================================ intro
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>Tracking gives us camera poses. Now we want <b>depth for every pixel</b>. DTAM does it by brute force: for each pixel, try many candidate depths, and for each candidate ask every other frame "does the colour match?".</p>
      <h3>Keyframes</h3>
      <p>Depth is estimated for special frames called <b>keyframes</b>. A keyframe $r$ stores:</p>
      <ul>
        <li>its <b>reference image</b> $I_r$ (RGB),</li>
        <li>its <b>pose</b> $T_{wr}$ (recall chapter 4: camera → world),</li>
        <li>an <b>inverse depth map</b> $\xi_r(\mathbf u)$: one number per pixel, the thing we want.</li>
      </ul>
      <p>While the keyframe is being built it also owns a <b>cost volume</b> $C_r$: for every pixel and every candidate depth, a running score of how badly that depth explains the other frames.</p>
      <h3>Candidate depths: inverse-depth layers</h3>
      <p>We pick a range $[\xi_{min}, \xi_{max}]$ and split it into $S$ evenly spaced <b>layers</b>. Recall from chapter 3: stepping evenly in inverse depth moves the matching point evenly along the epipolar line.</p>
      <div class="eq-card"><div class="eq-label">Layer k · paper §2.2.2</div>
      $$\xi_k = \xi_{min} + k\,\Delta\xi,\qquad \Delta\xi = \frac{\xi_{max}-\xi_{min}}{S-1},\qquad k = 0,\dots,S-1$$
      <div class="parts">
        <span>$S$</span><span>number of layers: 64 on desktop, 32 on phones in this implementation</span>
        <span>$\xi_k$</span><span>inverse depth tested by layer $k$; its depth is $z_k = 1/\xi_k$</span>
        <span>$\Delta\xi$</span><span>spacing between layers (the last layer is exactly $\xi_{max}$)</span>
      </div></div>
      <p><b>Example.</b> $\xi_{min}=0.2$, $\xi_{max}=1$, $S=5$: $\Delta\xi = 0.8/4 = 0.2$, layers $0.2, 0.4, 0.6, 0.8, 1.0$, depths $5, 2.5, 1.67, 1.25, 1$ m. Far layers are metres apart, near ones centimetres apart.</p>
    `);

    // ================================================================ W1: layers
    {
      const fig = L.figure(root, "<b>Inverse-depth layers.</b> Change $S$ and the range. Top: the depth $z_k$ of each layer along the pixel's ray (bunched near the camera). Bottom: where each layer lands in another frame, $f=500$ px, sideways baseline $b=0.1$ m: <i>evenly spaced</i>.");
      const c = L.canvas(fig.el, { aspect: 0.42, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const sS = L.slider(ctl, { label: "$S$", min: 4, max: 64, step: 4, value: 16, oninput: () => upd() });
      const sA = L.slider(ctl, { label: "$\\xi_{min}$", min: 0.05, max: 0.5, step: 0.05, value: 0.1, oninput: () => upd() });
      const sB = L.slider(ctl, { label: "$\\xi_{max}$", min: 0.6, max: 2, step: 0.1, value: 1, oninput: () => upd() });
      const fpx = 500, base = 0.1;
      const upd = () => {
        const n = sS.value, a = sA.value, b = sB.value, dxi = (b - a) / (n - 1);
        out.html = `Δξ = <b>${L.fmt(dxi, 4)}</b> · nearest two layers are <b>${L.fmt(1 / (b - dxi) - 1 / b, 3)} m</b> apart, farthest two <b>${L.fmt(1 / a - 1 / (a + dxi), 2)} m</b> · every step moves the match by f·b·Δξ = <b>${L.fmt(fpx * base * dxi, 2)} px</b>`;
        c.redraw();
      };
      c.draw = (ctx) => {
        const t = L.theme();
        const n = sS.value, a = sA.value, b = sB.value, dxi = (b - a) / (n - 1);
        const x0 = 12, x1 = c.w - 12;
        const zMax = 12;
        const y1 = c.h * 0.3, y2 = c.h * 0.78;
        // depth axis
        L.draw.line(ctx, x0, y1, x1, y1, t.line, 1.5);
        for (let z = 0; z <= zMax; z += 2) {
          const x = x0 + (z / zMax) * (x1 - x0);
          L.draw.line(ctx, x, y1 - 4, x, y1 + 4, t.faint, 1);
          L.draw.text(ctx, z + " m", x, y1 + 18, t.faint, { size: 11, align: "center" });
        }
        L.draw.text(ctx, "depth z of each layer along the ray", x0, y1 - 14, t.muted, { size: 12 });
        L.draw.dot(ctx, x0, y1, 5, t.fg);
        for (let k = 0; k < n; k++) {
          const z = 1 / (a + k * dxi);
          if (z > zMax) { L.draw.arrow(ctx, x1 - 16, y1, x1, y1, t.accent, 2, 7); continue; }
          L.draw.dot(ctx, x0 + (z / zMax) * (x1 - x0), y1, 3.2, t.accent);
        }
        // pixel axis in frame m
        const pxMax = fpx * base * 2;
        L.draw.line(ctx, x0, y2, x1, y2, t.line, 1.5);
        for (let p = 0; p <= pxMax; p += 20) {
          const x = x0 + (p / pxMax) * (x1 - x0);
          L.draw.line(ctx, x, y2 - 4, x, y2 + 4, t.faint, 1);
          L.draw.text(ctx, String(p), x, y2 + 18, t.faint, { size: 11, align: "center" });
        }
        L.draw.text(ctx, "shift of the match in frame m (px) = f·b·ξ", x0, y2 - 14, t.muted, { size: 12 });
        for (let k = 0; k < n; k++) {
          const p = fpx * base * (a + k * dxi);
          L.draw.dot(ctx, x0 + (p / pxMax) * (x1 - x0), y2, 3.2, t.accent3);
        }
      };
      upd();
    }

    // ================================================================ eq 3
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>The photometric error of one frame</h3>
      <p>Take pixel $\mathbf u$ of the keyframe and a guess $d$ for its inverse depth. Back-project it to 3D, move it into frame $m$, project it (chapter 4's pixel transfer), and compare colours:</p>
      <div class="eq-card"><div class="eq-label">Paper eq. (3) · photometric error</div>
      $$\rho_r(I_m,\mathbf u,d) = I_r(\mathbf u) - I_m\!\left(\pi\!\left(K\,T_{mr}\,\pi^{-1}(\mathbf u,d)\right)\right)$$
      <div class="parts">
        <span>$\pi^{-1}(\mathbf u,d)$</span><span>the 3D point on pixel $\mathbf u$'s ray at inverse depth $d$, in $r$'s frame</span>
        <span>$T_{mr}$</span><span>moves it into frame $m$'s coordinates ($T_{mr} = T_{wm}^{-1}T_{wr}$)</span>
        <span>$\pi(K\,\cdot)$</span><span>projects it to a pixel of frame $m$ (usually not an integer: sample with bilinear interpolation)</span>
        <span>$I_r(\mathbf u) - I_m(\dots)$</span><span>colour difference. Near 0 if $d$ is right and the point is visible in both frames</span>
      </div></div>
      <p><b>Flatland example</b> (the widgets below use a 2D world and 1D cameras). Focal length $f=100$, centre $c=50$, frame $m$ is $0.2$ m to the right of $r$. For a sideways move the transfer simplifies to $u_m = u - f\,b\,d$. Pixel $u=70$ at $d = 0.5$: $u_m = 70 - 100\cdot0.2\cdot0.5 = 60$. If $I_r(70)=0.62$ and $I_m(60)=0.55$ then $\rho = 0.07$.</p>
    `);

    // ================================================================ W2: one frame, drag the depth
    {
      const fig = L.figure(root, "<b>One frame, one pixel.</b> Top view of a flat world: a far wall and a nearer box. Drag the <b>orange</b> point along the reference pixel's ray (only the searched range is drawn thick), and drag camera <b>m</b>. Below: both 1D images, and $|\\rho|$ for this single frame at every layer. Try the stripes on the right: one frame gives several equally good depths.");
      const c = L.canvas(fig.el, { aspect: 0.8 });
      fig.add(c.el);
      const c2 = L.canvas(fig.el, { aspect: 0.5, scroll: true });
      fig.add(c2.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const cam = { x: 0.3, z: 0.05 };
      let d = 0.35;
      let imgR = renderImg(0, 0, 0, 1), imgM = renderImg(cam.x, cam.z, 0, 2);
      const sU = L.slider(ctl, { label: "pixel $u$", min: 0, max: W - 1, step: 1, value: 54, oninput: () => upd() });
      const geo = () => {
        const s = Math.min(c.w / 4.8, c.h / 5.1);
        const ox = c.w / 2, oy = c.h - (c.h - 5.1 * s) / 2 - 0.7 * s;
        return { s, X: (x) => ox + x * s, Y: (z) => oy - z * s, iX: (px) => (px - ox) / s, iZ: (py) => (oy - py) / s };
      };
      const upd = () => {
        const u = sU.value;
        const p = um(u, d, cam);
        const ir = imgR[u], im = sample(imgM, p);
        const inside = p >= 0 && p <= W - 1;
        out.html = `true ξ(u) = ${L.fmt(trueXi[u], 3)} (${kindOf(u)}) · d = <b>${L.fmt(d, 3)}</b> → u<sub>m</sub> = <b>${L.fmt(p, 2)}</b>${inside ? "" : " (outside the image!)"} · I<sub>r</sub>(u) = ${L.fmt(ir, 3)}, I<sub>m</sub>(u<sub>m</sub>) = ${L.fmt(im, 3)} · ρ = <b>${L.fmt(ir - im, 3)}</b>`;
        c.redraw(); c2.redraw();
      };
      L.drag(c, () => {
        const g = geo(), u = sU.value, dx = (u - CX) / F;
        return [{ x: g.X(cam.x), y: g.Y(cam.z) }, { x: g.X(dx / d), y: g.Y(1 / d) }];
      }, (i, p) => {
        const g = geo();
        if (i === 0) {
          cam.x = clamp(g.iX(p.x), -0.9, 0.9);
          cam.z = clamp(g.iZ(p.y), -0.5, 0.6);
          imgM = renderImg(cam.x, cam.z, 0, 2);
        } else {
          const dx = (sU.value - CX) / F, wx = g.iX(p.x), wz = g.iZ(p.y);
          const tz = (wx * dx + wz) / (dx * dx + 1);
          d = clamp(1 / Math.max(tz, 1e-3), XI_MIN, XI_MAX);
        }
        upd();
      });
      c.draw = (ctx) => {
        const t = L.theme(), g = geo(), u = sU.value, dx = (u - CX) / F;
        // surfaces, painted with their texture
        for (const s of SEGS) {
          const n = 260;
          for (let i = 0; i < n; i++) {
            const a0 = i / n, a1 = (i + 1) / n;
            const xa = s.x0 + (s.x1 - s.x0) * a0, xb = s.x0 + (s.x1 - s.x0) * a1;
            if (g.X(xb) < 0 || g.X(xa) > c.w) continue;
            L.draw.line(ctx, g.X(xa), g.Y(s.z0), g.X(xb) + 0.5, g.Y(s.z1), L.gray(s.tex((xa + xb) / 2)), 6);
          }
        }
        // frusta
        const drawCam = (cx, cz, col, label) => {
          for (const e of [0, W - 1]) {
            const ex = (e - CX) / F, tz = 4.3 - cz;
            L.draw.line(ctx, g.X(cx), g.Y(cz), g.X(cx + ex * tz), g.Y(cz + tz), t.line, 1, [4, 4]);
          }
          ctx.save(); ctx.fillStyle = col; ctx.beginPath();
          ctx.moveTo(g.X(cx), g.Y(cz)); ctx.lineTo(g.X(cx) - 9, g.Y(cz) + 12); ctx.lineTo(g.X(cx) + 9, g.Y(cz) + 12); ctx.closePath(); ctx.fill(); ctx.restore();
          L.draw.text(ctx, label, g.X(cx) + 12, g.Y(cz) + 14, col, { size: 13, bold: true });
        };
        drawCam(0, 0, t.accent, "r");
        // the reference ray + the searched segment
        const h = hit(0, 0, u);
        L.draw.line(ctx, g.X(0), g.Y(0), g.X(dx * h.t), g.Y(h.t), t.faint, 1.2);
        L.draw.line(ctx, g.X(dx / XI_MAX), g.Y(1 / XI_MAX), g.X(dx / XI_MIN), g.Y(1 / XI_MIN), t.accent, 3.5);
        L.draw.dot(ctx, g.X(dx * h.t), g.Y(h.t), 4, t.fg);
        // the guessed point and its line of sight to m
        const px = dx / d, pz = 1 / d;
        L.draw.line(ctx, g.X(px), g.Y(pz), g.X(cam.x), g.Y(cam.z), t.accent3, 1.5, [5, 4]);
        drawCam(cam.x, cam.z, t.accent3, "m");
        L.draw.handle(ctx, g.X(cam.x), g.Y(cam.z) + 6, t.accent3);
        L.draw.handle(ctx, g.X(px), g.Y(pz), t.accent2);
        L.draw.text(ctx, "ξ range", g.X(dx / XI_MIN) + 8, g.Y(1 / XI_MIN) + 4, t.accent, { size: 11 });
      };
      c2.draw = (ctx) => {
        const t = L.theme(), u = sU.value;
        const pl = 30, pr = 8, cw = (c2.w - pl - pr) / W, sh = 14;
        const strip = (img, y, label) => {
          for (let j = 0; j < W; j++) { ctx.fillStyle = L.gray(img[j]); ctx.fillRect(pl + j * cw, y, cw + 0.6, sh); }
          L.draw.text(ctx, label, pl - 4, y + 11, t.muted, { size: 12, align: "right" });
        };
        strip(imgR, 4, "Iᵣ");
        strip(imgM, 26, "Iₘ");
        const tri = (x, y, col) => { ctx.save(); ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - 5, y + 7); ctx.lineTo(x + 5, y + 7); ctx.closePath(); ctx.fill(); ctx.restore(); };
        tri(pl + (u + 0.5) * cw, 18, t.accent);
        const p = um(u, d, cam);
        if (p >= -0.5 && p <= W - 0.5) tri(pl + (p + 0.5) * cw, 40, t.accent2);
        // |rho| over the layers for this single frame
        const P = L.plot(c2, { x0: XI_MIN, x1: XI_MAX, y0: 0, y1: 0.8, pad: [58, 8, 26, pl] });
        P.axes(ctx, { xlabel: "inverse depth d", ylabel: "|ρ|", xticks: 3, yticks: 2 });
        const pts = [];
        for (let k = 0; k <= 120; k++) {
          const xi = XI_MIN + (k / 120) * (XI_MAX - XI_MIN);
          pts.push([P.X(xi), P.Y(Math.abs(imgR[u] - sample(imgM, um(u, xi, cam))))]);
        }
        L.draw.path(ctx, pts, t.accent3, 2);
        L.draw.line(ctx, P.X(trueXi[u]), P.Y(0), P.X(trueXi[u]), P.Y(0.8), t.fg, 1, [4, 4]);
        L.draw.dot(ctx, P.X(d), P.Y(Math.abs(imgR[u] - sample(imgM, um(u, d, cam)))), 5, t.accent2);
      };
      upd();
    }

    // ================================================================ eq 2 + running average
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Averaging many frames</h3>
      <p>One frame is ambiguous: many depths can match by accident. So DTAM averages the absolute error over every frame in $\mathcal I(r)$, the set of nearby frames that overlap the keyframe:</p>
      <div class="eq-card"><div class="eq-label">Paper eq. (2) · average photometric cost</div>
      $$C_r(\mathbf u,d) = \frac{1}{|\mathcal I(r)|}\sum_{m\in\mathcal I(r)} \big\|\rho_r(I_m,\mathbf u,d)\big\|_1$$
      <div class="parts">
        <span>$C_r(\mathbf u,d)$</span><span>one cell (<i>voxel</i>) of the cost volume: pixel $\mathbf u$, layer $d$</span>
        <span>$|\mathcal I(r)|$</span><span>how many frames were averaged</span>
        <span>$\|\cdot\|_1$</span><span>L1 norm: absolute value. For RGB, the sum of the three channels' absolute differences (what this implementation uses)</span>
      </div></div>
      <p><b>Example.</b> Four frames give $\rho = 0.10,\,-0.04,\,0.30,\,0.02$ at one voxel (the $0.30$ frame had the point hidden behind something). $C = (0.10+0.04+0.30+0.02)/4 = 0.115$. The absolute value matters: $+0.1$ and $-0.1$ must not cancel. RGB: $\rho = (0.05,-0.02,0.01)$ gives $\|\rho\|_1 = 0.08$.</p>
      <div class="key">You never need to keep old images. When frame $n$ arrives, update each voxel's average and throw the frame away: $$C_n = C_{n-1} + \frac{\|\rho_n\|_1 - C_{n-1}}{n}$$ This implementation stores the equivalent pair <b>(sum, count)</b> per voxel and divides when it needs $C$. E.g. $C_3 = 0.10$, $\|\rho_4\|_1 = 0.02$: $C_4 = 0.10 + (0.02-0.10)/4 = 0.08$; with sums, $(0.30 + 0.02)/4 = 0.08$.</div>
      <p><b>Memory.</b> A $256\times192$ keyframe with $S=32$ layers has $1.57$ million voxels: a 4-byte float sum plus a 1-byte counter each, ≈ 7.9 MB. The 8-bit counter is why a keyframe stops accepting frames at 250.</p>
      <h3>A shortcut: the epipolar segment is a straight line in $\xi$</h3>
      <p>With $T_{mr} = (R, \mathbf t)$ and $\pi^{-1}(\mathbf u,\xi) = \frac1\xi K^{-1}\dot{\mathbf u}$, multiply the 3D point by $\xi$ (scaling doesn't change which pixel it projects to):</p>
      <div class="eq-card"><div class="eq-label">Implementation · cost_update.wgsl</div>
      $$\xi\,K\,T_{mr}\,\pi^{-1}(\mathbf u,\xi) = \underbrace{K R K^{-1}\dot{\mathbf u}}_{\mathbf a} + \xi\,\underbrace{K\mathbf t}_{\mathbf b},\qquad \mathbf u_m = \pi(\mathbf a + \xi\,\mathbf b)$$
      <div class="parts">
        <span>$\mathbf a$</span><span>where the pixel lands at infinite depth ($\xi = 0$); one 3×3 product per pixel per frame</span>
        <span>$\mathbf b$</span><span>the same for every pixel; one per frame</span>
        <span>$\mathbf a + \xi\mathbf b$</span><span>each layer costs one multiply-add, then divide by the 3rd component</span>
      </div></div>
      <p><b>Example.</b> $K$ with $f=100$, $c=(50,40)$, $R=I$, $\mathbf t=(-0.2,0,0)$ (frame $m$ is 0.2 m to the right). Pixel $(70,40)$: $\mathbf a = (70,40,1)$, $\mathbf b = K\mathbf t = (-20,0,0)$. At $\xi=0.5$: $\mathbf a+\xi\mathbf b = (60,40,1)$ → pixel $(60,40)$, as in the flatland example.</p>
    `);

    // ================================================================ W4: fair averaging rule
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Which frames count for which pixel</h3>
      <p>Near layers shift further, so they leave frame $m$'s image first. If each voxel simply averaged the frames where it happened to land inside, the near layers of edge pixels would be averaged over only the small-baseline frames (which match everything well) and look falsely good. This implementation uses one rule per pixel instead:</p>
      <div class="key">Frame $m$ joins pixel $\mathbf u$'s row only if the <b>whole</b> segment $[\xi_{min},\xi_{max}]$ projects inside frame $m$. The image is convex, so checking the two end points is enough. Then every layer of the row is averaged over the same frames, exactly as eq. (2) intends.</div>
    `);
    {
      const fig = L.figure(root, "<b>Fair averaging.</b> Move camera $m$ sideways. Top: the reference image, with a bar under every pixel that this frame is allowed to update (green) or not (red). Bottom: frame $m$'s image, and the epipolar segment of the chosen pixel, dots at the 32 layers.");
      const c = L.canvas(fig.el, { aspect: 0.36, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const imgR = renderImg(0, 0, 0, 1);
      let imgM = null;
      const sB = L.slider(ctl, { label: "camera $m$ at $x$", min: -0.6, max: 0.6, step: 0.02, value: 0.36, oninput: () => { imgM = null; upd(); } });
      const sU = L.slider(ctl, { label: "pixel $u$", min: 0, max: W - 1, step: 1, value: 8, oninput: () => upd() });
      const ok = (u, cam) => {
        const a = um(u, XI_MIN, cam), b = um(u, XI_MAX, cam);
        return a >= 0 && a <= W - 1 && b >= 0 && b <= W - 1;
      };
      const upd = () => {
        const cam = { x: sB.value, z: 0 }, u = sU.value;
        let n = 0;
        for (let j = 0; j < W; j++) n += ok(j, cam);
        const a = um(u, XI_MIN, cam), b = um(u, XI_MAX, cam);
        out.html = `pixel ${u}: segment from u<sub>m</sub> = ${L.fmt(a, 1)} (ξ<sub>min</sub>) to ${L.fmt(b, 1)} (ξ<sub>max</sub>) → <b>${ok(u, cam) ? "inside: frame counts" : "leaves the image: frame skipped"}</b> · this frame updates <b>${n}</b> of ${W} pixels`;
        c.redraw();
      };
      c.draw = (ctx) => {
        const t = L.theme(), cam = { x: sB.value, z: 0 }, u = sU.value;
        if (!imgM) imgM = renderImg(cam.x, 0, 0, 2);
        // frame m axis spans -24 .. W+23 so out-of-image ends are visible
        const lo = -24, hi = W + 23, pl = 30, pr = 8;
        const X = (x) => pl + ((x - lo) / (hi - lo)) * (c.w - pl - pr);
        const cw = X(1) - X(0), sh = 16;
        const y0 = 8, y1 = c.h * 0.58;
        for (let j = 0; j < W; j++) {
          ctx.fillStyle = L.gray(imgR[j]); ctx.fillRect(X(j - 0.5), y0, cw + 0.6, sh);
          ctx.fillStyle = ok(j, cam) ? t.good : t.bad; ctx.fillRect(X(j - 0.5), y0 + sh + 2, cw + 0.6, 5);
          ctx.fillStyle = L.gray(imgM[j]); ctx.fillRect(X(j - 0.5), y1, cw + 0.6, sh);
        }
        L.draw.text(ctx, "Iᵣ", pl - 4, y0 + 12, t.muted, { size: 12, align: "right" });
        L.draw.text(ctx, "Iₘ", pl - 4, y1 + 12, t.muted, { size: 12, align: "right" });
        ctx.save(); ctx.strokeStyle = t.fg; ctx.lineWidth = 1.5; ctx.strokeRect(X(u - 0.5), y0 - 1, cw, sh + 2); ctx.restore();
        ctx.save(); ctx.strokeStyle = t.line; ctx.strokeRect(X(-0.5), y1 - 1, X(W - 0.5) - X(-0.5), sh + 2); ctx.restore();
        const a = um(u, XI_MIN, cam), b = um(u, XI_MAX, cam);
        const col = ok(u, cam) ? t.good : t.bad;
        const ys = y1 + sh + 12;
        L.draw.line(ctx, X(clamp(a, lo, hi)), ys, X(clamp(b, lo, hi)), ys, col, 3);
        for (let k = 0; k < S; k++) {
          const p = um(u, xiOf(k), cam);
          if (p >= lo && p <= hi) L.draw.dot(ctx, X(p), ys, 2.2, col);
        }
        L.draw.text(ctx, "ξmin", X(clamp(a, lo, hi)), ys + 17, t.muted, { size: 11, align: "center" });
        L.draw.text(ctx, "ξmax", X(clamp(b, lo, hi)), ys + 17, t.muted, { size: 11, align: "center" });
        L.draw.line(ctx, X(u), y0 + sh + 8, X(u), y1 - 2, t.faint, 1, [3, 3]);
      };
      upd();
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <p>Two more safeguards: a voxel seen by fewer than <b>3 frames</b> counts as <i>unobserved</i> (an average of one or two frames is not trustworthy), and unobserved voxels are ignored when looking for the minimum.</p>
      <h3>Algorithm: adding a frame</h3>
<pre><code>// once per new frame m (pose T_wm), every pixel u in parallel on the GPU
T_mr = inverse(T_wm) * T_wr
a = K R_mr K⁻¹ · (u, v, 1)          // lands here at ξ = 0
b = K t_mr                           // per-ξ shift
q0 = a + ξ_min·b ;  q1 = a + ξ_max·b
if q0.z ≤ 0 or q1.z ≤ 0: return      // behind camera m
if π(q0) or π(q1) outside image: return   // fair averaging
for k in 0 .. S-1:
    q = a + (ξ_min + k·Δξ)·b
    c = bilinear(I_m, π(q))          // RGB
    sum[u,k] += |I_r(u) − c|₁        // |ΔR| + |ΔG| + |ΔB|
    cnt[u,k] += 1
// whenever C is needed:
C(u,k) = sum[u,k] / cnt[u,k]  if cnt[u,k] ≥ 3, else unobserved</code></pre>
      <h3>Watch a cost volume form</h3>
      <p>Below is the same flatland scene, with $S=32$ layers and 64 pixels. Each column is one pixel's <b>cost row</b> $C(\mathbf u,\cdot)$. Add frames and watch the rows sharpen. Look at three kinds of pixel (the paper's Fig. 2):</p>
      <ul>
        <li><b>textureless</b> (left part of the wall): every depth matches equally, a flat row;</li>
        <li><b>textured</b> (middle, and the box): a single sharp minimum;</li>
        <li><b>repeating stripes</b> (right): each frame alone has several minima; frames with different baselines put the false ones at different depths, so the average keeps only the true one.</li>
      </ul>
    `);

    // ================================================================ W3: centrepiece
    {
      const fig = L.figure(root, "<b>Building the cost volume.</b> Press <b>+1 frame</b> a few times. Top: $I_r$, then the volume (pixel $u$ across, inverse depth up; dark = low cost). Dashed: true inverse depth. Dots: arg-min depth map. Tap or drag across the volume to pick a pixel; its row is plotted below, faint lines are single frames' $|\\rho|$, the shaded band is its trough. Grey cells are unobserved. Toggling the fair rule changes a few pixels near the left edge here; with real forward motion the effect is larger.");
      const cA = L.canvas(fig.el, { aspect: 0.62, scroll: true });
      fig.add(cA.el);
      const cB = L.canvas(fig.el, { aspect: 0.45, scroll: true });
      fig.add(cB.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ctl2 = L.controls(fig.el); fig.add(ctl2);
      const out = L.readout(fig.el); fig.add(out.el);
      const MAXF = 60;
      let nFrames = 1, sel = 54, fair = true, sigma = 0.02, minCount = 1;
      let imgR, imgs = [];
      let C = new Float64Array(W * S), cnt = new Int32Array(W * S), stats = [];
      const regen = () => {
        imgR = renderImg(0, 0, sigma, 1000);
        imgs = [];
        for (let k = 1; k <= MAXF; k++) imgs.push(null);
      };
      const frameImg = (k) => {
        if (!imgs[k - 1]) { const cam = frameCam(k); imgs[k - 1] = renderImg(cam.x, cam.z, sigma, 1000 + 7 * k); }
        return imgs[k - 1];
      };
      /** Does frame k contribute to voxel (u, layer)? */
      const uses = (u, cam, xi) => {
        if (fair) {
          const a = um(u, XI_MIN, cam), b = um(u, XI_MAX, cam);
          return a >= 0 && a <= W - 1 && b >= 0 && b <= W - 1;
        }
        const p = um(u, xi, cam);
        return p >= 0 && p <= W - 1;
      };
      const rebuild = () => {
        const sum = new Float64Array(W * S);
        cnt = new Int32Array(W * S);
        for (let k = 1; k <= nFrames; k++) {
          const cam = frameCam(k), im = frameImg(k);
          for (let u = 0; u < W; u++) {
            for (let l = 0; l < S; l++) {
              const xi = xiOf(l);
              if (!uses(u, cam, xi)) continue;
              sum[u * S + l] += Math.abs(imgR[u] - sample(im, um(u, xi, cam)));
              cnt[u * S + l]++;
            }
          }
        }
        C = new Float64Array(W * S);
        stats = [];
        for (let u = 0; u < W; u++) {
          let cmin = Infinity, cmax = 0, kmin = -1;
          for (let l = 0; l < S; l++) {
            const i = u * S + l;
            C[i] = cnt[i] >= minCount ? sum[i] / cnt[i] : NaN;
            if (isNaN(C[i])) continue;
            if (C[i] < cmin) { cmin = C[i]; kmin = l; }
            cmax = Math.max(cmax, C[i]);
          }
          let first = S, last = -1;
          const tau = Math.max(0.01, 0.1 * (cmax - cmin));
          if (kmin >= 0) {
            for (let l = 0; l < S; l++) {
              const v = C[u * S + l];
              if (!isNaN(v) && v <= cmin + tau) { first = Math.min(first, l); last = Math.max(last, l); }
            }
          }
          stats.push({ cmin, cmax, kmin, first, last, tau, width: kmin >= 0 ? last - first + 1 : S });
        }
      };
      const upd = (dataChanged = true) => {
        if (dataChanged) rebuild();
        const st = stats[sel];
        let used = 0;
        for (let k = 1; k <= nFrames; k++) {
          const cam = frameCam(k);
          if (fair ? uses(sel, cam, 0) : uses(sel, cam, xiOf(st.kmin >= 0 ? st.kmin : 0))) used++;
        }
        let se = 0, ne = 0;
        for (let u = 0; u < W; u++) if (stats[u].kmin >= 0) { se += (xiOf(stats[u].kmin) - trueXi[u]) ** 2; ne++; }
        const am = st.kmin >= 0 ? L.fmt(xiOf(st.kmin), 3) : "—";
        out.html = `frames added: <b>${nFrames}</b> · pixel u = <b>${sel}</b> (${kindOf(sel)}), ${fair ? "frames in its row" : "frames at its arg-min layer"}: ${used}<br>` +
          `arg min ξ = <b>${am}</b>, true ξ = ${L.fmt(trueXi[sel], 3)} · trough width = <b>${st.kmin >= 0 ? st.width : "—"}</b> layers → ${st.kmin >= 0 && st.width <= 6 ? "localised" : "not localisable"}<br>` +
          `RMS error of the whole arg-min map: <b>${ne ? L.fmt(Math.sqrt(se / ne), 3) : "—"}</b> over ${ne} pixels`;
        cA.redraw(); cB.redraw();
      };
      const geoA = () => {
        const pl = 34, pr = 6, top = 22, bot = cA.h - 20;
        const cw = (cA.w - pl - pr) / W, lh = (bot - top) / S;
        return { pl, pr, top, bot, cw, lh, X: (u) => pl + (u + 0.5) * cw, Y: (xi) => bot - ((xi - XI_MIN) / XI_STEP + 0.5) * lh };
      };
      cA.draw = (ctx) => {
        const t = L.theme(), g = geoA();
        for (let u = 0; u < W; u++) { ctx.fillStyle = L.gray(imgR[u]); ctx.fillRect(g.pl + u * g.cw, 4, g.cw + 0.6, 12); }
        for (let u = 0; u < W; u++) {
          for (let l = 0; l < S; l++) {
            const v = C[u * S + l];
            ctx.fillStyle = isNaN(v) ? t.panel2 : L.viridisish(v / 0.35);
            ctx.fillRect(g.pl + u * g.cw, g.bot - (l + 1) * g.lh, g.cw + 0.6, g.lh + 0.6);
          }
        }
        // true inverse depth
        const pts = [];
        for (let u = 0; u < W; u++) pts.push([g.pl + u * g.cw, g.Y(trueXi[u])], [g.pl + (u + 1) * g.cw, g.Y(trueXi[u])]);
        L.draw.path(ctx, pts, t.fg, 1.5, [4, 3]);
        for (let u = 0; u < W; u++) if (stats[u].kmin >= 0) L.draw.dot(ctx, g.X(u), g.Y(xiOf(stats[u].kmin)), 2.6, t.accent2);
        ctx.save(); ctx.strokeStyle = t.accent; ctx.lineWidth = 2; ctx.strokeRect(g.pl + sel * g.cw, g.top, g.cw, g.bot - g.top); ctx.restore();
        for (const xi of [0.15, 0.3, 0.45, 0.6]) L.draw.text(ctx, String(xi), g.pl - 4, g.Y(xi) + 4, t.faint, { size: 11, align: "right" });
        L.draw.text(ctx, "ξ", 6, g.top + 12, t.muted, { size: 12 });
        L.draw.text(ctx, "pixel u →", cA.w - 6, cA.h - 5, t.muted, { size: 12, align: "right" });
        L.draw.text(ctx, "0", g.pl, cA.h - 5, t.faint, { size: 11 });
      };
      cB.draw = (ctx) => {
        const t = L.theme(), st = stats[sel];
        const P = L.plot(cB, { x0: XI_MIN, x1: XI_MAX, y0: 0, y1: 0.6, pad: [14, 8, 26, 34] });
        P.axes(ctx, { xlabel: "inverse depth ξ", ylabel: `C(u=${sel}, ·)`, xticks: 3, yticks: 3 });
        if (st.kmin >= 0) {
          ctx.save(); ctx.globalAlpha = 0.16; ctx.fillStyle = t.accent;
          ctx.fillRect(P.X(xiOf(st.first) - XI_STEP / 2), P.Y(0.6), P.X(xiOf(st.last) + XI_STEP / 2) - P.X(xiOf(st.first) - XI_STEP / 2), P.Y(0) - P.Y(0.6));
          ctx.restore();
        }
        // single-frame curves (last 10 frames)
        for (let k = Math.max(1, nFrames - 9); k <= nFrames; k++) {
          const cam = frameCam(k), im = frameImg(k), pts = [];
          for (let l = 0; l < S; l++) {
            if (!uses(sel, cam, xiOf(l))) { L.draw.path(ctx, pts.splice(0), t.faint, 1); continue; }
            pts.push([P.X(xiOf(l)), P.Y(Math.min(0.6, Math.abs(imgR[sel] - sample(im, um(sel, xiOf(l), cam)))))]);
          }
          L.draw.path(ctx, pts, t.faint, 1);
        }
        let pts = [];
        for (let l = 0; l < S; l++) {
          const v = C[sel * S + l];
          if (isNaN(v)) { L.draw.path(ctx, pts, t.accent, 2.5); pts = []; continue; }
          pts.push([P.X(xiOf(l)), P.Y(Math.min(0.6, v))]);
        }
        L.draw.path(ctx, pts, t.accent, 2.5);
        L.draw.line(ctx, P.X(trueXi[sel]), P.Y(0), P.X(trueXi[sel]), P.Y(0.6), t.fg, 1, [4, 4]);
        if (st.kmin >= 0) L.draw.dot(ctx, P.X(xiOf(st.kmin)), P.Y(st.cmin), 5, t.accent2);
      };
      const pick = (e) => {
        const g = geoA(), p = cA.pos(e);
        if (p.y < g.top - 20) return;
        const u = clamp(Math.floor((p.x - g.pl) / g.cw), 0, W - 1);
        if (u !== sel) { sel = u; upd(false); }
      };
      let down = false;
      cA.el.addEventListener("pointerdown", (e) => { down = true; pick(e); });
      cA.el.addEventListener("pointermove", (e) => { if (down) pick(e); });
      for (const ev of ["pointerup", "pointercancel", "pointerleave"]) cA.el.addEventListener(ev, () => { down = false; });
      L.button(ctl, "+1 frame", () => { nFrames = Math.min(MAXF, nFrames + 1); upd(); }, "btn primary");
      L.button(ctl, "+10 frames", () => { nFrames = Math.min(MAXF, nFrames + 10); upd(); });
      L.button(ctl, "Reset", () => { nFrames = 1; upd(); });
      L.button(ctl, "textureless", () => { sel = 10; upd(false); });
      L.button(ctl, "textured", () => { sel = 44; upd(false); });
      L.button(ctl, "stripes", () => { sel = 54; upd(false); });
      L.toggle(ctl2, "fair rule (whole segment inside)", true, (v) => { fair = v; upd(); });
      L.slider(ctl2, { label: "noise $\\sigma$", min: 0, max: 0.08, step: 0.005, value: sigma, fmt: (v) => v.toFixed(3), oninput: (v) => { sigma = v; regen(); upd(); } });
      L.slider(ctl2, { label: "min views", min: 1, max: 5, step: 1, value: minCount, oninput: (v) => { minCount = v; upd(); } });
      regen();
      upd();
    }

    // ================================================================ reading the volume
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>The arg-min depth map, and why it is not enough</h3>
      <p>The obvious depth estimate is the best layer of each row: $\xi(\mathbf u) = \arg\min_d C(\mathbf u,d)$. It is fine where texture is strong, but in textureless regions the row is flat and <b>noise picks the minimum</b>, so the map is speckled (paper Fig. 3). More frames do not fix a flat row: there is simply no information there. Chapter 9 fixes this with smoothness.</p>
      <p>Why many frames still help:</p>
      <ul>
        <li><b>Occlusion</b>: a frame where the point is hidden gives a large $|\rho|$ at the true depth. With L1 and many frames it is one outlier among many (a squared error would let it dominate).</li>
        <li><b>Ambiguity</b>: each frame's false minima (repeating texture) sit at different depths for different baselines; only the true one lines up.</li>
        <li><b>Noise</b> averages down.</li>
      </ul>
      <h3>Per-pixel statistics</h3>
      <p>After accumulating, one pass over each row records what the solver needs (cost_minmax.wgsl):</p>
      <div class="eq-card"><div class="eq-label">Row statistics · this implementation</div>
      $$C_{min}(\mathbf u),\quad C_{max}(\mathbf u),\quad \arg\min_d C(\mathbf u,d),\quad \text{width} = k_{last}-k_{first}+1$$
      <div class="parts">
        <span>$C_{min}, C_{max}$</span><span>smallest and largest observed cost in the row (used in chapter 10 to limit the search)</span>
        <span>$\tau$</span><span>$\max(0.01,\ 0.1\,(C_{max}-C_{min}))$: "close to the minimum"</span>
        <span>$k_{first}, k_{last}$</span><span>first and last layer with $C \le C_{min}+\tau$</span>
        <span>width</span><span>the <b>trough width</b> in layers. ≤ 6 means <b>localised</b></span>
      </div></div>
      <p><b>Example.</b> Row $0.50, 0.30, 0.12, 0.10, 0.13, 0.40$: $C_{min}=0.10$, $C_{max}=0.50$, $\tau = \max(0.01, 0.04) = 0.04$, threshold $0.14$. Layers 2, 3, 4 qualify → width $= 4-2+1 = 3$: localised.</p>
      <p>The solver starts from the arg min only at localised pixels. Elsewhere it fills in a smooth interpolation of the localised neighbours (a coarse-to-fine "push–pull" average) and lets the regulariser refine it.</p>
      <h3>Choosing the depth range</h3>
      <p>The paper does not say how $[\xi_{min},\xi_{max}]$ is picked. This implementation uses what it already knows: the depth map the current 3D model predicts for the new keyframe's view. (The first keyframe uses the triangulated bootstrap points with the same margins and no cap.)</p>
      <div class="eq-card"><div class="eq-label">Depth range · slam.rs</div>
      $$\xi_{min} = \max\!\left(0.5\,p_{2},\ \tfrac{p_{50}}{6}\right),\qquad \xi_{max} = \min\!\left(1.6\,p_{98},\ 6\,p_{50}\right)$$
      <div class="parts">
        <span>$p_2, p_{50}, p_{98}$</span><span>2nd, 50th, 98th percentile of the predicted inverse depths (robust to a few wild values)</span>
        <span>$0.5,\ 1.6$</span><span>margins: allow surfaces a bit farther / nearer than predicted</span>
        <span>$6 = \sqrt{36}$</span><span>cap: $\xi_{max}/\xi_{min} \le 36$, so $S$ layers are not spread too thin</span>
      </div></div>
      <p><b>Example.</b> $p_2=0.2$, $p_{50}=0.4$, $p_{98}=0.9$: $\xi_{min}=\max(0.1, 0.067)=0.1$, $\xi_{max}=\min(1.44, 2.4)=1.44$.</p>
    `);

    // ================================================================ quiz
    const f3 = (v) => L.fmt(v, 3);
    const pn = (v) => (v < 0 ? `(${v})` : String(v));
    L.quiz(root, "costvolume", [
      { id: "layer", type: "num",
        gen: (r) => {
          const a = r.pick([0.1, 0.2, 0.25, 0.5]), span = r.pick([0.8, 1.6, 3.2]), S_ = r.pick([5, 9, 17, 33]);
          const k = r.int(1, S_ - 2), dxi = span / (S_ - 1), xi = a + k * dxi;
          return { q: String.raw`A keyframe uses $S=${S_}$ layers from $\xi_{min}=${a}$ to $\xi_{max}=${L.fmt(a + span, 3)}$. What is the inverse depth $\xi_k$ of layer $k=${k}$ (counting from 0), and its depth $z_k$ in metres?`,
            answer: [xi, 1 / xi], labels: [String.raw`$\xi_k$`, String.raw`$z_k$`], rtol: 0.005,
            explain: String.raw`$\Delta\xi = ${L.fmt(span, 3)}/${S_ - 1} = ${L.fmt(dxi, 5)}$, so $\xi_{${k}} = ${a} + ${k}\cdot${L.fmt(dxi, 5)} = ${L.fmt(xi, 5)}$ and $z = 1/\xi = ${L.fmt(1 / xi, 4)}$ m.` };
        } },
      { id: "epistep", type: "num",
        gen: (r) => {
          const f = r.pick([300, 400, 500, 600]), b = r.pick([0.05, 0.1, 0.2]), a = r.pick([0.1, 0.2]), span = r.pick([1.2, 1.8, 2.4]), S_ = r.pick([16, 32, 64]);
          const dxi = span / (S_ - 1), ans = f * b * dxi;
          return { q: String.raw`Frame $m$ is moved sideways by $b=${b}$ m from the keyframe (no rotation), $f=${f}$ px. The keyframe searches $S=${S_}$ layers in $[${a}, ${L.fmt(a + span, 3)}]$. How many pixels apart are consecutive layers' samples in frame $m$ (to 2 decimals)?`,
            answer: ans, rtol: 0.01,
            explain: String.raw`For a sideways move $u_m = u - f b\,\xi$, so a step $\Delta\xi$ moves the sample by $f b\,\Delta\xi = ${f}\cdot${b}\cdot${L.fmt(span, 3)}/${S_ - 1} = ${L.fmt(ans, 3)}$ px, the same for every layer.` };
        } },
      { id: "warp", type: "num",
        gen: (r) => {
          const f = r.pick([100, 200]), cx = r.pick([60, 80]), cy = r.pick([40, 50]);
          const u = cx + r.int(-30, 30), v = cy + r.int(-20, 20);
          const tx = r.pick([-0.2, -0.1, 0.1, 0.2]), ty = r.pick([0, 0.1, -0.1]), tz = r.pick([0, 0.1, -0.1, 0.2]);
          const xi = r.pick([0.5, 1, 2]);
          const b = [f * tx + cx * tz, f * ty + cy * tz, tz];
          const q = [u + xi * b[0], v + xi * b[1], 1 + xi * b[2]];
          return { q: String.raw`$K$ has $f_x=f_y=${f}$, $(c_x,c_y)=(${cx},${cy})$. $T_{mr}$ has $R=I$ and $\mathbf t=(${tx}, ${ty}, ${tz})$. Using $\mathbf u_m=\pi(\mathbf a+\xi\mathbf b)$ with $\mathbf a=KRK^{-1}\dot{\mathbf u}$, $\mathbf b=K\mathbf t$: where does pixel $(${u}, ${v})$ at $\xi=${xi}$ land in frame $m$ (2 decimals)?`,
            answer: [q[0] / q[2], q[1] / q[2]], labels: ["$u_m$", "$v_m$"], rtol: 0.002,
            explain: String.raw`$R=I$ so $\mathbf a=(${u}, ${v}, 1)$. $\mathbf b = K\mathbf t = (${f}\cdot${pn(tx)}+${cx}\cdot${pn(tz)},\ ${f}\cdot${pn(ty)}+${cy}\cdot${pn(tz)},\ ${tz}) = (${f3(b[0])}, ${f3(b[1])}, ${f3(b[2])})$. $\mathbf a+${xi}\mathbf b = (${f3(q[0])}, ${f3(q[1])}, ${f3(q[2])})$; divide by the last entry: $(${L.fmt(q[0] / q[2], 3)}, ${L.fmt(q[1] / q[2], 3)})$.` };
        } },
      { id: "rho", type: "num",
        gen: (r) => {
          const j = r.int(10, 90), fr = r.pick([0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8]);
          const A = r.float(0.2, 0.8, 2), B = r.float(0.2, 0.8, 2), Ir = r.float(0.2, 0.8, 2);
          const Im = A + fr * (B - A), ans = Ir - Im;
          return { q: String.raw`For some pixel and inverse depth, eq. (3)'s warp lands at $u_m = ${L.fmt(j + fr, 2)}$ in frame $m$ (1D image). $I_m(${j}) = ${A}$, $I_m(${j + 1}) = ${B}$, and $I_r(\mathbf u) = ${Ir}$. What is $\rho$ (with sign, 3 decimals)?`,
            answer: ans, tol: 0.0015,
            explain: String.raw`Linear interpolation: $I_m(${L.fmt(j + fr, 2)}) = ${A} + ${fr}\cdot(${B} - ${A}) = ${L.fmt(Im, 4)}$. $\rho = I_r - I_m = ${Ir} - ${L.fmt(Im, 4)} = ${L.fmt(ans, 4)}$.` };
        } },
      { id: "rgb", type: "num",
        gen: (r) => {
          const a = [0, 1, 2].map(() => r.float(0.1, 0.9, 2)), b = [0, 1, 2].map(() => r.float(0.1, 0.9, 2));
          const ans = a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);
          return { q: String.raw`This implementation compares colours. $I_r(\mathbf u) = (${a.join(", ")})$ (RGB) and the sampled colour in frame $m$ is $(${b.join(", ")})$. What is $\|\rho\|_1$?`,
            answer: ans, tol: 1e-3,
            explain: String.raw`$\|\rho\|_1 = |${a[0]}-${b[0]}| + |${a[1]}-${b[1]}| + |${a[2]}-${b[2]}| = ${L.fmt(Math.abs(a[0] - b[0]), 2)} + ${L.fmt(Math.abs(a[1] - b[1]), 2)} + ${L.fmt(Math.abs(a[2] - b[2]), 2)} = ${L.fmt(ans, 3)}$.` };
        } },
      { id: "avg", type: "num",
        gen: (r) => {
          const n = r.int(4, 6);
          const rs = Array.from({ length: n }, () => r.sign() * r.float(0.01, 0.12, 2));
          rs[r.int(0, n - 1)] = r.float(0.4, 0.7, 2);
          const ans = rs.reduce((s, v) => s + Math.abs(v), 0) / n;
          return { q: String.raw`One voxel $(\mathbf u, d)$ of a gray keyframe gets these photometric errors from the ${n} frames in $\mathcal I(r)$: $${rs.join(",\\ ")}$. What is $C_r(\mathbf u,d)$ by eq. (2) (3 decimals)?`,
            answer: ans, tol: 1.5e-3,
            explain: String.raw`Average the absolute values: $(${rs.map((v) => L.fmt(Math.abs(v), 2)).join(" + ")})/${n} = ${L.fmt(ans * n, 3)}/${n} = ${L.fmt(ans, 4)}$. (The big value is probably an occluded frame; L1 keeps its influence limited.)` };
        } },
      { id: "running", type: "num",
        gen: (r) => {
          const n = r.int(3, 19), Cold = r.float(0.05, 0.3, 2), rho = r.sign() * r.float(0.0, 0.5, 2);
          const ans = Cold + (Math.abs(rho) - Cold) / (n + 1);
          return { q: String.raw`A voxel has average cost $C=${Cold}$ after $${n}$ frames. Frame ${n + 1} arrives with $\rho = ${rho}$ at this voxel. What is the new average (4 decimals)?`,
            answer: ans, tol: 2e-4,
            explain: String.raw`$C_{new} = C + (|\rho| - C)/(n+1) = ${Cold} + (${L.fmt(Math.abs(rho), 2)} - ${Cold})/${n + 1} = ${L.fmt(ans, 5)}$. Same as $(${n}\cdot${Cold} + ${L.fmt(Math.abs(rho), 2)})/${n + 1}$ from the stored sum and count.` };
        } },
      { id: "argmin", type: "num",
        gen: (r) => {
          const n = 8, a = r.pick([0.1, 0.2, 0.3]), step = r.pick([0.05, 0.1, 0.2]);
          const km = r.int(0, n - 1);
          const row = Array.from({ length: n }, (_, k) => k === km ? r.float(0.05, 0.1, 2) : r.float(0.13, 0.6, 2));
          const ans = a + km * step;
          return { q: String.raw`A cost row has layers $\xi_k = ${a} + ${step}k$, $k = 0..7$, with costs $${row.join(",\\ ")}$. What inverse depth does the arg-min depth map give this pixel? What depth is that?`,
            answer: [ans, 1 / ans], labels: [String.raw`$\xi$`, "$z$"], rtol: 0.003,
            explain: String.raw`The smallest cost ${row[km]} is at $k=${km}$: $\xi = ${a} + ${step}\cdot${km} = ${L.fmt(ans, 3)}$, $z = 1/\xi = ${L.fmt(1 / ans, 4)}$ m.` };
        } },
      { id: "trough", type: "num",
        gen: (r) => {
          for (;;) {
            const n = 10, km = r.int(2, 7);
            const row = Array.from({ length: n }, (_, k) => Math.round(100 * Math.min(0.9, 0.1 + 0.03 * Math.abs(k - km) * r.float(0.3, 3, 2) + r.float(0, 0.04, 2))) / 100);
            row[km] = 0.1;
            let cmin = Math.min(...row), cmax = Math.max(...row);
            const tau = Math.max(0.01, 0.1 * (cmax - cmin)), th = cmin + tau;
            if (row.some((v) => Math.abs(v - th) < 0.004)) continue;
            let first = n, last = -1;
            row.forEach((v, k) => { if (v <= th) { first = Math.min(first, k); last = Math.max(last, k); } });
            const w = last - first + 1;
            return { q: String.raw`A cost row (layers 0–9): $${row.join(",\\ ")}$. Using this implementation's rule $\tau=\max(0.01,\,0.1(C_{max}-C_{min}))$, what is the trough width in layers?`,
              answer: w, tol: 0.01,
              explain: String.raw`$C_{min} = ${cmin}$, $C_{max} = ${cmax}$, $\tau = \max(0.01, ${L.fmt(0.1 * (cmax - cmin), 4)}) = ${L.fmt(tau, 4)}$, threshold $${L.fmt(th, 4)}$. Layers at or below it: first $${first}$, last $${last}$ → width $${last} - ${first} + 1 = ${w}$ (${w <= 6 ? "localised" : "not localised"}). Values in between count even if above the threshold.` };
          }
        } },
      { id: "fair", type: "num",
        gen: (r) => {
          const Wd = r.pick([64, 80, 100]);
          const rows = [];
          let cnt = 0;
          for (let i = 0; i < 5; i++) {
            const ins = r.next() < 0.5;
            let a, b;
            if (ins) { a = r.float(1, Wd - 12, 1); b = +(a + r.float(2, 10, 1)).toFixed(1); }
            else if (r.next() < 0.5) { a = r.float(-8, -0.3, 1); b = +(a + r.float(4, 12, 1)).toFixed(1); }
            else { a = r.float(Wd - 9, Wd - 2, 1); b = +(a + r.float(3, 10, 1)).toFixed(1); }
            if (r.next() < 0.5) [a, b] = [b, a];
            const ok = a >= 0 && a <= Wd - 1 && b >= 0 && b <= Wd - 1;
            cnt += ok;
            rows.push(`frame ${i + 1}: $u_m(\\xi_{min}) = ${a}$, $u_m(\\xi_{max}) = ${b}$`);
          }
          return { q: String.raw`Images are $${Wd}$ pixels wide (valid $u$ from $0$ to $${Wd - 1}$). For one keyframe pixel, the ends of its epipolar segment in five frames are:<br>${rows.join("<br>")}<br>Under this implementation's fair-averaging rule, how many of these frames contribute to this pixel's cost row?`,
            answer: cnt, tol: 0.01,
            explain: String.raw`A frame counts only if <i>both</i> ends lie in $[0, ${Wd - 1}]$. Checking each row gives $${cnt}$ frames. Every layer of the row is then averaged over exactly those frames.` };
        } },
      { id: "range", type: "num",
        gen: (r) => {
          const p2 = r.pick([0.05, 0.1, 0.2, 0.3]), p50 = r.pick([0.4, 0.5, 0.6, 0.8]), p98 = +(p50 + r.pick([0.2, 0.5, 1, 2, 3])).toFixed(2);
          const lo = Math.max(0.5 * p2, p50 / 6), hi = Math.min(1.6 * p98, 6 * p50);
          return { q: String.raw`The model predicts inverse-depth percentiles $p_2=${p2}$, $p_{50}=${p50}$, $p_{98}=${p98}$ for a new keyframe. Using this implementation's rule, what search range $[\xi_{min},\xi_{max}]$ does it get (3 decimals)?`,
            answer: [lo, hi], labels: [String.raw`$\xi_{min}$`, String.raw`$\xi_{max}$`], rtol: 0.005,
            explain: String.raw`$\xi_{min} = \max(0.5\cdot${p2},\ ${p50}/6) = \max(${L.fmt(0.5 * p2, 4)}, ${L.fmt(p50 / 6, 4)}) = ${L.fmt(lo, 4)}$; $\xi_{max} = \min(1.6\cdot${p98},\ 6\cdot${p50}) = \min(${L.fmt(1.6 * p98, 4)}, ${L.fmt(6 * p50, 4)}) = ${L.fmt(hi, 4)}$.` };
        } },
      { id: "memory", type: "num",
        gen: (r) => {
          const w = r.pick([256, 320, 512]), h = r.pick([144, 192, 240]), S_ = r.pick([32, 64]);
          const ans = (w * h * S_ * 5) / 1e6;
          return { q: String.raw`A keyframe is $${w}\times${h}$ pixels with $S=${S_}$ layers. Each voxel stores a 4-byte float sum and a 1-byte counter. How many megabytes ($10^6$ bytes) is the cost volume (2 decimals)?`,
            answer: ans, rtol: 0.01,
            explain: String.raw`$${w}\cdot${h}\cdot${S_} = ${w * h * S_}$ voxels × 5 bytes $= ${w * h * S_ * 5}$ bytes $\approx ${L.fmt(ans, 3)}$ MB. It does not grow with the number of frames.` };
        } },
      { id: "kinds", type: "mc",
        q: "A pixel's cost row stays almost flat (every layer within the noise of the others) even after 50 frames. Where is the pixel most likely?",
        choices: ["In a textureless region, e.g. a blank wall", "On a strong corner", "On a repeating stripe pattern", "On a surface whose true depth is exactly $\\xi_{min}$"],
        answer: 0,
        explain: "If the surface around the pixel has the same colour everywhere, every depth samples the same colour: no depth is preferred (paper Fig. 2a). Stripes give several distinct dips, corners one sharp dip." },
      { id: "many", type: "multi",
        q: "Which statements about averaging many frames are true? (select all)",
        choices: [
          "A frame in which the point is occluded is just one outlier; with the L1 norm its effect on the average stays limited",
          "Different baselines put the false minima of repeating texture at different depths, so they do not line up in the average",
          "With enough frames, textureless pixels get a sharp minimum",
          "The running average means old frames need not be stored",
          "Averaging many frames removes the need for regularisation"],
        answer: [0, 1, 3],
        explain: "Flat rows stay flat: no number of frames adds texture. That, and remaining noise, is why chapter 9 adds a smoothness term." },
      { id: "fairwhy", type: "mc",
        q: "Why does this implementation let a frame update a pixel's row only if the whole segment $[\\xi_{min},\\xi_{max}]$ lands inside that frame?",
        choices: [
          "Otherwise near layers (which leave the image first) would be averaged over fewer frames, mostly small-baseline ones that match everything, and look falsely good",
          "Samples outside the image would crash the GPU",
          "To save memory in the cost volume",
          "Because eq. (3) is undefined for pixels near the image border"],
        answer: 0,
        explain: "Eq. (2) assumes the same set $\\mathcal I(r)$ for every depth of a row. Mixing different frame sets per layer biases the arg min toward near depths." },
      { id: "minviews", type: "mc",
        q: "In this implementation, a voxel $(\\mathbf u, k)$ has been updated by only 2 frames. What happens to it when the solver looks for the minimum of the row?",
        choices: [
          "It is treated as unobserved and skipped, because fewer than 3 frames is not a trustworthy average",
          "It is used normally: any average is fine",
          "Its cost is doubled to penalise it",
          "The whole keyframe is discarded"],
        answer: 0,
        explain: "The minimum voxel count is 3 (min_voxel_views). Unobserved voxels are ignored in $C_{min}$, $C_{max}$ and the arg min." },
      { id: "linear", type: "mc",
        q: "Why are the layers evenly spaced in inverse depth $\\xi$ rather than in depth $z$?",
        choices: [
          "Equal steps in $\\xi$ are equal steps along the epipolar line, and the range can reach very far depths",
          "Equal steps in $\\xi$ are equal steps in depth",
          "The photometric error is a linear function of $\\xi$",
          "It needs fewer layers near the camera"],
        answer: 0,
        explain: "For a sideways baseline $u_m = u - fb\\xi$: the sample moves linearly in $\\xi$. Near depths get many layers and far depths few, matching how much each moves in the image." },
      { id: "noisy", type: "mc",
        q: "Why is the arg-min depth map speckled in blank regions even with many frames?",
        choices: [
          "The row there is nearly flat, so image noise decides which layer is lowest",
          "The fair-averaging rule removes those pixels",
          "Bilinear interpolation blurs the image",
          "The layers are too close together"],
        answer: 0,
        explain: "No texture, no information: every depth fits equally well. Chapter 9's regulariser fills such pixels from their neighbours." },
    ]);
  },
});
