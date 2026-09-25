// Chapter 6: sparse corner tracking (KLT): Shi–Tomasi corners, pyramidal
// Lucas–Kanade, forward–backward check. Mirrors crates/dtam-core/src/tracker/.
DTAM.chapter({
  id: "klt",
  order: 6,
  title: "Tracking corners (KLT)",
  subtitle: "Shi–Tomasi corners and pyramidal Lucas–Kanade, as used to bootstrap DTAM",
  minutes: 55,
  render(root, L) {
    // ------------------------------------------------------------ image helpers (tracker maths)
    /** Bilinear sample with edge clamping (klt.wgsl `sample`). Pixel centres at integers. */
    const sampleI = (img, x, y) => {
      const w = img.w, h = img.h, d = img.data;
      x = x < 0 ? 0 : x > w - 1 ? w - 1 : x;
      y = y < 0 ? 0 : y > h - 1 ? h - 1 : y;
      const x0 = Math.floor(x), y0 = Math.floor(y);
      const x1 = x0 + 1 < w ? x0 + 1 : w - 1, y1 = y0 + 1 < h ? y0 + 1 : h - 1;
      const fx = x - x0, fy = y - y0;
      const a = d[y0 * w + x0], b = d[y0 * w + x1], c = d[y1 * w + x0], e = d[y1 * w + x1];
      return (a + (b - a) * fx) * (1 - fy) + (c + (e - c) * fx) * fy;
    };
    /** 2× decimation with a separable [1 3 3 1]/8 filter (downsample.wgsl). */
    const down = (img) => {
      const W = Math.ceil(img.w / 2), H = Math.ceil(img.h / 2), out = new Float32Array(W * H);
      const px = (x, y) => img.data[Math.max(0, Math.min(img.h - 1, y)) * img.w + Math.max(0, Math.min(img.w - 1, x))];
      const k = [1, 3, 3, 1];
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        let acc = 0;
        for (let j = 0; j < 4; j++) { let row = 0; for (let i = 0; i < 4; i++) row += k[i] * px(2 * x - 1 + i, 2 * y - 1 + j); acc += k[j] * row; }
        out[y * W + x] = acc / 64;
      }
      return { w: W, h: H, data: out };
    };
    const pyramid = (img, n) => { const p = [img]; while (p.length < n) p.push(down(p[p.length - 1])); return p; };
    /** min "stretch" of [a b; b c] (smallest eigenvalue). */
    const lmin = (a, b, c) => 0.5 * (a + c - Math.sqrt((a - c) * (a - c) + 4 * b * b));
    const lmax = (a, b, c) => 0.5 * (a + c + Math.sqrt((a - c) * (a - c) + 4 * b * b));
    /**
     * Pyramidal Lucas–Kanade (klt.wgsl `track`). A, B: pyramids of the "from"
     * and "to" frames. Returns final position, ok, residual, and the path of
     * estimates (in level-0 pixels) for drawing.
     */
    const lk = (A, B, p0, guess, nLev, R = 7, maxIt = 20) => {
      const top = nLev - 1, n = (2 * R + 1) ** 2;
      let dx = (guess[0] - p0[0]) / 2 ** top, dy = (guess[1] - p0[1]) / 2 ** top;
      const T = new Float64Array(n), GX = new Float64Array(n), GY = new Float64Array(n);
      const path = [], its = [];
      let ok = true;
      for (let l = top; l >= 0; l--) {
        const s = 1 / 2 ** l, plx = (p0[0] + 0.5) * s - 0.5, ply = (p0[1] + 0.5) * s - 0.5;
        const a = A[l], b = B[l];
        let gxx = 0, gxy = 0, gyy = 0, k = 0;
        for (let y = -R; y <= R; y++) for (let x = -R; x <= R; x++, k++) {
          const qx = plx + x, qy = ply + y;
          T[k] = sampleI(a, qx, qy);
          GX[k] = 0.5 * (sampleI(a, qx + 1, qy) - sampleI(a, qx - 1, qy));
          GY[k] = 0.5 * (sampleI(a, qx, qy + 1) - sampleI(a, qx, qy - 1));
          gxx += GX[k] * GX[k]; gxy += GX[k] * GY[k]; gyy += GY[k] * GY[k];
        }
        const det = gxx * gyy - gxy * gxy;
        if (lmin(gxx, gxy, gyy) / n < 1e-7 || Math.abs(det) < 1e-12) { ok = false; its.push(0); break; }
        const ia = gyy / det, ib = -gxy / det, ic = gxx / det;
        let it = 0;
        while (it < maxIt) {
          let bx = 0, by = 0; k = 0;
          for (let y = -R; y <= R; y++) for (let x = -R; x <= R; x++, k++) {
            const diff = T[k] - sampleI(b, plx + x + dx, ply + y + dy);
            bx += diff * GX[k]; by += diff * GY[k];
          }
          const ddx = ia * bx + ib * by, ddy = ib * bx + ic * by;
          dx += ddx; dy += ddy; it++;
          path.push({ l, x: p0[0] + dx * 2 ** l, y: p0[1] + dy * 2 ** l });
          if (ddx * ddx + ddy * ddy < 1e-4) break;
        }
        its.push(it);
        if (l > 0) { dx *= 2; dy *= 2; }
      }
      const pos = [p0[0] + dx, p0[1] + dy];
      let residual = 0;
      if (ok) {
        for (let y = -R; y <= R; y++) for (let x = -R; x <= R; x++) residual += Math.abs(sampleI(A[0], p0[0] + x, p0[1] + y) - sampleI(B[0], pos[0] + x, pos[1] + y));
        residual /= n;
        if (pos[0] < 0 || pos[1] < 0 || pos[0] > A[0].w - 1 || pos[1] > A[0].h - 1) ok = false;
      }
      return { pos, ok, residual, path, its };
    };
    /** Forward, then backward from the result (klt.wgsl cs_main). */
    const trackFB = (A, B, p0, nLev, R = 7) => {
      const f = lk(A, B, p0, p0, nLev, R);
      if (!f.ok) return { ...f, fb: Infinity, back: null };
      const b = lk(B, A, f.pos, p0, nLev, R);
      if (!b.ok) return { ...f, ok: false, fb: Infinity, back: null };
      return { ...f, back: b.pos, fb: Math.hypot(b.pos[0] - p0[0], b.pos[1] - p0[1]) };
    };
    /** Shi–Tomasi score map, 5×5 window, mean (corners.wgsl); 0 within `border` of the edge. */
    const scoreMap = (img, border = 12) => {
      const { w, h, data } = img, out = new Float32Array(w * h), Rr = 2, nn = 25;
      const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
      for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
        gx[y * w + x] = 0.5 * (data[y * w + x + 1] - data[y * w + x - 1]);
        gy[y * w + x] = 0.5 * (data[(y + 1) * w + x] - data[(y - 1) * w + x]);
      }
      const b = Math.max(border, Rr + 1);
      for (let y = b; y + b < h; y++) for (let x = b; x + b < w; x++) {
        let a = 0, bb = 0, c = 0;
        for (let v = -Rr; v <= Rr; v++) for (let u = -Rr; u <= Rr; u++) {
          const i = (y + v) * w + x + u; a += gx[i] * gx[i]; bb += gx[i] * gy[i]; c += gy[i] * gy[i];
        }
        out[y * w + x] = lmin(a / nn, bb / nn, c / nn);
      }
      return out;
    };
    /** Tile maxima → threshold → strongest-first with a min-distance grid (tracker/mod.rs). */
    const detect = (img, score, quality = 0.02, minDist = 12, TILE = 16, maxN = 800) => {
      const { w, h } = img, cands = [];
      for (let ty = 0; ty < Math.ceil(h / TILE); ty++) for (let tx = 0; tx < Math.ceil(w / TILE); tx++) {
        let best = 0, bx = tx * TILE, by = ty * TILE;
        for (let y = ty * TILE; y < Math.min(h, ty * TILE + TILE); y++) for (let x = tx * TILE; x < Math.min(w, tx * TILE + TILE); x++) {
          if (score[y * w + x] > best) { best = score[y * w + x]; bx = x; by = y; }
        }
        cands.push({ x: bx, y: by, s: best });
      }
      const best = Math.max(0, ...cands.map((c) => c.s)), thr = Math.max(best * quality, 1e-5);
      const sorted = cands.filter((c) => c.s >= thr).sort((a, b) => b.s - a.s);
      const cell = Math.max(1, minDist), gw = Math.ceil(w / cell) + 1, gh = Math.ceil(h / cell) + 1, occ = new Uint8Array(gw * gh);
      const kept = [], blocked = [];
      for (const c of sorted) {
        if (kept.length >= maxN) break;
        const cx = Math.min(gw - 1, Math.floor(c.x / cell)), cy = Math.min(gh - 1, Math.floor(c.y / cell));
        let free = true;
        for (let y = Math.max(0, cy - 1); y <= Math.min(gh - 1, cy + 1); y++) for (let x = Math.max(0, cx - 1); x <= Math.min(gw - 1, cx + 1); x++) if (occ[y * gw + x]) free = false;
        if (!free) { blocked.push(c); continue; }
        occ[cy * gw + cx] = 1;
        kept.push(c);
      }
      return { kept, blocked, below: cands.filter((c) => c.s < thr), thr, best, nTiles: cands.length };
    };
    /** Offscreen canvas from RGBA bytes, for fast drawImage. */
    const toCanvas = (w, h, rgba) => {
      const cv = document.createElement("canvas");
      cv.width = w; cv.height = h;
      const cx = cv.getContext("2d");
      const id = cx.createImageData(w, h);
      id.data.set(rgba);
      cx.putImageData(id, 0, 0);
      return cv;
    };
    const grayRGBA = (img) => {
      const o = new Uint8ClampedArray(img.w * img.h * 4);
      for (let i = 0; i < img.w * img.h; i++) { const g = Math.round(Math.max(0, Math.min(1, img.data[i])) * 255); o[4 * i] = o[4 * i + 1] = o[4 * i + 2] = g; o[4 * i + 3] = 255; }
      return o;
    };
    /** Fit an image of iw×ih into a box; returns mapping for pixel centres. */
    const fitBox = (bx, by, bw, bh, iw, ih) => {
      const s = Math.min(bw / iw, bh / ih), ox = bx + (bw - iw * s) / 2, oy = by + (bh - ih * s) / 2;
      return { s, ox, oy, X: (u) => ox + (u + 0.5) * s, Y: (v) => oy + (v + 0.5) * s, U: (x) => (x - ox) / s - 0.5, V: (y) => (y - oy) / s - 0.5 };
    };
    const blit = (ctx, cv, f, iw, ih) => { ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(cv, f.ox, f.oy, iw * f.s, ih * f.s); ctx.restore(); };
    const box = (ctx, f, u, v, R, color, width = 2) => {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.strokeRect(f.X(u - R) - f.s / 2, f.Y(v - R) - f.s / 2, (2 * R + 1) * f.s, (2 * R + 1) * f.s); ctx.restore();
    };
    const narrow = (window.innerWidth || 800) < 560;

    // ================================================================ 1. aperture problem
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>DTAM's dense tracker needs a 3D model, and at start-up there is none. So the first few seconds run a classic <b>sparse</b> pipeline: follow a few hundred distinctive points from frame to frame (this chapter), then recover camera motion and 3D structure from them (chapter 7). The point tracker is <b>KLT</b> (Kanade–Lucas–Tomasi): Shi–Tomasi corners + Lucas–Kanade tracking.</p>
      <h3>Which points can be tracked? The aperture problem</h3>
      <p>Track a point by following the small square <b>window</b> of pixels around it. On a flat wall every shift of the window looks the same. On a straight edge, shifts <i>along</i> the edge look the same: only motion across the edge is measurable. Only at a <b>corner</b> (or in rich texture) does every direction of shift change the window's content.</p>
    `);
    {
      const fig = L.figure(root, "<b>The aperture problem.</b> Drag the window (the square) over the flat background, along an edge, onto a corner, into the texture. Middle: the error $\\mathrm{SSD}(\\delta)$ of shifting the window by $\\delta$ (±4 px; dark = the same, bright = different). Right: gradients of the window's pixels (dots) and the stretch $\\mathbf n^\\top M\\mathbf n$ in every direction $\\mathbf n$ (curve); the arrow is the weakest direction.");
      const SW = 64, SH = 40, syn = new Float32Array(SW * SH), rr = L.rng(5);
      for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
        let v = 0.22;
        const ca = Math.cos(0.35), sa = Math.sin(0.35), dx = x - 19, dy = y - 20, qx = ca * dx + sa * dy, qy = -sa * dx + ca * dy;
        const sd = Math.max(Math.abs(qx), Math.abs(qy)) - 10;
        v += 0.55 * Math.min(1, Math.max(0, 0.5 - sd / 1.5));
        if (x >= 42) v = 0.5 + 0.2 * Math.sin(x * 1.1 + y * 0.4) * Math.cos(y * 0.9 - x * 0.3) + 0.08 * Math.sin(0.07 * x * y);
        syn[y * SW + x] = v + (rr.next() - 0.5) * 0.01;
      }
      const img = { w: SW, h: SH, data: syn };
      const I = (x, y) => syn[Math.max(0, Math.min(SH - 1, y)) * SW + Math.max(0, Math.min(SW - 1, x))];
      let cv = null;
      const R = 3;
      let cx = 25, cy = 33, dragging = false;
      const c = L.canvas(fig.el, { aspect: narrow ? 1.05 : 0.5 });
      fig.add(c.el);
      const ro = L.readout(fig.el); fig.add(ro.el);
      let F;
      c.draw = (ctx) => {
        const t = L.theme();
        if (!cv) cv = toCanvas(SW, SH, grayRGBA(img));
        const stacked = c.h > c.w * 0.8; // phone layout (aspect chosen at creation)
        const imgBox = stacked ? [0, 0, c.w, c.w * 0.625] : [0, 0, c.w * 0.58, c.h];
        const pw = stacked ? c.w / 2 : c.w * 0.42, ph = stacked ? c.h - imgBox[3] - 4 : c.h / 2;
        const p1 = stacked ? [0, imgBox[3] + 4, pw, ph] : [c.w * 0.58 + 6, 0, pw - 6, ph - 3];
        const p2 = stacked ? [pw, imgBox[3] + 4, pw, ph] : [c.w * 0.58 + 6, ph + 3, pw - 6, ph - 3];
        F = fitBox(imgBox[0], imgBox[1], imgBox[2], imgBox[3], SW, SH);
        blit(ctx, cv, F, SW, SH);
        box(ctx, F, cx, cy, R, t.accent2, 2.5);
        // SSD surface
        const S = [];
        for (let v = -4; v <= 4; v++) for (let u = -4; u <= 4; u++) {
          let s = 0;
          for (let y = -R; y <= R; y++) for (let x = -R; x <= R; x++) s += (I(cx + x + u, cy + y + v) - I(cx + x, cy + y)) ** 2;
          S.push(s);
        }
        const side = Math.min(p1[2], p1[3] - 18), sx0 = p1[0] + (p1[2] - side) / 2, sy0 = p1[1] + 16, cell = side / 9;
        for (let j = 0; j < 9; j++) for (let i = 0; i < 9; i++) {
          ctx.fillStyle = L.viridisish(Math.min(1, Math.sqrt(S[j * 9 + i] / 6)));
          ctx.fillRect(sx0 + i * cell, sy0 + j * cell, cell + 0.5, cell + 0.5);
        }
        L.draw.dot(ctx, sx0 + 4.5 * cell, sy0 + 4.5 * cell, 2.5, "#fff");
        L.draw.text(ctx, "SSD(δ), δ ∈ [−4, 4]²", p1[0] + p1[2] / 2, p1[1] + 12, t.muted, { size: 11, align: "center" });
        // structure tensor + polar plot
        let a = 0, b = 0, cc = 0; const grads = [];
        for (let y = -R; y <= R; y++) for (let x = -R; x <= R; x++) {
          const gx = 0.5 * (I(cx + x + 1, cy + y) - I(cx + x - 1, cy + y)), gy = 0.5 * (I(cx + x, cy + y + 1) - I(cx + x, cy + y - 1));
          a += gx * gx; b += gx * gy; cc += gy * gy; grads.push([gx, gy]);
        }
        const lo = lmin(a, b, cc), hi = lmax(a, b, cc);
        const ox = p2[0] + p2[2] / 2, oy = p2[1] + 10 + (p2[3] - 10) / 2, R0 = Math.min(p2[2], p2[3] - 14) * 0.46;
        L.draw.line(ctx, ox - R0, oy, ox + R0, oy, t.line, 1);
        L.draw.line(ctx, ox, oy - R0, ox, oy + R0, t.line, 1);
        for (const [gx, gy] of grads) L.draw.dot(ctx, ox + (gx / 0.25) * R0, oy + (gy / 0.25) * R0, 1.8, t.faint);
        const pts = [], norm = Math.max(hi, 0.3);
        for (let k = 0; k <= 96; k++) {
          const th = (k / 96) * Math.PI * 2, nx = Math.cos(th), ny = Math.sin(th);
          const q = a * nx * nx + 2 * b * nx * ny + cc * ny * ny, rad = (R0 * q) / norm;
          pts.push([ox + rad * nx, oy + rad * ny]);
        }
        L.draw.path(ctx, pts, t.accent, 2);
        // weakest direction: minimise n^T M n (angle of the smaller axis)
        const thMin = 0.5 * Math.atan2(2 * b, a - cc) + Math.PI / 2;
        L.draw.arrow(ctx, ox, oy, ox + Math.cos(thMin) * R0 * 0.9, oy + Math.sin(thMin) * R0 * 0.9, t.bad, 2);
        L.draw.text(ctx, "gradients · nᵀMn", p2[0] + p2[2] / 2, p2[1] + 12, t.muted, { size: 11, align: "center" });
        const kind = hi < 0.02 ? "flat: nothing to track" : lo < 0.05 ? "edge: can only measure motion across it" : "corner / texture: trackable";
        ro.html = `window centre (${cx}, ${cy}), 7×7 pixels<br>M = [${a.toFixed(3)} ${b.toFixed(3)}; ${b.toFixed(3)} ${cc.toFixed(3)}]<br>` +
          `weakest stretch λ<sub>min</sub> = <b>${lo.toFixed(4)}</b>, strongest λ<sub>max</sub> = ${hi.toFixed(4)} → <b>${kind}</b>`;
      };
      const move = (e) => {
        if (!F) return;
        const p = c.pos(e);
        cx = Math.max(R + 1, Math.min(SW - R - 2, Math.round(F.U(p.x))));
        cy = Math.max(R + 1, Math.min(SH - R - 2, Math.round(F.V(p.y))));
        c.redraw();
      };
      c.el.addEventListener("pointerdown", (e) => {
        if (!F) return;
        const p = c.pos(e);
        if (p.x < F.ox || p.y < F.oy || p.x > F.ox + SW * F.s || p.y > F.oy + SH * F.s) return;
        dragging = true; c.el.setPointerCapture(e.pointerId); e.preventDefault(); move(e);
      });
      c.el.addEventListener("pointermove", (e) => { if (dragging) move(e); });
      c.el.addEventListener("pointerup", () => { dragging = false; });
      c.el.addEventListener("pointercancel", () => { dragging = false; });
    }

    // ================================================================ 2. structure tensor
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>The structure tensor</h3>
      <p>Shift the window by a small $\boldsymbol\delta$. By the tangent-line approximation (chapter 5), each pixel changes by about $\nabla I\cdot\boldsymbol\delta = I_x\delta_u + I_y\delta_v$, so</p>
      <div class="eq-card"><div class="eq-label">Window error for a small shift</div>
      $$\mathrm{SSD}(\boldsymbol\delta) = \sum_{\mathbf q\in W}\big(I(\mathbf q+\boldsymbol\delta) - I(\mathbf q)\big)^2 \approx \sum_{\mathbf q\in W}(\nabla I(\mathbf q)\cdot\boldsymbol\delta)^2 = \boldsymbol\delta^\top M\,\boldsymbol\delta$$
      <div class="parts">
        <span>$W$</span><span>the window's pixels (e.g. 15×15 around the point)</span>
        <span>$\nabla I = (I_x, I_y)$</span><span>image gradient by central differences (chapter 1)</span>
        <span>$\boldsymbol\delta^\top M\boldsymbol\delta$</span><span>a quadratic form (chapter 2): how fast the error grows in direction $\boldsymbol\delta$</span>
      </div></div>
      <div class="eq-card"><div class="eq-label">Structure tensor</div>
      $$M = \sum_{\mathbf q\in W}\begin{pmatrix}I_x^2 & I_xI_y\\ I_xI_y & I_y^2\end{pmatrix}$$
      <div class="parts">
        <span>$\sum I_x^2$</span><span>how much horizontal change the window contains</span>
        <span>$\sum I_y^2$</span><span>vertical change</span>
        <span>$\sum I_xI_y$</span><span>whether the two go together (a diagonal edge makes this large)</span>
      </div></div>
      <p>A window is trackable if the error grows in <b>every</b> direction, i.e. if even the weakest stretch of $M$ is large. From chapter 2, for $M = \begin{pmatrix}a&b\\b&c\end{pmatrix}$:</p>
      <div class="eq-card"><div class="eq-label">Shi–Tomasi score ("Good features to track", 1994)</div>
      $$\lambda_{\min} = \min_{\|\mathbf n\|=1}\mathbf n^\top M\mathbf n = \tfrac12\Big(a + c - \sqrt{(a-c)^2 + 4b^2}\Big)$$
      <div class="parts">
        <span>$\lambda_{\min}\approx 0$, $\lambda_{\max}\approx 0$</span><span>flat</span>
        <span>$\lambda_{\min}\approx 0$, $\lambda_{\max}$ large</span><span>edge (the weak direction runs along the edge)</span>
        <span>both large</span><span>corner or texture: a good feature</span>
      </div></div>
      <div class="note"><b>Worked example.</b> Three pixels with gradients $(1,0)$, $(0.5,0.5)$, $(0,1)$: $a = 1 + 0.25 + 0 = 1.25$, $b = 0 + 0.25 + 0 = 0.25$, $c = 1.25$. $\lambda_{\min} = \frac12(2.5 - \sqrt{0 + 0.25}) = 1.0$. If all three were $(1,0)$ (a vertical edge): $a = 3$, $b = c = 0$, $\lambda_{\min} = \frac12(3 - 3) = 0$.</div>
      <p>This repo (<code>corners.wgsl</code>) scores every pixel with a 5×5 window and divides $M$ by the pixel count (25), which scales $\lambda_{\min}$ by $\frac1{25}$; only the thresholds care.</p>

      <h3>Picking well-spread corners</h3>
      <p>The strongest pixels cluster on a few very textured spots. Good geometry (chapter 7) needs points spread over the whole image, so this repo:</p>
      <ol>
        <li>Splits the image into <b>16×16 tiles</b> and keeps only the best pixel of each tile (a GPU reduction). This is a cheap <b>non-maximum suppression</b>: no two candidates from one tile.</li>
        <li>Drops candidates below $\max(0.02\cdot\text{best score},\ 10^{-5})$ (<code>quality_level</code>, <code>min_response</code>) and anything within 12 px of the border.</li>
        <li>Sorts the rest strongest-first and accepts a candidate only if the 3×3 block of <b>grid cells</b> (cell size = <code>min_distance</code> = 12 px) around it contains no point yet, up to 800 points.</li>
      </ol>
      <div class="note"><b>Worked example.</b> Cell size 12. A candidate at $(50, 30)$ is in cell $(\lfloor 50/12\rfloor, \lfloor 30/12\rfloor) = (4, 2)$. An existing point at $(70, 20)$ is in cell $(5, 1)$: a neighbouring cell, so the candidate is rejected. A point at $(80, 20)$ (cell $(6,1)$) would not block it.</div>
    `);

    // shared frame loading for the real-image widgets
    const framesP = Promise.all([L.loadImage("img/frame_120.png"), L.loadImage("img/frame_123.png")]).then(([a, b]) => {
      const S = { A: a, B: b, cvA: toCanvas(a.w, a.h, a.rgba), cvB: toCanvas(b.w, b.h, b.rgba) };
      S.score = scoreMap(a);
      S.pA = pyramid(a, 3); S.pB = pyramid(b, 3);
      return S;
    });
    const loading = (ctx, c, err) => L.draw.text(ctx, err ? "Could not load the demo frames." : "Loading frames…", c.w / 2, c.h / 2, L.theme().muted, { size: 14, align: "center" });

    // ---- widget 2: Shi–Tomasi on a real frame
    let sharedPoint = null; // [u, v] chosen for the LK widgets
    const lkListeners = [];
    {
      const fig = L.figure(root, "<b>Corners in a real frame.</b> Frame 120 of the demo video (192×192). Move the sliders; tap anywhere to see that pixel's score. Filled: accepted corners. Hollow: tile winners blocked by a stronger neighbour. Tick the heat map to see the Shi–Tomasi score everywhere.");
      const c = L.canvas(fig.el, { aspect: 1, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ro = L.readout(fig.el); fig.add(ro.el);
      let S = null, err = false, heatCv = null, res = null, tap = null, F;
      const q = L.slider(ctl, { label: "quality level", min: 0.005, max: 0.3, step: 0.005, value: 0.02, fmt: (v) => v.toFixed(3), oninput: () => { res = null; c.redraw(); } });
      const md = L.slider(ctl, { label: "min distance (px)", min: 4, max: 32, step: 1, value: 12, oninput: () => { res = null; c.redraw(); } });
      const heat = L.toggle(ctl, "score heat map", false, () => c.redraw());
      framesP.then((s) => { S = s; c.redraw(); }, () => { err = true; c.redraw(); });
      c.draw = (ctx) => {
        const t = L.theme();
        if (!S) return loading(ctx, c, err);
        F = fitBox(0, 0, c.w, c.h, S.A.w, S.A.h);
        blit(ctx, S.cvA, F, S.A.w, S.A.h);
        if (heat.checked) {
          if (!heatCv) {
            let mx = 0; for (const v of S.score) mx = Math.max(mx, v);
            const o = new Uint8ClampedArray(S.A.w * S.A.h * 4);
            for (let i = 0; i < S.A.w * S.A.h; i++) {
              const m = L.viridisish(Math.sqrt(S.score[i] / mx)).match(/\d+/g).map(Number);
              o[4 * i] = m[0]; o[4 * i + 1] = m[1]; o[4 * i + 2] = m[2]; o[4 * i + 3] = 215;
            }
            heatCv = toCanvas(S.A.w, S.A.h, o);
          }
          blit(ctx, heatCv, F, S.A.w, S.A.h);
        }
        if (!res) res = detect(S.A, S.score, q.value, md.value);
        const rad = Math.max(3, F.s * 1.6);
        for (const k of res.blocked) { L.draw.dot(ctx, F.X(k.x), F.Y(k.y), rad, "rgba(0,0,0,0)", t.accent2); }
        for (const k of res.kept) L.draw.dot(ctx, F.X(k.x), F.Y(k.y), rad, t.accent3, "#fff");
        let h = `best score ${res.best.toExponential(2)}, threshold ${res.thr.toExponential(2)}; ${res.nTiles} tiles → ${res.nTiles - res.below.length} above threshold → <b>${res.kept.length} corners</b> (${res.blocked.length} blocked by min distance)`;
        if (tap) {
          const u = Math.round(tap[0]), v = Math.round(tap[1]);
          box(ctx, F, u, v, 2, "#fff", 2);
          const sc = u >= 0 && v >= 0 && u < S.A.w && v < S.A.h ? S.score[v * S.A.w + u] : 0;
          h += `<br>tapped pixel (${u}, ${v}): score λ<sub>min</sub>/25 = <b>${sc.toExponential(2)}</b>` + (sc === 0 ? " (inside the 12 px border: ignored)" : sc >= res.thr ? " (above threshold)" : " (below threshold)");
        }
        ro.html = h;
      };
      c.el.addEventListener("pointerdown", (e) => {
        if (!F || !S) return;
        const p = c.pos(e);
        tap = [Math.max(0, Math.min(S.A.w - 1, F.U(p.x))), Math.max(0, Math.min(S.A.h - 1, F.V(p.y)))];
        c.redraw();
      });
    }

    // ================================================================ 3. Lucas–Kanade
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Lucas–Kanade: finding where the window went</h3>
      <p>Assume a point's small neighbourhood looks the same in the next frame, just moved: <b>brightness constancy</b>.</p>
      <div class="eq-card"><div class="eq-label">Brightness constancy and the LK cost</div>
      $$I_{\text{next}}(\mathbf q + \mathbf d) \approx I_{\text{prev}}(\mathbf q)\ \ \text{for all } \mathbf q\in W, \qquad E(\mathbf d) = \sum_{\mathbf q\in W}\big(I_{\text{next}}(\mathbf q+\mathbf d) - I_{\text{prev}}(\mathbf q)\big)^2$$
      <div class="parts">
        <span>$\mathbf d$</span><span>the unknown motion of the point (2 numbers)</span>
        <span>$I_{\text{prev}}$ window</span><span>the <b>template</b>, cut out once around the point</span>
        <span>$I_{\text{next}}(\mathbf q + \mathbf d)$</span><span>the next frame sampled at shifted, sub-pixel positions (bilinear, chapter 1)</span>
      </div></div>
      <p>That is a least-squares problem with residuals $r_\mathbf q = I_{\text{next}}(\mathbf q+\mathbf d) - I_{\text{prev}}(\mathbf q)$, so we use Gauss–Newton (chapter 5). The Jacobian row of pixel $\mathbf q$ is the image gradient $\nabla I(\mathbf q)$, so $J^\top J$ is exactly the structure tensor:</p>
      <div class="eq-card"><div class="eq-label">One Lucas–Kanade step</div>
      $$M\,\boldsymbol\delta = \mathbf b, \qquad \mathbf b = \sum_{\mathbf q\in W}\nabla I(\mathbf q)\,\big(I_{\text{prev}}(\mathbf q) - I_{\text{next}}(\mathbf q + \mathbf d)\big), \qquad \mathbf d \leftarrow \mathbf d + \boldsymbol\delta$$
      <div class="parts">
        <span>$M$</span><span>structure tensor of the template window ($= J^\top J$)</span>
        <span>$\mathbf b$</span><span>$= -J^\top\mathbf r$: each pixel's gradient weighted by its brightness mismatch</span>
        <span>$\boldsymbol\delta = M^{-1}\mathbf b$</span><span>a 2×2 solve (chapter 2's inverse formula)</span>
      </div></div>
      <p>This repo takes the gradients from the <b>template</b> (previous frame), not from the moving next-frame window. At the answer the two windows coincide, so this changes little, and $M$ and $M^{-1}$ are computed <b>once per pyramid level</b> instead of every iteration. It also shows why corners matter: LK must invert $M$, which is impossible (or wildly unstable) when $\lambda_{\min}\approx 0$.</p>
      <div class="note"><b>Worked example.</b> $M = \begin{pmatrix}2 & 0.5\\0.5 & 1\end{pmatrix}$, $\mathbf b = (0.3, -0.1)$. $\det M = 2 - 0.25 = 1.75$. $M^{-1} = \frac{1}{1.75}\begin{pmatrix}1 & -0.5\\-0.5 & 2\end{pmatrix}$. $\boldsymbol\delta = \frac{1}{1.75}(0.3 + 0.05,\ -0.15 - 0.2) = (0.2, -0.2)$.</div>

      <h3>Pyramids: large motions</h3>
      <p>The linearisation only holds for motions of about a pixel or two (the basin, chapter 5). Real motions are larger, so LK runs <b>coarse-to-fine</b> on image pyramids of both frames:</p>
      <ul>
        <li>Level $l$ has $2^{-l}$ the resolution. A point at level-0 pixel $u$ sits at $u_l = (u + \tfrac12)/2^l - \tfrac12$ (pixel-centre convention, chapter 1).</li>
        <li>Start at the top level with $\mathbf d = (\text{guess} - \mathbf p_0)/2^{\text{top}}$ (the guess is the old position: no motion).</li>
        <li>Iterate LK on that level; then <b>double</b> $\mathbf d$ and move one level down. The finest level adds the last sub-pixel correction.</li>
      </ul>
      <p>This repo's tracker pyramid uses a slightly smoother $[1\,3\,3\,1]/8$ filter than chapter 1's 2×2 box (same pixel-centre convention), up to 4 levels, stopping before the top level gets narrower than 4 windows (60 px).</p>
      <pre><code>// Pyramidal LK for one point (klt.wgsl), window radius R = 7 (15×15), 20 iterations max
d = (guess − p0) / 2^top
for l = top … 0:
    pl = (p0 + ½) / 2^l − ½                      // point on this level
    for each q in window: T[q] = I_prev,l(pl+q); G[q] = ∇I_prev,l(pl+q)
    M = Σ G Gᵀ ;  if λmin(M)/n &lt; 1e-7: FAIL      // untrackable here
    repeat ≤ 20 times:
        b = Σ G[q] · (T[q] − I_next,l(pl + q + d))
        δ = M⁻¹ b ;  d = d + δ
        if |δ| &lt; 0.01 px: break
    if l &gt; 0: d = 2·d
pos = p0 + d
residual = mean |I_prev(p0+o) − I_next(pos+o)| over the level-0 window
FAIL if pos is outside the image</code></pre>
    `);

    // ---- widget 3: LK on a real patch
    {
      const fig = L.figure(root, "<b>Lucas–Kanade on real frames.</b> Tap a point in frame 120 (left). It is tracked into frame 123 (right, 3 frames later). Dots show every iteration's estimate, coloured by pyramid level (orange = level 2, blue = level 1, green = level 0). Try 1 level vs 3 levels, a corner vs an edge vs the flat desk.");
      const c = L.canvas(fig.el, { aspect: 0.52, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ro = L.readout(fig.el); fig.add(ro.el);
      let S = null, err = false, res = null, FA, FB;
      const lev = L.slider(ctl, { label: "pyramid levels", min: 1, max: 3, step: 1, value: 3, oninput: () => { res = null; c.redraw(); } });
      const rad = L.slider(ctl, { label: "window radius", min: 2, max: 10, step: 1, value: 7, oninput: () => { res = null; c.redraw(); } });
      framesP.then((s) => {
        S = s;
        if (!sharedPoint) {
          const d = detect(S.A, S.score);
          const k = d.kept.find((q) => q.x > 30 && q.x < 160 && q.y > 30 && q.y < 160) || d.kept[0];
          sharedPoint = k ? [k.x, k.y] : [96, 96];
        }
        c.redraw(); for (const f of lkListeners) f();
      }, () => { err = true; c.redraw(); });
      const levColor = (t, l) => [t.accent3, t.accent, t.accent2][l] || t.accent2;
      c.draw = (ctx) => {
        const t = L.theme();
        if (!S) return loading(ctx, c, err);
        const gap = 6, bw = (c.w - gap) / 2;
        FA = fitBox(0, 0, bw, c.h, S.A.w, S.A.h);
        FB = fitBox(bw + gap, 0, bw, c.h, S.B.w, S.B.h);
        blit(ctx, S.cvA, FA, S.A.w, S.A.h);
        blit(ctx, S.cvB, FB, S.B.w, S.B.h);
        L.draw.text(ctx, "frame 120", FA.ox + 4, FA.oy + 14, "#fff", { size: 12, bold: true });
        L.draw.text(ctx, "frame 123", FB.ox + 4, FB.oy + 14, "#fff", { size: 12, bold: true });
        const p0 = sharedPoint, R = rad.value;
        if (!res) res = trackFB(S.pA, S.pB, p0, lev.value, R);
        box(ctx, FA, p0[0], p0[1], R, t.accent2, 2);
        L.draw.dot(ctx, FA.X(p0[0]), FA.Y(p0[1]), 2.5, t.accent2);
        // ghost of the start position in the right image
        box(ctx, FB, p0[0], p0[1], R, "rgba(255,255,255,0.6)", 1);
        let prev = [FB.X(p0[0]), FB.Y(p0[1])];
        for (const q of res.path) {
          const x = FB.X(Math.max(-5, Math.min(S.B.w + 4, q.x))), y = FB.Y(Math.max(-5, Math.min(S.B.h + 4, q.y)));
          L.draw.line(ctx, prev[0], prev[1], x, y, "#fff", 1);
          L.draw.dot(ctx, x, y, 2.5, levColor(t, q.l));
          prev = [x, y];
        }
        if (res.ok) box(ctx, FB, res.pos[0], res.pos[1], R, res.fb <= 1 && res.residual <= 0.08 ? t.good : t.bad, 2);
        const lvNames = res.its.map((n, k) => `L${lev.value - 1 - k}: ${n}`).join(", ");
        let h = `p₀ = (${p0[0].toFixed(1)}, ${p0[1].toFixed(1)}); iterations per level ${lvNames}<br>`;
        if (!res.ok) h += `<b style="color:var(--bad)">track failed</b> (${res.its[res.its.length - 1] === 0 ? "λmin too small: untrackable window" : "left the image or backward track failed"})`;
        else {
          h += `d = (${(res.pos[0] - p0[0]).toFixed(2)}, ${(res.pos[1] - p0[1]).toFixed(2)}) px, residual = <b>${res.residual.toFixed(3)}</b> (max 0.08), forward–backward error = <b>${res.fb.toFixed(2)} px</b> (max 1)<br>`;
          h += res.fb <= 1 && res.residual <= 0.08 ? `<b style="color:var(--good)">kept</b>` : `<b style="color:var(--bad)">dropped</b>`;
        }
        ro.html = h;
      };
      c.el.addEventListener("pointerdown", (e) => {
        if (!FA || !S) return;
        const p = c.pos(e);
        if (p.x > FA.ox + S.A.w * FA.s) return;
        sharedPoint = [Math.max(0, Math.min(S.A.w - 1, FA.U(p.x))), Math.max(0, Math.min(S.A.h - 1, FA.V(p.y)))];
        res = null; c.redraw();
        for (const f of lkListeners) f();
      });
    }

    // ---- widget 4: basin vs shift
    {
      const fig = L.figure(root, "<b>How far can LK reach?</b> Frame 120 is shifted right by a known amount (x-axis) and the point you tapped above is tracked with 1, 2 or 3 pyramid levels. y-axis: final error in pixels (× = track failed). Each extra level roughly doubles the reachable motion.");
      const c = L.canvas(fig.el, { aspect: 0.55, scroll: true });
      fig.add(c.el);
      const ro = L.readout(fig.el); fig.add(ro.el);
      let S = null, err = false, shifted = null, curves = null, forPt = null;
      const MAXS = 30;
      const compute = () => {
        if (!S || !sharedPoint) return;
        if (!shifted) {
          shifted = [];
          for (let s = 0; s <= MAXS; s += 1) {
            const { w, h, data } = S.A, o = new Float32Array(w * h);
            for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) o[y * w + x] = data[y * w + Math.max(0, x - s)];
            shifted.push(pyramid({ w, h, data: o }, 3));
          }
        }
        forPt = sharedPoint.slice();
        curves = [1, 2, 3].map((nl) => shifted.map((P, s) => {
          const r = lk(S.pA, P, forPt, forPt, nl, 7);
          return r.ok ? Math.hypot(r.pos[0] - forPt[0] - s, r.pos[1] - forPt[1]) : null;
        }));
      };
      lkListeners.push(() => { curves = null; c.redraw(); });
      framesP.then((s) => { S = s; c.redraw(); }, () => { err = true; c.redraw(); });
      c.draw = (ctx) => {
        const t = L.theme();
        if (!S || !sharedPoint) return loading(ctx, c, err);
        if (!curves) compute();
        const P = L.plot(c, { x0: 0, x1: MAXS, y0: 0, y1: 6, pad: [14, 12, 28, 34] });
        P.axes(ctx, { xticks: 6, yticks: 3, fmt: (v) => v.toFixed(0) });
        L.draw.text(ctx, "true shift (px)", c.w - 12, c.h - 4, t.muted, { size: 12, align: "right" });
        L.draw.text(ctx, "error (px)", 40, 12, t.muted, { size: 12 });
        const cols = [t.accent2, t.accent, t.accent3];
        const reach = [];
        curves.forEach((cv, k) => {
          let seg = [];
          let r = -1;
          cv.forEach((e, s) => {
            if (e === null) { L.draw.path(ctx, seg, cols[k], 2.5); seg = []; L.draw.text(ctx, "×", P.X(s), P.Y(5.6) + 4 + 8 * k, cols[k], { size: 13, align: "center", bold: true }); return; }
            seg.push([P.X(s), P.Y(Math.min(6, e))]);
            if (e < 0.5 && r === s - 1) r = s;
          });
          L.draw.path(ctx, seg, cols[k], 2.5);
          reach.push(r);
        });
        ro.html = `point (${forPt[0].toFixed(1)}, ${forPt[1].toFixed(1)}). Largest shift tracked to &lt; 0.5 px from 0 upward: ` +
          reach.map((r, k) => `<b style="color:${cols[k]}">${k + 1} level${k ? "s" : ""}: ${r} px</b>`).join(", ");
      };
    }

    // ================================================================ 4. forward-backward + lifetime
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Is the track trustworthy? Forward–backward check</h3>
      <p>LK always returns <i>some</i> answer, even when the point was occluded, left the view, or slid along an edge. A cheap test: track the result <b>back</b> into the previous frame (starting from the new position, with the original position as the guess). A correct track comes home; a wrong one usually doesn't.</p>
      <div class="eq-card"><div class="eq-label">Forward–backward error</div>
      $$e_{fb} = \big\|\mathbf p_0 - \mathrm{LK}_{\text{next}\to\text{prev}}\big(\mathrm{LK}_{\text{prev}\to\text{next}}(\mathbf p_0)\big)\big\|$$
      <div class="parts">
        <span>$\mathbf p_0$</span><span>position in the previous frame</span>
        <span>inner LK</span><span>forward track: the new position $\mathbf p_1$</span>
        <span>outer LK</span><span>backward track from $\mathbf p_1$</span>
        <span>$e_{fb}$</span><span>round-trip distance; this repo drops the track if $e_{fb} > 1$ px</span>
      </div></div>
      <div class="note"><b>Worked example.</b> $\mathbf p_0 = (40, 60)$, forward gives $(45.2, 58.9)$, backward returns $(40.6, 60.8)$. $e_{fb} = \sqrt{0.6^2 + 0.8^2} = 1.0$ px: right at the limit.</div>
      <p>A track is also dropped if the <b>residual</b> (mean absolute brightness difference between the two 15×15 windows) exceeds 0.08, if LK failed (untrackable $M$ or left the image), or if it lands in the same 12-px grid cell as an older track (two tracks collapsed onto one spot; the younger one goes).</p>
    `);

    // ---- widget 5: track all corners
    {
      const fig = L.figure(root, "<b>The whole tracker on one frame pair.</b> Corners detected in frame 120 are tracked into frame 123 with the forward–backward check. Green: kept. Red: dropped (hollow ring = where the backward track ended). Tick <b>occluder</b> and tap to drop a grey square into frame 123, like a hand in front of the camera. Flip the background to see the motion.");
      const c = L.canvas(fig.el, { aspect: 1, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const ctl2 = L.controls(fig.el); fig.add(ctl2);
      const ro = L.readout(fig.el); fig.add(ro.el);
      let S = null, err = false, tracks = null, showA = false, F, occ = [70, 100], occB = null;
      const OCC = 22; // half size of the occluder square (px)
      const lev = L.slider(ctl, { label: "pyramid levels", min: 1, max: 3, step: 1, value: 2, oninput: () => { tracks = null; c.redraw(); } });
      const md = L.slider(ctl, { label: "min distance (px)", min: 4, max: 24, step: 1, value: 12, oninput: () => { tracks = null; c.redraw(); } });
      const fbMax = L.slider(ctl, { label: "max f–b error (px)", min: 0.1, max: 3, step: 0.1, value: 1, oninput: () => c.redraw() });
      const resMax = L.slider(ctl, { label: "max residual", min: 0.01, max: 0.2, step: 0.01, value: 0.08, oninput: () => c.redraw() });
      const occT = L.toggle(ctl2, "occluder", false, () => { occB = null; tracks = null; c.redraw(); });
      const exag = L.toggle(ctl2, "arrows ×4", true, () => c.redraw());
      L.button(ctl2, "Flip background 120 ↔ 123", () => { showA = !showA; c.redraw(); });
      framesP.then((s) => { S = s; c.redraw(); }, () => { err = true; c.redraw(); });
      const target = () => {
        if (!occT.checked) return { pyr: S.pB, cv: S.cvB };
        if (!occB) {
          const { w, h } = S.B, data = new Float32Array(S.B.data), rgba = new Uint8ClampedArray(S.B.rgba);
          for (let y = Math.max(0, Math.round(occ[1]) - OCC); y <= Math.min(h - 1, Math.round(occ[1]) + OCC); y++) {
            for (let x = Math.max(0, Math.round(occ[0]) - OCC); x <= Math.min(w - 1, Math.round(occ[0]) + OCC); x++) {
              data[y * w + x] = 0.5;
              rgba[4 * (y * w + x)] = rgba[4 * (y * w + x) + 1] = rgba[4 * (y * w + x) + 2] = 128;
            }
          }
          occB = { pyr: pyramid({ w, h, data }, 3), cv: toCanvas(w, h, rgba) };
        }
        return occB;
      };
      c.draw = (ctx) => {
        const t = L.theme();
        if (!S) return loading(ctx, c, err);
        const tg = target();
        if (!tracks) {
          const d = detect(S.A, S.score, 0.02, md.value);
          tracks = d.kept.map((k) => ({ p0: [k.x, k.y], r: trackFB(S.pA, tg.pyr, [k.x, k.y], lev.value, 7) }));
        }
        F = fitBox(0, 0, c.w, c.h, S.A.w, S.A.h);
        blit(ctx, showA ? S.cvA : tg.cv, F, S.A.w, S.A.h);
        L.draw.text(ctx, showA ? "frame 120" : "frame 123", F.ox + 6, F.oy + 16, "#fff", { size: 13, bold: true });
        const m = exag.checked ? 4 : 1;
        let kept = 0, nFail = 0, nFb = 0, nRes = 0;
        const disp = [];
        for (const { p0, r } of tracks) {
          const bad = !r.ok ? "fail" : r.fb > fbMax.value ? "fb" : r.residual > resMax.value ? "res" : "";
          if (bad === "fail") nFail++; else if (bad === "fb") nFb++; else if (bad === "res") nRes++; else { kept++; disp.push(Math.hypot(r.pos[0] - p0[0], r.pos[1] - p0[1])); }
          const col = bad ? t.bad : t.good;
          const ex = p0[0] + (r.pos[0] - p0[0]) * m, ey = p0[1] + (r.pos[1] - p0[1]) * m;
          if (r.ok && Math.hypot(r.pos[0] - p0[0], r.pos[1] - p0[1]) * m < 80) L.draw.arrow(ctx, F.X(p0[0]), F.Y(p0[1]), F.X(ex), F.Y(ey), col, 2, 7);
          L.draw.dot(ctx, F.X(p0[0]), F.Y(p0[1]), 2.5, col);
          if (bad && r.back && Math.hypot(r.back[0] - p0[0], r.back[1] - p0[1]) < 60) L.draw.dot(ctx, F.X(r.back[0]), F.Y(r.back[1]), 4, "rgba(0,0,0,0)", t.bad);
        }
        disp.sort((a, b) => a - b);
        ro.html = `${tracks.length} corners → <b style="color:var(--good)">${kept} kept</b>; dropped: ${nFb} forward–backward, ${nRes} residual, ${nFail} LK failed` +
          (disp.length ? `<br>median motion of kept tracks: ${disp[Math.floor(disp.length / 2)].toFixed(2)} px` : "");
      };
      c.el.addEventListener("pointerdown", (e) => {
        if (!F || !S || !occT.checked) return;
        const p = c.pos(e);
        occ = [Math.max(0, Math.min(S.B.w - 1, F.U(p.x))), Math.max(0, Math.min(S.B.h - 1, F.V(p.y)))];
        occB = null; tracks = null; c.redraw();
      });
    }

    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Track lifetime</h3>
      <p>Tracks live across many frames. Each carries a stable id and an <b>age</b> (frames survived); chapter 7 uses long-lived tracks to find frame pairs with enough parallax. Per frame, this repo (<code>tracker/mod.rs</code>) does:</p>
      <pre><code>// one frame of the KLT tracker (all LK work runs on the GPU, one thread per point)
build the new frame's pyramid
for each live track, oldest first:
    fwd = LK(prev → cur, from p, guess p)          // no motion prediction
    bwd = LK(cur → prev, from fwd.pos, guess p)
    drop if LK failed, |bwd.pos − p| > 1 px, or residual > 0.08
    drop if its 12-px grid cell is already taken  // an older track got there first
    otherwise: p = fwd.pos, age += 1
detect candidates: best Shi–Tomasi pixel per 16×16 tile, score ≥ max(0.02·best, 1e-5)
strongest first: spawn a new track (age 0) if the 3×3 cells around it are empty,
                 until 800 tracks</code></pre>
      <div class="key"><b>Summary.</b> Track windows with a strong structure tensor (large $\lambda_{\min}$). LK is Gauss–Newton on a 2-number motion: $M\boldsymbol\delta = \mathbf b$. Pyramids extend its reach; the forward–backward check and residual test throw away tracks that went wrong. New corners refill empty parts of the image.</div>
    `);

    // ================================================================ quiz
    const f3 = (v) => L.fmt(v, 3), f4 = (v) => L.fmt(v, 4);
    L.quiz(root, "klt", [
      { id: "aperture", type: "mc",
        q: "A tracking window sits on a long straight edge, with nothing else in it. Which motion can be measured?",
        choices: ["Only the component across the edge (perpendicular to it)", "Only the component along the edge", "Both components, but less accurately than at a corner", "Neither: an edge carries no information"],
        answer: 0,
        explain: "Sliding along the edge leaves the window unchanged (the aperture problem); moving across it changes every pixel on the edge. $M$ has one large and one near-zero stretch." },
      { id: "tensor", type: "num",
        gen: (r) => {
          const G = Array.from({ length: 3 }, () => [r.pick([-1, -0.5, 0, 0.5, 1, 2]), r.pick([-1, -0.5, 0, 0.5, 1, 2])]);
          const a = G.reduce((s, g) => s + g[0] * g[0], 0), b = G.reduce((s, g) => s + g[0] * g[1], 0), c = G.reduce((s, g) => s + g[1] * g[1], 0);
          return { q: String.raw`A window has three pixels with gradients $(I_x, I_y) = ${G.map((g) => `(${g[0]}, ${g[1]})`).join(",\\ ")}$. Give the structure tensor entries.`,
            answer: [a, b, c], labels: ["$\\sum I_x^2$", "$\\sum I_xI_y$", "$\\sum I_y^2$"], tol: 1e-6,
            explain: String.raw`$\sum I_x^2 = ${G.map((g) => `(${g[0]})^2`).join("+")} = ${a}$; $\sum I_xI_y = ${G.map((g) => `(${g[0]})(${g[1]})`).join("+")} = ${b}$; $\sum I_y^2 = ${c}$.` };
        } },
      { id: "shi", type: "num",
        gen: (r) => {
          let a, b, c;
          do { a = r.int(1, 9); c = r.int(1, 9); b = r.int(-3, 3); } while (a * c - b * b < 0);
          const v = lmin(a, b, c);
          return { q: String.raw`$M = \begin{pmatrix}${a} & ${b}\\ ${b} & ${c}\end{pmatrix}$. What is its Shi–Tomasi score $\lambda_{\min}$? (to 3 decimals)`,
            answer: v, tol: 2e-3,
            explain: String.raw`$\lambda_{\min} = \frac12\big(${a} + ${c} - \sqrt{(${a} - ${c})^2 + 4\cdot (${b})^2}\big) = \frac12\big(${a + c} - \sqrt{${(a - c) ** 2 + 4 * b * b}}\big) = ${f3(v)}$.` };
        } },
      { id: "classify", type: "mc",
        q: "Which window is the best feature to track?",
        choices: ["$M = \\begin{pmatrix}3 & 0\\\\ 0 & 3\\end{pmatrix}$", "$M = \\begin{pmatrix}9 & 0\\\\ 0 & 0.1\\end{pmatrix}$", "$M = \\begin{pmatrix}5 & 4.9\\\\ 4.9 & 5\\end{pmatrix}$", "$M = \\begin{pmatrix}0.02 & 0\\\\ 0 & 0.02\\end{pmatrix}$"],
        answer: 0,
        explain: "Compare $\\lambda_{\\min}$: 3; 0.1 (vertical edge); $\\frac12(10 - \\sqrt{0 + 96.04}) = 0.1$ (a diagonal edge: large entries but one weak direction); 0.02 (flat). Big numbers on the diagonal alone don't make a corner." },
      { id: "quadform", type: "num",
        gen: (r) => {
          const a = r.int(1, 6), c = r.int(1, 6), b = r.int(-2, 2), dx = r.pick([-1, 0.5, 1, 2, -0.5]), dy = r.pick([-1, 0.5, 1, -2, 0]);
          const v = a * dx * dx + 2 * b * dx * dy + c * dy * dy;
          return { q: String.raw`Structure tensor $M = \begin{pmatrix}${a} & ${b}\\ ${b} & ${c}\end{pmatrix}$. Approximately how much does the window's SSD grow for a shift $\boldsymbol\delta = (${dx}, ${dy})$?`,
            answer: v, tol: 1e-6,
            explain: String.raw`$\boldsymbol\delta^\top M\boldsymbol\delta = a\delta_u^2 + 2b\,\delta_u\delta_v + c\,\delta_v^2 = ${a}\cdot${dx * dx} + 2\cdot${b}\cdot${dx * dy} + ${c}\cdot${dy * dy} = ${f4(v)}$.` };
        } },
      { id: "thresh", type: "num",
        gen: (r) => {
          const best = r.pick([2e-4, 4e-4, 1e-3, 2.5e-3, 5e-3, 8e-3]), ql = r.pick([0.02, 0.05, 0.01]);
          const thr = Math.max(best * ql, 1e-5);
          return { q: String.raw`The best corner score in a frame is $${best}$; quality level $${ql}$; min response $10^{-5}$. What score must a candidate reach? (e.g. 2e-5)`,
            answer: thr, rtol: 0.01,
            explain: String.raw`$\max(${ql}\cdot${best},\ 10^{-5}) = \max(${(best * ql).toExponential(2)},\ 1\text{e-}5) = ${thr.toExponential(2)}$.` };
        } },
      { id: "tiles", type: "num",
        gen: (r) => {
          const [w, h] = r.pick([[640, 480], [320, 240], [192, 192], [1280, 720], [480, 360], [200, 150]]);
          const n = Math.ceil(w / 16) * Math.ceil(h / 16);
          return { q: String.raw`A ${w}×${h} frame is split into 16×16 tiles (partial tiles at the edges count), each giving at most one candidate. What is the maximum number of candidates?`,
            answer: n, tol: 0.5,
            explain: String.raw`$\lceil ${w}/16\rceil\cdot\lceil ${h}/16\rceil = ${Math.ceil(w / 16)}\cdot${Math.ceil(h / 16)} = ${n}$.` };
        } },
      { id: "grid", type: "num",
        gen: (r) => {
          const cx = r.int(20, 150), cy = r.int(20, 150);
          const ex = cx + r.int(-30, 30), ey = cy + r.int(-30, 30);
          const c1 = [Math.floor(cx / 12), Math.floor(cy / 12)], c2 = [Math.floor(ex / 12), Math.floor(ey / 12)];
          const ok = Math.abs(c1[0] - c2[0]) > 1 || Math.abs(c1[1] - c2[1]) > 1 ? 1 : 0;
          return { q: String.raw`Min-distance grid with cell size 12 px. A new candidate is at $(${cx}, ${cy})$; the only existing track is at $(${ex}, ${ey})$. Give the candidate's cell indices, and 1 if it may be spawned or 0 if not.`,
            answer: [c1[0], c1[1], ok], labels: ["cell x", "cell y", "allowed (1/0)"], tol: 1e-6,
            explain: String.raw`Candidate cell $(\lfloor ${cx}/12\rfloor, \lfloor ${cy}/12\rfloor) = (${c1[0]}, ${c1[1]})$; track cell $(${c2[0]}, ${c2[1]})$. ${ok ? "Not within the 3×3 neighbourhood: allowed." : "Within the 3×3 neighbourhood (each index differs by at most 1): rejected."}` };
        } },
      { id: "lin", type: "num",
        gen: (r) => {
          const b = r.float(0.2, 0.8, 2), gx = r.float(-0.1, 0.1, 2), gy = r.float(-0.1, 0.1, 2), dx = r.pick([0.5, -0.5, 1, -1, 2]), dy = r.pick([0.5, -0.5, 1, -1, 0]);
          const v = b + gx * dx + gy * dy;
          return { q: String.raw`At the current estimate, $I_{\text{next}}(\mathbf q + \mathbf d) = ${b}$ and the gradient is $\nabla I = (${gx}, ${gy})$. Linearising, what is $I_{\text{next}}(\mathbf q + \mathbf d + \boldsymbol\delta)$ for $\boldsymbol\delta = (${dx}, ${dy})$? (to 4 decimals)`,
            answer: v, tol: 1e-4,
            explain: String.raw`$${b} + ${gx}\cdot(${dx}) + ${gy}\cdot(${dy}) = ${f4(v)}$.` };
        } },
      { id: "bvec", type: "num",
        gen: (r) => {
          const px = Array.from({ length: 3 }, () => ({ T: r.float(0.2, 0.8, 1), I: r.float(0.2, 0.8, 1), g: [r.pick([-0.2, -0.1, 0, 0.1, 0.2]), r.pick([-0.2, -0.1, 0, 0.1, 0.2])] }));
          const b = [0, 1].map((k) => px.reduce((s, p) => s + p.g[k] * (p.T - p.I), 0));
          return { q: String.raw`Three window pixels: template values $I_{\text{prev}} = ${px.map((p) => p.T).join(", ")}$; next-frame values at the current estimate $I_{\text{next}}(\mathbf q+\mathbf d) = ${px.map((p) => p.I).join(", ")}$; gradients $${px.map((p) => `(${p.g[0]}, ${p.g[1]})`).join(",\\ ")}$. Compute $\mathbf b = \sum\nabla I\,(I_{\text{prev}} - I_{\text{next}})$. (to 4 decimals)`,
            answer: b, labels: ["$b_u$", "$b_v$"], tol: 1e-4,
            explain: String.raw`Mismatches $I_{\text{prev}} - I_{\text{next}} = ${px.map((p) => f3(p.T - p.I)).join(", ")}$. $b_u = ${px.map((p) => `${p.g[0]}\cdot(${f3(p.T - p.I)})`).join(" + ")} = ${f4(b[0])}$, $b_v = ${px.map((p) => `${p.g[1]}\cdot(${f3(p.T - p.I)})`).join(" + ")} = ${f4(b[1])}$.` };
        } },
      { id: "lkstep", type: "num",
        gen: (r) => {
          let a, b, c, det;
          do { a = r.int(1, 6); c = r.int(1, 6); b = r.pick([-1, -0.5, 0, 0.5, 1, 2]); det = a * c - b * b; } while (det < 1);
          const bu = r.float(-1, 1, 1), bv = r.float(-1, 1, 1);
          const d = [(c * bu - b * bv) / det, (-b * bu + a * bv) / det];
          return { q: String.raw`One Lucas–Kanade iteration: $M = \begin{pmatrix}${a} & ${b}\\ ${b} & ${c}\end{pmatrix}$, $\mathbf b = (${bu}, ${bv})$. Solve $M\boldsymbol\delta = \mathbf b$. (to 3 decimals)`,
            answer: d, labels: ["$\\delta_u$", "$\\delta_v$"], tol: 2e-3,
            explain: String.raw`$\det M = ${a}\cdot${c} - (${b})^2 = ${det}$. $\boldsymbol\delta = \frac{1}{${det}}\begin{pmatrix}${c} & ${-b}\\ ${-b} & ${a}\end{pmatrix}\begin{pmatrix}${bu}\\ ${bv}\end{pmatrix} = (${f3(d[0])}, ${f3(d[1])})$.` };
        } },
      { id: "level", type: "num",
        gen: (r) => {
          const u = r.int(10, 400), l = r.int(1, 3);
          const v = (u + 0.5) / 2 ** l - 0.5;
          return { q: String.raw`A point is at level-0 column $u = ${u}$. What is its column at pyramid level $${l}$ (pixel-centre convention)?`,
            answer: v, tol: 1e-4,
            explain: String.raw`$u_l = (u + \tfrac12)/2^l - \tfrac12 = ${u + 0.5}/${2 ** l} - 0.5 = ${f4(v)}$.` };
        } },
      { id: "c2f", type: "num",
        gen: (r) => {
          const d3 = r.float(-3, 3, 1), c2 = r.float(-0.8, 0.8, 1), c1 = r.float(-0.8, 0.8, 1), c0 = r.float(-0.5, 0.5, 1);
          const d0 = ((d3 * 2 + c2) * 2 + c1) * 2 + c0;
          return { q: String.raw`Pyramidal LK with 4 levels (0–3). On level 3 it finds $d_u = ${d3}$ (level-3 pixels). The later levels add corrections $${c2}$ (level 2), $${c1}$ (level 1) and $${c0}$ (level 0), each in its own level's pixels. What is the final $d_u$ in level-0 pixels?`,
            answer: d0, tol: 1e-4,
            explain: String.raw`Double before each finer level: level 2: $2\cdot${d3} + (${c2}) = ${f4(d3 * 2 + c2)}$; level 1: $2\cdot${f4(d3 * 2 + c2)} + (${c1}) = ${f4((d3 * 2 + c2) * 2 + c1)}$; level 0: $2\cdot${f4((d3 * 2 + c2) * 2 + c1)} + (${c0}) = ${f4(d0)}$.` };
        } },
      { id: "fb", type: "num",
        gen: (r) => {
          let p, back, e;
          do {
            p = [r.int(20, 150), r.int(20, 150)];
            back = [+(p[0] + r.float(-1.5, 1.5, 1)).toFixed(1), +(p[1] + r.float(-1.5, 1.5, 1)).toFixed(1)];
            e = Math.hypot(back[0] - p[0], back[1] - p[1]);
          } while (Math.abs(e - 1) < 0.01);
          return { q: String.raw`A track starts at $\mathbf p_0 = (${p[0]}, ${p[1]})$. The backward track ends at $(${back[0]}, ${back[1]})$. Give the forward–backward error (px, to 3 decimals) and 1 if this repo keeps the track (by this test) or 0 if it drops it.`,
            answer: [e, e > 1 ? 0 : 1], labels: ["$e_{fb}$", "kept (1/0)"], tol: 2e-3,
            explain: String.raw`$e_{fb} = \sqrt{(${f3(back[0] - p[0])})^2 + (${f3(back[1] - p[1])})^2} = ${f3(e)}$ px, ${e > 1 ? "above" : "not above"} the 1 px limit → ${e > 1 ? "dropped" : "kept"}.` };
        } },
      { id: "resid", type: "num",
        gen: (r) => {
          let A, B, m;
          do {
            A = Array.from({ length: 4 }, () => r.float(0.2, 0.8, 2));
            const spread = r.pick([0.05, 0.1, 0.2]);
            B = A.map((a) => +(a + r.float(-spread, spread, 2)).toFixed(2));
            m = A.reduce((s, a, i) => s + Math.abs(a - B[i]), 0) / 4;
          } while (Math.abs(m - 0.08) < 0.002);
          return { q: String.raw`A (tiny, 4-pixel) window: previous-frame values $${A.join(", ")}$; tracked next-frame values $${B.join(", ")}$. Give the residual (mean absolute difference, to 4 decimals), and 1 if it passes this repo's limit of 0.08 or 0 if not.`,
            answer: [m, m > 0.08 ? 0 : 1], labels: ["residual", "passes (1/0)"], tol: 1e-4,
            explain: String.raw`Differences $${A.map((a, i) => f3(Math.abs(a - B[i]))).join(", ")}$, mean $${f4(m)}$ → ${m > 0.08 ? "above 0.08: dropped" : "passes"}.` };
        } },
      { id: "drop", type: "multi",
        q: "In this repo, which of these make the tracker drop a track? (select all)",
        choices: ["Forward–backward error above 1 px", "Mean absolute window difference above 0.08", "The template window's $\\lambda_{\\min}$ is too small on some pyramid level", "The tracked position leaves the image", "Another, older track landed in the same 12-px grid cell", "The track is older than 30 frames", "The point moved more than 5 px"],
        answer: [0, 1, 2, 3, 4],
        explain: "Age and motion size are not limits: long tracks are the valuable ones, and pyramids handle large motions." },
      { id: "tmplgrad", type: "mc",
        q: "Why does this repo's LK use the gradients of the template (previous frame) window rather than of the shifted next-frame window?",
        choices: ["$M$ and its inverse then stay fixed on a level, computed once instead of every iteration; at convergence the windows coincide anyway", "Template gradients are more accurate because the previous frame is sharper", "The next frame's gradients cannot be computed at sub-pixel positions", "It makes LK converge from any starting point"],
        answer: 0,
        explain: "It's a speed trick with almost no accuracy cost; sub-pixel gradients are perfectly possible with bilinear sampling, and the basin is still limited." },
      { id: "fbwhy", type: "mc",
        q: "Why does the forward–backward check catch a track that jumped onto the wrong spot (e.g. a similar-looking patch, or an occluded point)?",
        choices: ["Tracking back from the wrong spot rarely lands on the original point, so the round trip doesn't close", "The backward track uses a bigger window", "The residual of a wrong track is always exactly zero", "Wrong tracks always leave the image"],
        answer: 0,
        explain: "A correct match is consistent in both directions; an accidental one usually isn't. The residual test complements it (a wrong match often has a large brightness mismatch)." },
    ]);
  },
});
