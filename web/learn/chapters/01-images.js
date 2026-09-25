// Chapter 1: images as arrays of numbers — luma, coordinates, bilinear
// sampling, gradients, smoothing, pyramids and coarse-to-fine.
DTAM.chapter({
  id: "images",
  order: 1,
  title: "Images as numbers",
  subtitle: "Pixels, brightness, sub-pixel sampling, gradients and image pyramids",
  minutes: 40,
  render(root, L) {
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const fx = (v, d = 3) => L.fmt(v, d);

    // ---------------------------------------------------------------- image helpers
    /** Gray image {w, h, data} (+ optional rgba) from the procedural generator. */
    let frame = L.makeImage(192, 192, "scene", 5);
    const frameListeners = [];
    const px = (img, u, v) => img.data[clamp(v, 0, img.h - 1) * img.w + clamp(u, 0, img.w - 1)];
    /** Bilinear sample, pixel centres at integers, clamped at the border (as in klt.wgsl). */
    const sample = (img, u, v) => {
      u = clamp(u, 0, img.w - 1); v = clamp(v, 0, img.h - 1);
      const u0 = Math.floor(u), v0 = Math.floor(v);
      const u1 = Math.min(u0 + 1, img.w - 1), v1 = Math.min(v0 + 1, img.h - 1);
      const a = u - u0, b = v - v0;
      const top = (1 - a) * img.data[v0 * img.w + u0] + a * img.data[v0 * img.w + u1];
      const bot = (1 - a) * img.data[v1 * img.w + u0] + a * img.data[v1 * img.w + u1];
      return (1 - b) * top + b * bot;
    };
    /** 2x2 box downsample (frame.rs / pyr.wgsl): size ceil(w/2) x ceil(h/2), edge clamp. */
    const down2 = (img) => {
      const w = Math.ceil(img.w / 2), h = Math.ceil(img.h / 2), data = new Float32Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let s = 0;
        for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) s += px(img, 2 * x + dx, 2 * y + dy);
        data[y * w + x] = s / 4;
      }
      return { w, h, data };
    };
    /** Offscreen canvas for an image; col(i) -> [r, g, b] 0..255. */
    const bitmap = (img, col) => {
      const cv = document.createElement("canvas");
      cv.width = img.w; cv.height = img.h;
      const cx = cv.getContext("2d");
      const id = cx.createImageData(img.w, img.h);
      for (let i = 0; i < img.w * img.h; i++) {
        const [r, g, b] = col ? col(i) : [0, 0, 0].fill(Math.round(clamp(img.data[i], 0, 1) * 255));
        id.data[4 * i] = r; id.data[4 * i + 1] = g; id.data[4 * i + 2] = b; id.data[4 * i + 3] = 255;
      }
      cx.putImageData(id, 0, 0);
      return cv;
    };
    const blit = (ctx, cv, x, y, w, h) => {
      ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(cv, x, y, w, h); ctx.restore();
    };
    /** A text label on a theme-coloured pill (readable over any pixel colour). */
    const pill = (ctx, s, x, y, t, size = 11, color) => {
      ctx.save();
      ctx.font = `${size}px ${t.mono}`;
      const w = ctx.measureText(s).width + 6;
      ctx.globalAlpha = 0.85; ctx.fillStyle = t.panel;
      ctx.beginPath(); ctx.roundRect(x - w / 2, y - size * 0.72, w, size * 1.35, 4); ctx.fill();
      ctx.restore();
      L.draw.text(ctx, s, x, y + size * 0.36, color || t.fg, { size, align: "center", mono: true });
    };

    // Swap in the real demo frame (Rec.709 luma, like frame.rs) when it loads.
    (async () => {
      try {
        const im = await L.loadImage("img/frame_120.png");
        const w = Number(im.w) || 0, h = Number(im.h) || 0;
        if (!w || !h || !im.rgba) return;
        const data = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) data[i] = (0.2126 * im.rgba[4 * i] + 0.7152 * im.rgba[4 * i + 1] + 0.0722 * im.rgba[4 * i + 2]) / 255;
        frame = { w, h, data, rgba: im.rgba };
        for (const f of frameListeners) f();
      } catch (e) { /* keep the procedural image (e.g. opened from file://) */ }
    })();

    // ================================================================ 1. pixels
    root.insertAdjacentHTML("beforeend", String.raw`
      <style>
        #images table.mat th { font-weight: 400; color: var(--muted); padding: 3px 7px; }
        #images .swatches { display: flex; gap: 10px; flex-wrap: wrap; }
      </style>
      <p>To a computer an image is a grid of numbers. Everything DTAM does (tracking, depth, the cost volume) is arithmetic on these numbers, so we start here.</p>
      <h3>Pixels and coordinates</h3>
      <ul>
        <li>A <b>pixel</b> $\mathbf u = (u, v)$: $u$ = column (x, rightwards), $v$ = row (y, <b>downwards</b>), origin at the top-left.</li>
        <li>Its <b>intensity</b> $I(\mathbf u)$: stored as a byte 0–255, used as a number in $[0, 1]$ (divide by 255).</li>
        <li>Memory is one long list, row after row (<i>row-major</i>): pixel $(u, v)$ of a $w$-wide image sits at index $i = v\,w + u$.</li>
      </ul>
    `);
    {
      const fig = L.figure(root, "<b>A real frame, as numbers.</b> Drag on the left image to move the magnifier. Tap a cell on the right to read that pixel's coordinates, index and value.");
      const c = L.canvas(fig.el, { aspect: 0.5 });
      fig.add(c.el);
      const out = L.readout(fig.el); fig.add(out.el);
      let bmp = bitmap(frame);
      let win = { u: 60, v: 60 }, sel = { u: 64, v: 64 };
      let dragging = false;
      const layout = () => {
        const n = c.w < 520 ? 6 : 9;
        const S = Math.min(c.h - 2, (c.w - 12) / 2);
        return { n, S, zx: c.w - S, sc: S / frame.w };
      };
      const info = () => {
        const { w } = frame;
        const i = sel.v * w + sel.u;
        const val = px(frame, sel.u, sel.v);
        let rgb = "";
        if (frame.rgba) {
          const R = frame.rgba[4 * i], G = frame.rgba[4 * i + 1], B = frame.rgba[4 * i + 2];
          rgb = ` · RGB (${R}, ${G}, ${B}) → 0.2126·${R} + 0.7152·${G} + 0.0722·${B} = ${(0.2126 * R + 0.7152 * G + 0.0722 * B).toFixed(1)}`;
        }
        out.html = `pixel <b>(u, v) = (${sel.u}, ${sel.v})</b> · index v·w + u = ${sel.v}·${w} + ${sel.u} = <b>${i}</b>${rgb} · I = <b>${Math.round(val * 255)}</b>/255 = <b>${val.toFixed(3)}</b>`;
      };
      const moveWin = (p) => {
        const { n, sc } = layout();
        win.u = clamp(Math.round(p.x / sc - n / 2), 0, frame.w - n);
        win.v = clamp(Math.round(p.y / sc - n / 2), 0, frame.h - n);
        sel.u = clamp(sel.u, win.u, win.u + n - 1); sel.v = clamp(sel.v, win.v, win.v + n - 1);
        info(); c.redraw();
      };
      c.el.addEventListener("pointerdown", (e) => {
        const p = c.pos(e), { n, S, zx } = layout();
        if (p.x < S) { dragging = true; c.el.setPointerCapture(e.pointerId); moveWin(p); e.preventDefault(); }
        else if (p.x >= zx && p.y < S) {
          sel.u = win.u + clamp(Math.floor(((p.x - zx) / S) * n), 0, n - 1);
          sel.v = win.v + clamp(Math.floor((p.y / S) * n), 0, n - 1);
          info(); c.redraw();
        }
      });
      c.el.addEventListener("pointermove", (e) => { if (dragging) moveWin(c.pos(e)); });
      c.el.addEventListener("pointerup", () => { dragging = false; });
      c.el.addEventListener("pointercancel", () => { dragging = false; });
      c.draw = (ctx) => {
        const t = L.theme(), { n, S, zx, sc } = layout();
        blit(ctx, bmp, 0, 0, S, S);
        ctx.save(); ctx.strokeStyle = t.accent; ctx.lineWidth = 2;
        ctx.strokeRect(win.u * sc, win.v * sc, n * sc, n * sc); ctx.restore();
        const cs = S / n;
        for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
          const u = win.u + i, v = win.v + j, val = px(frame, u, v);
          ctx.fillStyle = L.gray(val);
          ctx.fillRect(zx + i * cs, j * cs, cs + 0.5, cs + 0.5);
          pill(ctx, String(Math.round(val * 255)), zx + (i + 0.5) * cs, (j + 0.5) * cs, t, cs < 24 ? 11 : 12);
        }
        ctx.save(); ctx.strokeStyle = t.accent2; ctx.lineWidth = 3;
        ctx.strokeRect(zx + (sel.u - win.u) * cs, (sel.v - win.v) * cs, cs, cs); ctx.restore();
        L.draw.line(ctx, win.u * sc + n * sc, win.v * sc, zx, 0, t.faint, 1, [3, 3]);
        L.draw.line(ctx, win.u * sc + n * sc, (win.v + n) * sc, zx, S, t.faint, 1, [3, 3]);
      };
      frameListeners.push(() => { bmp = bitmap(frame); info(); c.redraw(); });
      info();
    }

    // ================================================================ 2. luma
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>From colour to one brightness number</h3>
      <p>Cameras deliver three numbers per pixel (red, green, blue). Tracking needs one. We use <b>luma</b>, a weighted sum matching how bright each colour looks to a human eye:</p>
      <div class="eq-card"><div class="eq-label">Luma (Rec. 709) · frame.rs, pyr.wgsl</div>
        $$I = 0.2126\,R + 0.7152\,G + 0.0722\,B$$
        <div class="parts">
          <span>$R, G, B$</span><span>the colour channels (0–255, or 0–1)</span>
          <span>$0.7152$</span><span>green dominates: the eye is most sensitive to green</span>
          <span>$0.0722$</span><span>blue looks dark, so it counts little</span>
          <span>sum of weights</span><span>$= 1$, so white stays white: $R=G=B=255 \Rightarrow I=255$</span>
        </div>
      </div>
      <p><b>Example.</b> $(R,G,B) = (200, 100, 50)$: $0.2126\cdot200 + 0.7152\cdot100 + 0.0722\cdot50 = 42.52 + 71.52 + 3.61 = 117.65$, stored as 118.</p>
      <p class="note">Another common set of weights (Rec. 601) is $0.299, 0.587, 0.114$. Any fixed choice works for DTAM: tracking only needs brightness to be <i>consistent</i> between frames. This implementation uses Rec. 709 on the raw (gamma-encoded) values. The cost volume (chapter 8) uses all three colour channels.</p>
    `);
    {
      const fig = L.figure(root, "<b>Colour → luma.</b> Move the sliders. Pure blue at full strength is darker than mid-grey; pure green is almost as bright as white.");
      const c = L.canvas(fig.el, { aspect: 0.3, scroll: true, maxHeight: 200 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const upd = () => {
        const [R, G, B] = [sr.value, sg.value, sb.value];
        const Y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
        out.html = `I = 0.2126·${R} + 0.7152·${G} + 0.0722·${B} = ${(0.2126 * R).toFixed(1)} + ${(0.7152 * G).toFixed(1)} + ${(0.0722 * B).toFixed(1)} = <b>${Y.toFixed(1)}</b> · plain average (R+G+B)/3 = ${((R + G + B) / 3).toFixed(1)}`;
        c.redraw();
      };
      const sr = L.slider(ctl, { label: "R", min: 0, max: 255, step: 1, value: 40, oninput: upd });
      const sg = L.slider(ctl, { label: "G", min: 0, max: 255, step: 1, value: 60, oninput: upd });
      const sb = L.slider(ctl, { label: "B", min: 0, max: 255, step: 1, value: 230, oninput: upd });
      c.draw = (ctx) => {
        const t = L.theme();
        const [R, G, B] = [sr.value, sg.value, sb.value];
        const Y = 0.2126 * R + 0.7152 * G + 0.0722 * B, A = (R + G + B) / 3;
        const boxes = [["colour", `rgb(${R},${G},${B})`], ["luma I", L.gray(Y / 255)], ["plain average", L.gray(A / 255)]];
        const gap = 10, bw = (c.w - 2 * gap) / 3, bh = c.h - 22;
        boxes.forEach(([lab, col], k) => {
          const x = k * (bw + gap);
          ctx.fillStyle = col; ctx.strokeStyle = t.line; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.roundRect(x, 0, bw, bh, 8); ctx.fill(); ctx.stroke();
          L.draw.text(ctx, lab, x + bw / 2, c.h - 5, t.muted, { size: 12, align: "center" });
        });
      };
      upd();
    }

    // ================================================================ 3. bilinear
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Between pixels: bilinear interpolation</h3>
      <p>In this implementation <b>pixel centres sit at integer coordinates</b> (geom.rs): pixel $(3, 5)$ covers $u \in [2.5, 3.5]$, $v \in [4.5, 5.5]$.</p>
      <p>Projecting a 3D point almost never lands on a centre; it lands at, say, $(37.42, 18.9)$. DTAM does this millions of times per frame, and its optimisers nudge positions by tiny fractions of a pixel, so we need $I$ at <b>any</b> real-valued position.</p>
      <p><b>Bilinear interpolation</b>: blend the 4 surrounding pixels, each weighted by how close the point is to it.</p>
      <div class="eq-card"><div class="eq-label">Bilinear sampling</div>
        $$I(u, v) = (1-a)(1-b)\,I_{00} + a(1-b)\,I_{10} + (1-a)\,b\,I_{01} + a\,b\,I_{11}$$
        <div class="parts">
          <span>$u_0 = \lfloor u \rfloor,\ v_0 = \lfloor v \rfloor$</span><span>the top-left of the 4 neighbours (round down)</span>
          <span>$a = u - u_0,\ b = v - v_0$</span><span>fractional parts, each in $[0, 1)$</span>
          <span>$I_{00}, I_{10}, I_{01}, I_{11}$</span><span>$I(u_0, v_0)$, $I(u_0{+}1, v_0)$, $I(u_0, v_0{+}1)$, $I(u_0{+}1, v_0{+}1)$</span>
          <span>weights</span><span>each equals the area of the rectangle <b>opposite</b> its corner; they sum to 1</span>
        </div>
      </div>
      <p><b>Example.</b> $(u, v) = (2.25, 7.5)$ with $I(2,7)=0.2$, $I(3,7)=0.6$, $I(2,8)=0.4$, $I(3,8)=1.0$. Then $a = 0.25$, $b = 0.5$.<br>
      Top row: $0.75\cdot0.2 + 0.25\cdot0.6 = 0.30$. Bottom row: $0.75\cdot0.4 + 0.25\cdot1.0 = 0.55$. Blend: $0.5\cdot0.30 + 0.5\cdot0.55 = 0.425$.</p>
    `);
    {
      const fig = L.figure(root, "<b>Bilinear sampling.</b> Drag the orange point. Each neighbour's weight is the area of the opposite sub-rectangle. The strip below shows a row profile: bilinear is continuous, nearest-neighbour jumps.");
      const c = L.canvas(fig.el, { aspect: 0.82, maxHeight: 560 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const W = 5, H = 4;
      const R0 = L.rng(11);
      let G = { w: W, h: H, data: Float32Array.from({ length: W * H }, () => R0.float(0.05, 0.95, 2)) };
      let p = { u: 1.3, v: 1.6 }, nearest = false;
      const lay = () => {
        const gh = c.h * 0.68;
        const cs = Math.min((c.w - 8) / W, (gh - 8) / H);
        const ox = (c.w - cs * W) / 2, oy = 4;
        return { cs, ox, oy, gh };
      };
      const toC = (u, v) => { const { cs, ox, oy } = lay(); return { x: ox + (u + 0.5) * cs, y: oy + (v + 0.5) * cs }; };
      const val = (u, v) => nearest ? px(G, Math.round(u), Math.round(v)) : sample(G, u, v);
      const info = () => {
        const u0 = Math.min(Math.floor(p.u), W - 2), v0 = Math.min(Math.floor(p.v), H - 2);
        const a = p.u - u0, b = p.v - v0;
        const I = (i, j) => px(G, u0 + i, v0 + j).toFixed(2);
        if (nearest) {
          out.html = `nearest: round(${p.u.toFixed(2)}, ${p.v.toFixed(2)}) = (${Math.round(p.u)}, ${Math.round(p.v)}) → I = <b>${px(G, Math.round(p.u), Math.round(p.v)).toFixed(3)}</b>`;
        } else {
          out.html = `a = ${a.toFixed(2)}, b = ${b.toFixed(2)} · I = ${((1 - a) * (1 - b)).toFixed(3)}·${I(0, 0)} + ${(a * (1 - b)).toFixed(3)}·${I(1, 0)} + ${((1 - a) * b).toFixed(3)}·${I(0, 1)} + ${(a * b).toFixed(3)}·${I(1, 1)} = <b>${sample(G, p.u, p.v).toFixed(3)}</b>`;
        }
      };
      L.drag(c, () => [toC(p.u, p.v)], (_, q) => {
        const { cs, ox, oy } = lay();
        p.u = clamp((q.x - ox) / cs - 0.5, 0, W - 1);
        p.v = clamp((q.y - oy) / cs - 0.5, 0, H - 1);
        info();
      });
      L.toggle(ctl, "nearest neighbour instead", false, (v) => { nearest = v; info(); c.redraw(); });
      L.button(ctl, "New pixel values", () => {
        G = { w: W, h: H, data: Float32Array.from({ length: W * H }, () => R0.float(0.05, 0.95, 2)) };
        info(); c.redraw();
      });
      c.draw = (ctx) => {
        const t = L.theme(), { cs, ox, oy, gh } = lay();
        for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
          ctx.fillStyle = L.gray(px(G, i, j));
          ctx.fillRect(ox + i * cs, oy + j * cs, cs + 0.5, cs + 0.5);
        }
        ctx.save(); ctx.strokeStyle = t.line; ctx.lineWidth = 1; ctx.strokeRect(ox, oy, cs * W, cs * H); ctx.restore();
        const u0 = Math.min(Math.floor(p.u), W - 2), v0 = Math.min(Math.floor(p.v), H - 2);
        const a = p.u - u0, b = p.v - v0;
        const P = toC(p.u, p.v), A = toC(u0, v0), B = toC(u0 + 1, v0 + 1);
        if (!nearest) {
          // the 4 sub-rectangles; each shaded with the weight of the opposite corner
          ctx.save(); ctx.globalAlpha = 0.28; ctx.fillStyle = t.accent;
          ctx.fillRect(A.x, A.y, B.x - A.x, B.y - A.y); ctx.restore();
          ctx.save(); ctx.strokeStyle = t.accent; ctx.lineWidth = 2;
          ctx.strokeRect(A.x, A.y, B.x - A.x, B.y - A.y); ctx.restore();
          L.draw.line(ctx, P.x, A.y, P.x, B.y, t.accent, 1.5, [4, 3]);
          L.draw.line(ctx, A.x, P.y, B.x, P.y, t.accent, 1.5, [4, 3]);
          const wts = [[0, 0, (1 - a) * (1 - b)], [1, 0, a * (1 - b)], [0, 1, (1 - a) * b], [1, 1, a * b]];
          for (const [i, j, wgt] of wts) {
            // label the weight in the sub-rectangle opposite corner (i, j)
            const xm = i ? (A.x + P.x) / 2 : (P.x + B.x) / 2, ym = j ? (A.y + P.y) / 2 : (P.y + B.y) / 2;
            if (Math.abs((i ? P.x - A.x : B.x - P.x)) > 26 && Math.abs((j ? P.y - A.y : B.y - P.y)) > 14) pill(ctx, "w" + i + j + "=" + wgt.toFixed(2), xm, ym, t, 11, t.accent);
          }
        } else {
          const n = toC(Math.round(p.u), Math.round(p.v));
          ctx.save(); ctx.strokeStyle = t.accent2; ctx.lineWidth = 3; ctx.strokeRect(n.x - cs / 2, n.y - cs / 2, cs, cs); ctx.restore();
        }
        for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
          const q = toC(i, j);
          L.draw.dot(ctx, q.x, q.y, 3, t.accent3);
          pill(ctx, px(G, i, j).toFixed(2), q.x, q.y + Math.min(16, cs * 0.28), t, 11);
          if (j === 0) L.draw.text(ctx, "u=" + i, q.x, oy + 13, t.fg, { size: 11, align: "center" });
        }
        L.draw.handle(ctx, P.x, P.y, t.accent2);
        // profile strip along row v
        const pl = { x0: 0, x1: W - 1, y0: 0, y1: 1 };
        const top = gh + 8, bot = c.h - 16;
        const X = (u) => ox + (u + 0.5) * cs, Y = (y) => bot - y * (bot - top);
        L.draw.line(ctx, X(0), bot, X(W - 1), bot, t.line, 1);
        L.draw.line(ctx, X(0), top, X(W - 1), top, t.line, 1);
        const lin = [], nn = [];
        for (let k = 0; k <= 200; k++) {
          const u = pl.x0 + (k / 200) * (pl.x1 - pl.x0);
          lin.push([X(u), Y(sample(G, u, p.v))]);
          nn.push([X(u), Y(px(G, Math.round(u), Math.round(p.v)))]);
        }
        L.draw.path(ctx, nn, t.faint, 1.5, [4, 3]);
        L.draw.path(ctx, lin, t.accent, 2);
        L.draw.dot(ctx, X(p.u), Y(val(p.u, p.v)), 5, t.accent2);
        L.draw.text(ctx, `row v = ${p.v.toFixed(2)}:  bilinear (solid), nearest (dashed)`, X(0), c.h - 3, t.muted, { size: 11 });
      };
      info();
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>Two reasons we never just round to the nearest pixel:</p>
      <ul>
        <li><b>Accuracy</b>: depth and pose come from sub-pixel shifts.</li>
        <li><b>Smoothness</b>: optimisers (chapters 5, 6, 11) follow slopes. A nearest-neighbour image is flat steps: slope 0 almost everywhere, then a jump. Bilinear changes continuously.</li>
      </ul>
      <pre><code>function sample(I, u, v):           # I is w×h, pixel centres at integers
    u = clamp(u, 0, w-1);  v = clamp(v, 0, h-1)   # outside: repeat the border
    u0 = floor(u);  v0 = floor(v)
    u1 = min(u0+1, w-1);  v1 = min(v0+1, h-1)
    a = u - u0;  b = v - v0
    top    = (1-a)*I[v0][u0] + a*I[v0][u1]
    bottom = (1-a)*I[v1][u0] + a*I[v1][u1]
    return (1-b)*top + b*bottom</code></pre>
      <p class="note">GPUs have this built in (texture sampling), but this implementation does it by hand in its shaders (<code>klt.wgsl</code>, <code>track_common.wgsl</code>) to keep full 32-bit precision.</p>
    `);

    // ================================================================ 4. gradients
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Gradients: how brightness changes</h3>
      <p>The <b>slope</b> of the image in each direction is estimated from neighbours. This implementation uses <b>central differences</b>:</p>
      <div class="eq-card"><div class="eq-label">Image gradient (central differences)</div>
        $$I_x(\mathbf u) = \tfrac12\big(I(u{+}1, v) - I(u{-}1, v)\big), \qquad I_y(\mathbf u) = \tfrac12\big(I(u, v{+}1) - I(u, v{-}1)\big)$$
        $$\nabla I = (I_x,\, I_y), \qquad |\nabla I| = \sqrt{I_x^2 + I_y^2}$$
        <div class="parts">
          <span>$I_x$</span><span>brightness change per pixel moving right</span>
          <span>$I_y$</span><span>brightness change per pixel moving <b>down</b></span>
          <span>$\tfrac12$</span><span>the two samples are 2 pixels apart: rise / run = difference / 2</span>
          <span>$\nabla I$</span><span>the gradient vector: points towards <b>steepest brightness increase</b>, perpendicular to edges</span>
          <span>$|\nabla I|$</span><span>edge strength: 0 on flat areas, large across edges</span>
        </div>
      </div>
      <p><b>Example.</b> $I(u{-}1,v)=0.30$, $I(u{+}1,v)=0.70$, $I(u,v{-}1)=0.50$, $I(u,v{+}1)=0.20$. Then $I_x = 0.20$, $I_y = -0.15$, $|\nabla I| = \sqrt{0.04+0.0225} = 0.25$. The gradient points right and up (brighter that way).</p>
      <div class="key">The gradient predicts small moves. Stepping by $(\delta u, \delta v)$ changes brightness by about
      $$I(u{+}\delta u,\, v{+}\delta v) \approx I(u, v) + I_x\,\delta u + I_y\,\delta v.$$
      This one line is the heart of every tracker in this guide: it turns "which shift makes the images match?" into a linear equation.</div>
      <p class="note">Along an edge (perpendicular to $\nabla I$) brightness does not change: $I_x\delta u + I_y\delta v = 0$. Across it (along $\nabla I$) it changes fastest.</p>
    `);
    {
      const fig = L.figure(root, "<b>Gradients on a real frame.</b> Drag the probe. Arrow: $\\nabla I$ (towards brighter). Dashed: the edge direction (no change). Switch views to see $|\\nabla I|$, $I_x$ and $I_y$ over the whole image (mid-grey = 0).");
      const c = L.canvas(fig.el, { aspect: 0.62 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      let probe = { u: 100, v: 90 }, view = "image";
      let views = {};
      const build = () => {
        const n = frame.w * frame.h, gx = new Float32Array(n), gy = new Float32Array(n);
        for (let v = 0; v < frame.h; v++) for (let u = 0; u < frame.w; u++) {
          gx[v * frame.w + u] = 0.5 * (px(frame, u + 1, v) - px(frame, u - 1, v));
          gy[v * frame.w + u] = 0.5 * (px(frame, u, v + 1) - px(frame, u, v - 1));
        }
        const g = (x) => Math.round(clamp(x, 0, 1) * 255);
        views = {
          image: bitmap(frame),
          mag: bitmap(frame, (i) => [0, 0, 0].fill(g(4 * Math.hypot(gx[i], gy[i])))),
          ix: bitmap(frame, (i) => [0, 0, 0].fill(g(0.5 + 3 * gx[i]))),
          iy: bitmap(frame, (i) => [0, 0, 0].fill(g(0.5 + 3 * gy[i]))),
        };
      };
      build();
      const lay = () => { const S = Math.min(c.h, c.w * 0.62); return { S, sc: S / frame.w, zx: S + 10, zs: Math.min(c.w - S - 10, c.h) }; };
      const info = () => {
        const { u, v } = probe;
        const l = px(frame, u - 1, v), r = px(frame, u + 1, v), up = px(frame, u, v - 1), dn = px(frame, u, v + 1);
        const Ix = 0.5 * (r - l), Iy = 0.5 * (dn - up);
        const ang = (Math.atan2(-Iy, Ix) * 180) / Math.PI;
        out.html = `(u, v) = (${u}, ${v}) · I<sub>x</sub> = (${r.toFixed(3)} − ${l.toFixed(3)})/2 = <b>${Ix.toFixed(3)}</b> · I<sub>y</sub> = (${dn.toFixed(3)} − ${up.toFixed(3)})/2 = <b>${Iy.toFixed(3)}</b> · |∇I| = <b>${Math.hypot(Ix, Iy).toFixed(3)}</b> · direction ${ang.toFixed(0)}° (counter-clockwise from right, as seen on screen)`;
      };
      L.drag(c, () => { const { sc } = lay(); return [{ x: (probe.u + 0.5) * sc, y: (probe.v + 0.5) * sc }]; }, (_, q) => {
        const { sc, S } = lay();
        probe.u = clamp(Math.floor(Math.min(q.x, S - 1) / sc), 1, frame.w - 2);
        probe.v = clamp(Math.floor(Math.min(q.y, S - 1) / sc), 1, frame.h - 2);
        info();
      });
      for (const [k, lab] of [["image", "Image"], ["mag", "|∇I|"], ["ix", "I<sub>x</sub>"], ["iy", "I<sub>y</sub>"]]) {
        L.button(ctl, lab, () => { view = k; c.redraw(); });
      }
      c.draw = (ctx) => {
        const t = L.theme(), { S, sc, zx, zs } = lay();
        blit(ctx, views[view], 0, 0, S, S);
        const { u, v } = probe;
        const Ix = 0.5 * (px(frame, u + 1, v) - px(frame, u - 1, v)), Iy = 0.5 * (px(frame, u, v + 1) - px(frame, u, v - 1));
        const m = Math.hypot(Ix, Iy), cx = (u + 0.5) * sc, cy = (v + 0.5) * sc;
        if (m > 1e-4) {
          const L0 = 16 + Math.min(40, 400 * m), dx = Ix / m, dy = Iy / m;
          L.draw.line(ctx, cx - dy * 28, cy + dx * 28, cx + dy * 28, cy - dx * 28, t.accent3, 2, [5, 4]);
          L.draw.arrow(ctx, cx, cy, cx + dx * L0, cy + dy * L0, t.accent2, 3);
        }
        L.draw.handle(ctx, cx, cy, t.accent);
        // 3x3 neighbourhood
        const cs = zs / 3;
        for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
          const x = zx + (i + 1) * cs, y = (j + 1) * cs, val = px(frame, u + i, v + j);
          ctx.fillStyle = L.gray(val); ctx.fillRect(x, y, cs + 0.5, cs + 0.5);
          if (i * j === 0 && i + j !== 0) {
            ctx.save(); ctx.strokeStyle = i ? t.accent2 : t.accent4; ctx.lineWidth = 3; ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4); ctx.restore();
          }
          pill(ctx, val.toFixed(2), x + cs / 2, y + cs / 2, t, cs > 50 ? 12 : 11);
        }
        L.draw.text(ctx, "3×3 around the probe", zx, Math.min(c.h - 3, zs + 14), t.muted, { size: 11 });
      };
      frameListeners.push(() => { build(); info(); c.redraw(); });
      info();
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <pre><code>Ix = 0.5 * (sample(I, u+1, v) - sample(I, u-1, v))
Iy = 0.5 * (sample(I, u, v+1) - sample(I, u, v-1))</code></pre>
      <p>That is exactly what <code>klt.wgsl</code> and <code>track_common.wgsl</code> compute (with bilinear samples, so it works at non-integer positions too). Chapter 10 also uses <i>forward</i> differences, $I(u{+}1,v) - I(u,v)$, for the regulariser.</p>
    `);

    // ================================================================ 5. smoothing
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Smoothing</h3>
      <p>Pixel values are noisy, and a difference of two noisy numbers is noisier still. <b>Averaging neighbours first</b> cancels much of the noise. The simplest smoother is the <b>box filter</b>: replace each pixel by the mean of a $k\times k$ window around it.</p>
      <div class="eq-card"><div class="eq-label">3×3 box filter</div>
        $$\bar I(u, v) = \frac19 \sum_{i=-1}^{1}\sum_{j=-1}^{1} I(u+i,\, v+j)$$
        <div class="parts">
          <span>$\sum_i \sum_j$</span><span>add up all 9 pixels of the window centred on $(u, v)$</span>
          <span>$\frac19$</span><span>divide by the count: weights sum to 1, so a flat image is unchanged</span>
        </div>
      </div>
      <p>Weighted averages work too, as long as the weights sum to 1. The corner tracker's pyramid uses weights $\tfrac18(1, 3, 3, 1)$ along each axis (<code>downsample.wgsl</code>), which blurs a little more smoothly than a box.</p>
    `);
    {
      const fig = L.figure(root, "<b>Smoothing before differentiating.</b> Top: a noisy row of pixels (faint) and its box-filtered version. Bottom: their central-difference gradients. Increase $k$ and watch the gradient noise drop while the edge stays visible.");
      const c = L.canvas(fig.el, { aspect: 0.7, scroll: true });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const N = 120;
      let seed = 4, sig = [];
      const make = () => {
        const r = L.rng(seed);
        sig = Array.from({ length: N }, (_, x) => {
          let v = x < 40 ? 0.2 : 0.7;
          if (x >= 70) v = 0.7 - Math.min(1, (x - 70) / 25) * 0.4;
          const g = Math.sqrt(-2 * Math.log(r.next() + 1e-9)) * Math.cos(2 * Math.PI * r.next());
          return v + 0.05 * g;
        });
      };
      make();
      const box = (a, k) => a.map((_, x) => {
        let s = 0; const h = (k - 1) / 2;
        for (let i = -h; i <= h; i++) s += a[clamp(x + i, 0, a.length - 1)];
        return s / k;
      });
      const grad = (a) => a.map((_, x) => 0.5 * (a[clamp(x + 1, 0, a.length - 1)] - a[clamp(x - 1, 0, a.length - 1)]));
      const sk = L.slider(ctl, { label: "box width $k$", min: 1, max: 15, step: 2, value: 5, oninput: () => c.redraw() });
      L.button(ctl, "New noise", () => { seed++; make(); c.redraw(); });
      c.draw = (ctx) => {
        const t = L.theme(), k = sk.value, sm = box(sig, k);
        const half = c.h / 2;
        const p1 = L.plot({ w: c.w, h: half }, { x0: 0, x1: N - 1, y0: 0, y1: 1, pad: [8, 8, 20, 34] });
        p1.axes(ctx, { xticks: 4, yticks: 2, fmt: (v) => Math.round(v * 100) / 100 });
        L.draw.path(ctx, sig.map((v, x) => [p1.X(x), p1.Y(v)]), t.faint, 1.5);
        L.draw.path(ctx, sm.map((v, x) => [p1.X(x), p1.Y(v)]), t.accent, 2.5);
        L.draw.text(ctx, "I (faint) and box-smoothed Ī", p1.X(0) + 4, 20, t.muted, { size: 12 });
        ctx.save(); ctx.translate(0, half);
        const p2 = L.plot({ w: c.w, h: half }, { x0: 0, x1: N - 1, y0: -0.2, y1: 0.3, pad: [8, 8, 20, 34] });
        p2.axes(ctx, { xticks: 4, yticks: 2, fmt: (v) => Math.round(v * 100) / 100 });
        L.draw.path(ctx, grad(sig).map((v, x) => [p2.X(x), p2.Y(clamp(v, -0.2, 0.3))]), t.faint, 1.5);
        L.draw.path(ctx, grad(sm).map((v, x) => [p2.X(x), p2.Y(v)]), t.accent2, 2.5);
        L.draw.text(ctx, "gradient of I (faint) and of Ī", p2.X(0) + 4, 20, t.muted, { size: 12 });
        ctx.restore();
      };
    }

    // ================================================================ 6. pyramids
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Image pyramids</h3>
      <p>An <b>image pyramid</b> is the same image at halving resolutions: level 0 is the original, level $l$ has $2^l$ times fewer pixels along each side. Each level is made from the one below by averaging 2×2 blocks:</p>
      <div class="eq-card"><div class="eq-label">2×2 box downsample · frame.rs, pyr.wgsl</div>
        $$I_{l+1}(x, y) = \tfrac14\big(I_l(2x, 2y) + I_l(2x{+}1, 2y) + I_l(2x, 2y{+}1) + I_l(2x{+}1, 2y{+}1)\big)$$
        <div class="parts">
          <span>$(x, y)$</span><span>a pixel of the coarser level $l+1$</span>
          <span>$(2x, 2y) \ldots$</span><span>the 2×2 block of level $l$ it covers</span>
          <span>size</span><span>$w_{l+1} = \lceil w_l / 2\rceil$ (round up; a missing column/row repeats the last one)</span>
        </div>
      </div>
      <p><b>Example.</b> A 640×480 image: levels 320×240, 160×120, 80×60. A 101-wide image: $\lceil 101/2\rceil = 51$ (the last output column averages column 100 with itself).</p>
      <pre><code>function downsample2(I):                 # w×h  ->  ceil(w/2)×ceil(h/2)
    for each output pixel (x, y):
        s = 0
        for (dx, dy) in (0,0), (1,0), (0,1), (1,1):
            s += I[min(2y+dy, h-1)][min(2x+dx, w-1)]
        out[y][x] = s / 4

pyramid[0] = I
for l in 1 .. levels-1:  pyramid[l] = downsample2(pyramid[l-1])</code></pre>
    `);
    {
      const fig = L.figure(root, "<b>A 5-level pyramid.</b> Drag the point on level 0. The same scene point is marked on every level, using the pixel-centre rule below; the outlined square is the pixel it falls in.");
      const c = L.canvas(fig.el, { aspect: 0.66 });
      fig.add(c.el);
      const out = L.readout(fig.el); fig.add(out.el);
      const NL = 5;
      let pyr = [], bmps = [], pt = { u: 120.0, v: 70.0 };
      const build = () => {
        pyr = [frame];
        for (let l = 1; l < NL; l++) pyr.push(down2(pyr[l - 1]));
        bmps = pyr.map((im) => bitmap(im));
      };
      build();
      const lay = () => {
        const S = Math.min(c.h, (c.w - 8) / 1.5);
        const rects = [{ x: 0, y: 0, s: S }];
        let y = 0;
        for (let l = 1; l < NL; l++) { const s = S / 2 ** l; rects.push({ x: S + 8, y, s }); y += s + 4; }
        return { S, rects };
      };
      const toL = (u, l) => (u + 0.5) / 2 ** l - 0.5;
      const info = () => {
        out.html = pyr.map((im, l) => `L${l} ${im.w}×${im.h}: (<b>${toL(pt.u, l).toFixed(2)}</b>, <b>${toL(pt.v, l).toFixed(2)}</b>)`).join(" · ");
      };
      L.drag(c, () => { const { S } = lay(); const s = S / frame.w; return [{ x: (pt.u + 0.5) * s, y: (pt.v + 0.5) * s }]; }, (_, q) => {
        const { S } = lay(), s = S / frame.w;
        pt.u = clamp(Math.min(q.x, S) / s - 0.5, 0, frame.w - 1);
        pt.v = clamp(q.y / s - 0.5, 0, frame.h - 1);
        info();
      });
      c.draw = (ctx) => {
        const t = L.theme(), { rects } = lay();
        rects.forEach((r, l) => {
          const im = pyr[l];
          blit(ctx, bmps[l], r.x, r.y, r.s, r.s);
          const s = r.s / im.w;
          const ul = toL(pt.u, l), vl = toL(pt.v, l);
          if (l > 0) {
            ctx.save(); ctx.strokeStyle = t.accent2; ctx.lineWidth = 2;
            ctx.strokeRect(r.x + (Math.round(ul)) * s, r.y + Math.round(vl) * s, s, s); ctx.restore();
            L.draw.dot(ctx, r.x + (ul + 0.5) * s, r.y + (vl + 0.5) * s, 3, t.accent);
          }
        });
        const s0 = rects[0].s / frame.w;
        L.draw.handle(ctx, (pt.u + 0.5) * s0, (pt.v + 0.5) * s0, t.accent);
        rects.forEach((r, l) => { if (r.s >= 22) pill(ctx, "L" + l, r.x + 12, r.y + 10, t, 11); });
      };
      frameListeners.push(() => { build(); info(); c.redraw(); });
      info();
    }

    // ================================================================ 7. pixel-centre convention
    root.insertAdjacentHTML("beforeend", String.raw`
      <h4>Mapping coordinates between levels</h4>
      <p>Level-1 pixel 0 is the average of level-0 pixels 0 and 1, so its centre lies at level-0 position $0.5$, not $0$. With pixel centres at integers, the correct rule (<code>Intrinsics::level</code> in geom.rs, <code>to_level</code> in klt.wgsl) is:</p>
      <div class="eq-card"><div class="eq-label">Pixel-centred level mapping</div>
        $$u_l = \frac{u_0 + 0.5}{2^l} - 0.5, \qquad u_0 = (u_l + 0.5)\,2^l - 0.5$$
        <div class="parts">
          <span>$u_0 + 0.5$</span><span>shift so the image's left <b>edge</b> is at 0 (edges, not centres, scale cleanly)</span>
          <span>$\div 2^l$</span><span>pixels are $2^l$ times wider at level $l$</span>
          <span>$-\,0.5$</span><span>shift back so centres are at integers again</span>
          <span>displacements</span><span>a <i>difference</i> of positions just scales: $\Delta u_l = \Delta u_0 / 2^l$</span>
        </div>
      </div>
      <p><b>Example.</b> $u_0 = 10$ at level 2: $(10.5)/4 - 0.5 = 2.125$. Naively $10/4 = 2.5$ would be off by $0.375$ level-2 pixels (1.5 original pixels).</p>
    `);
    {
      const fig = L.figure(root, "<b>Why the ±0.5.</b> Each row is one pyramid level covering the same 8 original pixels. Drag the marker: the green dot (pixel-centred rule) stays on the same scene position in every row; the red ring (naive $u_0/2^l$) drifts left.");
      const c = L.canvas(fig.el, { aspect: 0.55, maxHeight: 360 });
      fig.add(c.el);
      const out = L.readout(fig.el); fig.add(out.el);
      let u0 = 5.0;
      const lay = () => { const pad = 12, W = c.w - 2 * pad, rowH = (c.h - 30) / 4; return { pad, W, rowH, cw0: W / 8 }; };
      const info = () => {
        out.html = [0, 1, 2, 3].map((l) => `L${l}: u = <b>${((u0 + 0.5) / 2 ** l - 0.5).toFixed(3)}</b> (naive ${(u0 / 2 ** l).toFixed(3)})`).join(" · ");
      };
      L.drag(c, () => { const { pad, cw0 } = lay(); return [{ x: pad + (u0 + 0.5) * cw0, y: c.h - 14 }]; }, (_, q) => {
        const { pad, cw0 } = lay();
        u0 = clamp((q.x - pad) / cw0 - 0.5, -0.5, 7.5);
        info();
      });
      c.draw = (ctx) => {
        const t = L.theme(), { pad, W, rowH, cw0 } = lay();
        for (let l = 0; l < 4; l++) {
          const n = 8 / 2 ** l, cw = W / n, y = l * rowH + 4, h = rowH - 8;
          for (let i = 0; i < n; i++) {
            ctx.save(); ctx.fillStyle = i % 2 ? t.panel2 : t.panel; ctx.strokeStyle = t.line;
            ctx.fillRect(pad + i * cw, y, cw, h); ctx.strokeRect(pad + i * cw, y, cw, h); ctx.restore();
            L.draw.line(ctx, pad + (i + 0.5) * cw, y + h - 8, pad + (i + 0.5) * cw, y + h, t.faint, 1.5);
            L.draw.text(ctx, String(i), pad + (i + 0.5) * cw, y + h - 11, t.faint, { size: 11, align: "center" });
          }
          L.draw.text(ctx, "L" + l, pad + 4, y + 14, t.muted, { size: 12, bold: true });
          const ul = (u0 + 0.5) / 2 ** l - 0.5, naive = u0 / 2 ** l;
          const xs = pad + (ul + 0.5) * cw, xn = pad + (naive + 0.5) * cw;
          if (l > 0) {
            ctx.save(); ctx.strokeStyle = t.bad; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(xn, y + h / 2, 8, 0, 7); ctx.stroke(); ctx.restore();
          }
          L.draw.dot(ctx, xs, y + h / 2, 6, t.accent3);
        }
        const xh = pad + (u0 + 0.5) * cw0;
        L.draw.line(ctx, xh, 4, xh, c.h - 14, t.accent, 1, [3, 3]);
        L.draw.handle(ctx, xh, c.h - 14, t.accent);
      };
      info();
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>The camera's intrinsics follow the same rule (chapter 3 explains them): focal length halves each level, $f_l = f/2^l$, and the image centre maps like any other point, $c_l = (c + 0.5)/2^l - 0.5$.</p>
    `);

    // ================================================================ 8. coarse to fine
    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>Why coarse-to-fine</h3>
      <p>Trackers find a shift by repeatedly stepping "downhill" using the gradient (chapters 5, 6, 11). That only works if the start is already in the right valley, roughly within the size of the image's smallest features. A 30-pixel motion is far outside that.</p>
      <ul>
        <li>At level 3 a 30-pixel motion is only $30/8 \approx 3.75$ pixels.</li>
        <li>The coarse image is also smoother: fine texture, which creates many false valleys, has been averaged away.</li>
        <li>So: solve at the top, double the answer, refine one level down, repeat.</li>
      </ul>
      <pre><code>d = initial_guess / 2^top             # displacement, in top-level pixels
for l = top down to 0:
    repeat a few times:  d += small_step(pyramid_A[l], pyramid_B[l], d)
    if l > 0:  d = 2 * d               # one level finer: pixels are half as big</code></pre>
    `);
    {
      const fig = L.figure(root, "<b>Coarse-to-fine on a 1D image.</b> The bottom plot shows the matching error for every candidate shift, at each level (bold = selected level). Set a large true shift, then compare <i>Level 0 only</i> (gets stuck in a false valley) with <i>Coarse-to-fine</i>.");
      const c = L.canvas(fig.el, { aspect: 0.72 });
      fig.add(c.el);
      const ctl = L.controls(fig.el); fig.add(ctl);
      const out = L.readout(fig.el); fig.add(out.el);
      const N = 512, TOP = 4;
      const f = (x) => 0.5 + 0.2 * Math.sin(2 * Math.PI * x / 170) + 0.12 * Math.sin(2 * Math.PI * x / 47 + 1) + 0.08 * Math.sin(2 * Math.PI * x / 13 + 2) + 0.06 * Math.sin(2 * Math.PI * x / 5.3);
      const pyr1 = (a) => {
        const P = [a];
        for (let l = 1; l <= TOP; l++) {
          const s = P[l - 1], n = Math.ceil(s.length / 2), o = new Float64Array(n);
          for (let i = 0; i < n; i++) o[i] = (s[2 * i] + s[Math.min(2 * i + 1, s.length - 1)]) / 2;
          P.push(o);
        }
        return P;
      };
      const samp = (a, x) => { x = clamp(x, 0, a.length - 1); const i = Math.floor(x), j = Math.min(i + 1, a.length - 1), t = x - i; return a[i] * (1 - t) + a[j] * t; };
      const P0 = pyr1(Float64Array.from({ length: N }, (_, x) => f(x)));
      let P1 = [], curves = [];
      const D0 = -10, D1 = 56;
      const margin = (l) => Math.ceil(64 / 2 ** l) + 2;
      const cost = (l, dl) => {
        const a = P0[l], b = P1[l], m = margin(l);
        let s = 0, n = 0;
        for (let x = m; x < a.length - m; x++) { const r = samp(b, x + dl) - a[x]; s += r * r; n++; }
        return s / n;
      };
      const setShift = () => {
        const s = ss.value;
        P1 = pyr1(Float64Array.from({ length: N }, (_, x) => f(x - s)));
        curves = [];
        for (let l = 0; l <= TOP; l++) {
          const pts = [];
          for (let k = 0; k <= 220; k++) { const D = D0 + (k / 220) * (D1 - D0); pts.push([D, cost(l, D / 2 ** l)]); }
          const mx = Math.max(...pts.map((p) => p[1])) || 1;
          curves.push(pts.map(([D, v]) => [D, v / mx]));
        }
        run = null; trail = [];
      };
      // one Lucas–Kanade step at level l (template gradient, as in klt.wgsl)
      const lkStep = (l, d) => {
        const a = P0[l], b = P1[l], m = margin(l);
        let sg = 0, sr = 0;
        for (let x = m; x < a.length - m; x++) {
          const g = 0.5 * (samp(a, x + 1) - samp(a, x - 1));
          sg += g * g; sr += g * (a[x] - samp(b, x + d));
        }
        return sg > 1e-12 ? sr / sg : 0;
      };
      let run = null, trail = [], acc = 0;
      const start = (top) => { run = { l: top, d: 0, it: 0, done: false, single: top === 0 }; trail = [{ l: top, D: 0 }]; };
      const ss = L.slider(ctl, { label: "true shift", min: 0, max: 44, step: 1, value: 30, oninput: () => { setShift(); info(); c.redraw(); } });
      const sl = L.slider(ctl, { label: "show level", min: 0, max: TOP, step: 1, value: TOP, oninput: () => c.redraw() });
      L.button(ctl, "Coarse-to-fine", () => { start(TOP); sl.value = TOP; });
      L.button(ctl, "Level 0 only", () => { start(0); sl.value = 0; });
      const info = () => {
        if (!run) { out.html = `true shift = <b>${ss.value}</b> px. Press a button to run.`; return; }
        const D = run.d * 2 ** run.l;
        out.html = `level ${run.l}, iteration ${run.it}: estimate d = ${run.d.toFixed(2)} level-${run.l} px = <b>${D.toFixed(2)}</b> original px (true ${ss.value})` + (run.done ? (Math.abs(D - ss.value) < 0.3 ? " · <b>converged ✓</b>" : " · <b>stuck in a wrong valley ✗</b>") : "");
      };
      L.loop(c, (_, dt) => {
        if (!run || run.done) return;
        acc += dt;
        if (acc < 0.12) return;
        acc = 0;
        run.d += lkStep(run.l, run.d);
        run.it++;
        trail.push({ l: run.l, D: run.d * 2 ** run.l });
        if (run.it >= (run.single ? 20 : 8)) {
          if (run.l > 0) { run.l--; run.d *= 2; run.it = 0; sl.value = run.l; } else run.done = true;
        }
        info(); c.redraw();
      });
      c.draw = (ctx) => {
        const t = L.theme(), lv = sl.value;
        const est = run ? run.d * 2 ** run.l : 0;
        // top: signals at the selected level
        const hTop = c.h * 0.36;
        const pa = L.plot({ w: c.w, h: hTop }, { x0: 0, x1: N, y0: 0.05, y1: 0.95, pad: [6, 8, 16, 34] });
        const a = P0[lv], b = P1[lv], s = 2 ** lv;
        const ptsA = [], ptsB = [];
        for (let i = 0; i < a.length; i++) {
          const x0 = (i + 0.5) * s;
          ptsA.push([pa.X(x0), pa.Y(a[i])]);
          ptsB.push([pa.X(x0), pa.Y(samp(b, i + est / s))]);
        }
        L.draw.path(ctx, ptsA, t.accent, 2);
        L.draw.path(ctx, ptsB, t.accent2, 1.5, [5, 3]);
        L.draw.text(ctx, `level ${lv}: reference (solid), live shifted back by the estimate (dashed)`, pa.X(0), 12, t.muted, { size: 11 });
        // bottom: cost curves
        ctx.save(); ctx.translate(0, hTop);
        const pc = L.plot({ w: c.w, h: c.h - hTop }, { x0: D0, x1: D1, y0: 0, y1: 1.05, pad: [10, 8, 30, 34] });
        pc.axes(ctx, { xticks: 6, yticks: 2, xlabel: "candidate shift (original px)", fmt: (v) => Math.round(v * 10) / 10 });
        const cols = [t.accent, t.accent3, t.accent4, t.accent2, t.fg];
        curves.forEach((pts, l) => {
          L.draw.path(ctx, pts.map(([D, v]) => [pc.X(D), pc.Y(v)]), cols[l], l === lv ? 3 : 1, l === lv ? null : [3, 3]);
        });
        L.draw.line(ctx, pc.X(ss.value), pc.Y(0), pc.X(ss.value), pc.Y(1.05), t.good, 1.5, [2, 3]);
        for (const tr of trail) {
          const pts = curves[tr.l]; const k = clamp(Math.round(((tr.D - D0) / (D1 - D0)) * 220), 0, 220);
          L.draw.dot(ctx, pc.X(clamp(tr.D, D0, D1)), pc.Y(pts[k][1]), 3, cols[tr.l]);
        }
        if (run) { const pts = curves[run.l]; const k = clamp(Math.round(((est - D0) / (D1 - D0)) * 220), 0, 220); L.draw.dot(ctx, pc.X(clamp(est, D0, D1)), pc.Y(pts[k][1]), 6, cols[run.l], t.panel); }
        let lx = pc.X(D0) + 4;
        for (let l = 0; l <= TOP; l++) { L.draw.text(ctx, "L" + l, lx, 12, cols[l], { size: 12, bold: l === lv }); lx += 26; }
        ctx.restore();
      };
      setShift(); info();
    }
    root.insertAdjacentHTML("beforeend", String.raw`
      <div class="key">Coarse levels have wide, simple valleys (large motions are small there); fine levels have sharp, accurate valleys. Coarse-to-fine gets both. DTAM's dense tracker (chapter 11) and the corner tracker (chapter 6) both run this way.</div>
    `);

    // ================================================================ quiz
    const tbl = (rows, colHead, rowHead) => {
      let h = `<table class="mat">`;
      if (colHead) h += `<tr><th></th>${colHead.map((x) => `<th>${x}</th>`).join("")}</tr>`;
      rows.forEach((r, i) => { h += `<tr>${rowHead ? `<th>${rowHead[i]}</th>` : ""}${r.map((x) => `<td>${x}</td>`).join("")}</tr>`; });
      return h + `</table>`;
    };
    L.quiz(root, "images", [
      { id: "luma", type: "num", gen: (r) => {
        const R = r.int(0, 255), G = r.int(0, 255), B = r.int(0, 255);
        const Y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
        return { q: String.raw`A pixel has $(R, G, B) = (${R}, ${G}, ${B})$. What is its luma $I$ with this implementation's (Rec. 709) weights, on the 0–255 scale? (1 decimal)`,
          answer: Y, tol: 0.06,
          explain: String.raw`$I = 0.2126\cdot${R} + 0.7152\cdot${G} + 0.0722\cdot${B} = ${fx(0.2126 * R)} + ${fx(0.7152 * G)} + ${fx(0.0722 * B)} = ${fx(Y, 2)}$.` };
      } },
      { id: "green", type: "mc",
        q: "Why does green get by far the largest weight in the luma formula?",
        choices: ["The human eye is most sensitive to green, so green contributes most to perceived brightness",
          "Camera sensors have twice as many green sites, so green is counted twice",
          "Green is always the largest channel in natural images",
          "It makes the weights sum to 1"],
        answer: 0,
        explain: "Luma approximates perceived brightness. The weights sum to 1 regardless of how they are split; it is the eye's sensitivity that makes green dominant." },
      { id: "index", type: "num", gen: (r) => {
        const w = r.pick([160, 192, 320, 640]), u = r.int(0, w - 1), v = r.int(1, 99);
        return { q: String.raw`An image is $${w}$ pixels wide, stored row-major. At which array index is pixel $(u, v) = (${u}, ${v})$? (indices start at 0)`,
          answer: v * w + u,
          explain: String.raw`$i = v\,w + u = ${v}\cdot${w} + ${u} = ${v * w + u}$. Each of the $${v}$ rows above holds $${w}$ pixels.` };
      } },
      { id: "unindex", type: "num", gen: (r) => {
        const w = r.pick([160, 192, 320, 640]), u = r.int(0, w - 1), v = r.int(1, 99), i = v * w + u;
        return { q: String.raw`In a row-major image $${w}$ pixels wide, array index $${i}$ holds which pixel $(u, v)$?`,
          answer: [u, v], labels: ["$u$", "$v$"],
          explain: String.raw`$v = \lfloor ${i} / ${w} \rfloor = ${v}$ (whole rows), $u = ${i} - ${v}\cdot${w} = ${u}$ (the remainder).` };
      } },
      { id: "bilin", type: "num", gen: (r) => {
        const u0 = r.int(2, 60), v0 = r.int(2, 40);
        const a = r.pick([0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8]), b = r.pick([0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8]);
        const I00 = r.float(0, 1, 2), I10 = r.float(0, 1, 2), I01 = r.float(0, 1, 2), I11 = r.float(0, 1, 2);
        const top = (1 - a) * I00 + a * I10, bot = (1 - a) * I01 + a * I11, val = (1 - b) * top + b * bot;
        return { q: String.raw`Pixel values: $I(${u0}, ${v0}) = ${I00}$, $I(${u0 + 1}, ${v0}) = ${I10}$, $I(${u0}, ${v0 + 1}) = ${I01}$, $I(${u0 + 1}, ${v0 + 1}) = ${I11}$. What is the bilinear sample at $(${fx(u0 + a)}, ${fx(v0 + b)})$? (3 decimals)`,
          answer: val, tol: 0.0015,
          explain: String.raw`$a = ${a}$, $b = ${b}$. Top row: $${fx(1 - a)}\cdot${I00} + ${a}\cdot${I10} = ${fx(top, 4)}$. Bottom row: $${fx(1 - a)}\cdot${I01} + ${a}\cdot${I11} = ${fx(bot, 4)}$. Blend: $${fx(1 - b)}\cdot${fx(top, 4)} + ${b}\cdot${fx(bot, 4)} = ${fx(val, 4)}$.` };
      } },
      { id: "bweight", type: "num", gen: (r) => {
        const u0 = r.int(3, 90), v0 = r.int(3, 90);
        const a = r.pick([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]), b = r.pick([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]);
        const [i, j] = r.pick([[0, 0], [1, 0], [0, 1], [1, 1]]);
        const wgt = (i ? a : 1 - a) * (j ? b : 1 - b);
        return { q: String.raw`When sampling bilinearly at $(${fx(u0 + a)}, ${fx(v0 + b)})$, what weight does pixel $(${u0 + i}, ${v0 + j})$ get?`,
          answer: wgt, tol: 1e-3,
          explain: String.raw`$a = ${a}$, $b = ${b}$. Pixel $(${u0 + i}, ${v0 + j})$ is the ${j ? "bottom" : "top"}-${i ? "right" : "left"} neighbour, weight $${i ? "a" : "(1-a)"}\,${j ? "b" : "(1-b)"} = ${fx(i ? a : 1 - a)}\cdot${fx(j ? b : 1 - b)} = ${fx(wgt, 4)}$: the area of the opposite sub-rectangle.` };
      } },
      { id: "nearest", type: "mc",
        q: "An optimiser adjusts a sub-pixel position by following the slope of the sampled brightness. Why is nearest-neighbour sampling a bad choice?",
        choices: ["Its output is piecewise constant: the slope is zero almost everywhere and jumps at pixel borders, so there is nothing to follow",
          "It is much slower to compute than bilinear sampling",
          "It returns values outside [0, 1]",
          "It only works at integer coordinates, so it cannot be evaluated at 3.7"],
        answer: 0,
        explain: "Nearest-neighbour can be evaluated anywhere (it just rounds), and it is fast, but the result is a staircase. Bilinear is continuous with a useful slope between pixel centres." },
      { id: "grad", type: "num", gen: (r) => {
        const g = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => r.float(0, 1, 2)));
        const u = r.int(5, 100), v = r.int(5, 100);
        const Ix = 0.5 * (g[1][2] - g[1][0]), Iy = 0.5 * (g[2][1] - g[0][1]), m = Math.hypot(Ix, Iy);
        return { q: String.raw`The 3×3 neighbourhood of pixel $(${u}, ${v})$ (columns $u$, rows $v$):` +
            tbl(g, [u - 1, u, u + 1], [v - 1, v, v + 1]) +
            String.raw`Compute $I_x$, $I_y$ (central differences) and $|\nabla I|$ at $(${u}, ${v})$. (3 decimals)`,
          answer: [Ix, Iy, m], labels: ["$I_x$", "$I_y$", "$|\\nabla I|$"], tol: 0.0015,
          explain: String.raw`$I_x = \tfrac12(I(${u + 1},${v}) - I(${u - 1},${v})) = \tfrac12(${g[1][2]} - ${g[1][0]}) = ${fx(Ix, 4)}$. $I_y = \tfrac12(I(${u},${v + 1}) - I(${u},${v - 1})) = \tfrac12(${g[2][1]} - ${g[0][1]}) = ${fx(Iy, 4)}$ (row ${v + 1} is <i>below</i>). $|\nabla I| = \sqrt{${fx(Ix, 4)}^2 + ${fx(Iy, 4)}^2} = ${fx(m, 4)}$. The corner pixels are not used.` };
      } },
      { id: "taylor", type: "num", gen: (r) => {
        const I0 = r.float(0.3, 0.7, 2), Ix = r.float(-0.1, 0.1, 2), Iy = r.float(-0.1, 0.1, 2);
        const du = r.pick([-0.5, -0.3, -0.2, 0.2, 0.3, 0.4, 0.5, 0.8]), dv = r.pick([-0.6, -0.4, -0.2, 0.1, 0.3, 0.5]);
        const p = I0 + Ix * du + Iy * dv;
        return { q: String.raw`At pixel $\mathbf u$: $I = ${I0}$, $I_x = ${Ix}$, $I_y = ${Iy}$. Using the gradient, predict the brightness at $\mathbf u + (${du}, ${dv})$. (4 decimals)`,
          answer: p, tol: 0.00015,
          explain: String.raw`$I + I_x\delta u + I_y\delta v = ${I0} + (${Ix})(${du}) + (${Iy})(${dv}) = ${I0} + ${fx(Ix * du, 4)} + ${fx(Iy * dv, 4)} = ${fx(p, 4)}$.` };
      } },
      { id: "edge", type: "multi",
        q: "A pixel sits on a vertical edge: dark on the left, bright on the right, constant up and down. Which statements are true there? (select all)",
        choices: ["$I_x > 0$", "$I_y \\approx 0$", "$\\nabla I$ points to the right, across the edge",
          "$\\nabla I$ points up or down, along the edge", "$I_x < 0$ because the dark side comes first",
          "$|\\nabla I| \\approx 0$ because each side is flat"],
        answer: [0, 1, 2],
        explain: "Brightness increases to the right, so $I_x>0$; nothing changes vertically, so $I_y\\approx0$. The gradient points towards brighter (right), perpendicular to the edge, and is large on the edge itself." },
      { id: "box", type: "num", gen: (r) => {
        const g = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => r.int(10, 250)));
        const s = g.flat().reduce((a, b) => a + b, 0);
        return { q: "Apply a 3×3 box filter: what is the smoothed value of the centre pixel of this window? (2 decimals)" + tbl(g),
          answer: s / 9, tol: 0.006,
          explain: String.raw`Sum of all 9 values $= ${s}$; divided by 9: $${fx(s / 9, 3)}$.` };
      } },
      { id: "smooth", type: "mc",
        q: "Why smooth an image before computing gradients?",
        choices: ["Differences of neighbouring pixels amplify noise; averaging first cancels much of it",
          "Smoothing makes edges sharper, so gradients get larger",
          "Central differences are undefined on unsmoothed images",
          "Smoothing removes the need for bilinear interpolation"],
        answer: 0,
        explain: "Noise is independent from pixel to pixel, so averaging reduces it, while a real edge (a consistent change) survives. Smoothing actually makes edges softer, not sharper." },
      { id: "down", type: "num", gen: (r) => {
        const g = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => r.int(0, 255)));
        const x = r.int(0, 1), y = r.int(0, 1);
        const blk = [g[2 * y][2 * x], g[2 * y][2 * x + 1], g[2 * y + 1][2 * x], g[2 * y + 1][2 * x + 1]];
        const val = blk.reduce((a, b) => a + b, 0) / 4;
        return { q: String.raw`A 4×4 level-0 image (columns $u = 0..3$, rows $v = 0..3$):` + tbl(g, [0, 1, 2, 3], [0, 1, 2, 3]) +
            String.raw`With 2×2 box downsampling, what is level-1 pixel $(${x}, ${y})$?`,
          answer: val, tol: 0.01,
          explain: String.raw`It averages level-0 columns $${2 * x}, ${2 * x + 1}$ and rows $${2 * y}, ${2 * y + 1}$: $(${blk.join(" + ")})/4 = ${fx(val)}$.` };
      } },
      { id: "size", type: "num", gen: (r) => {
        const w = r.int(150, 700), h = r.int(100, 500), l = r.int(1, 3);
        let a = w, b = h; const steps = [];
        for (let k = 0; k < l; k++) { a = Math.ceil(a / 2); b = Math.ceil(b / 2); steps.push(`${a}×${b}`); }
        return { q: String.raw`An image is $${w}\times${h}$. What size is pyramid level $${l}$ (repeated 2×2 downsampling, rounding up)?`,
          answer: [a, b], labels: ["width", "height"],
          explain: String.raw`Halve and round up ${l} time(s): ${w}×${h} → ${steps.join(" → ")}.` };
      } },
      { id: "tolevel", type: "num", gen: (r) => {
        const l = r.int(1, 3), s = 2 ** l, dir = r.int(0, 1);
        if (dir === 0) {
          const u = r.float(5, 300, 1), v = r.float(5, 200, 1);
          const ul = (u + 0.5) / s - 0.5, vl = (v + 0.5) / s - 0.5;
          return { q: String.raw`A point is at level-0 position $(${u}, ${v})$ (pixel centres at integers). Where is it at pyramid level $${l}$? (3 decimals)`,
            answer: [ul, vl], labels: ["$u_" + l + "$", "$v_" + l + "$"], tol: 0.0015,
            explain: String.raw`$u_${l} = (${u} + 0.5)/${s} - 0.5 = ${fx(ul, 4)}$, $v_${l} = (${v} + 0.5)/${s} - 0.5 = ${fx(vl, 4)}$.` };
        }
        const ul = r.float(1, 60, 2), vl = r.float(1, 40, 2);
        const u = (ul + 0.5) * s - 0.5, v = (vl + 0.5) * s - 0.5;
        return { q: String.raw`A corner was found at level-${l} position $(${ul}, ${vl})$. Where is it in the original (level-0) image? (2 decimals)`,
          answer: [u, v], labels: ["$u_0$", "$v_0$"], tol: 0.006,
          explain: String.raw`$u_0 = (${ul} + 0.5)\cdot${s} - 0.5 = ${fx(u, 3)}$, $v_0 = (${vl} + 0.5)\cdot${s} - 0.5 = ${fx(v, 3)}$.` };
      } },
      { id: "intr", type: "num", gen: (r) => {
        const W = r.pick([640, 480, 320, 1280]), f = r.int(250, 900), l = r.int(1, 3), s = 2 ** l;
        const cx = (W - 1) / 2, fl = f / s, cl = (cx + 0.5) / s - 0.5;
        return { q: String.raw`A camera has focal length $f = ${f}$ px and image-centre column $c_x = ${fx(cx)}$ (image width $${W}$). What are $f$ and $c_x$ at pyramid level $${l}$? (3 decimals)`,
          answer: [fl, cl], labels: ["$f_" + l + "$", "$c_{x," + l + "}$"], tol: 0.0015,
          explain: String.raw`$f_${l} = ${f}/${s} = ${fx(fl, 4)}$. $c_{x,${l}} = (${fx(cx)} + 0.5)/${s} - 0.5 = ${fx(cl, 4)}$. Check: level ${l} is $${W / s}$ wide and its centre column is $(${W / s} - 1)/2 = ${fx((W / s - 1) / 2, 4)}$ ✓.` };
      } },
      { id: "c2f", type: "num", gen: (r) => {
        const M = r.int(9, 70), m = r.pick([1, 2, 3]);
        let l = 0; while (M / 2 ** l > m) l++;
        return { q: String.raw`Between two frames the image moves by $${M}$ pixels. Your tracker only converges if the motion is at most $${m}$ pixel(s) at the level it starts on. What is the lowest (finest) pyramid level it can start on?`,
          answer: l,
          explain: String.raw`Motion at level $l$ is $${M}/2^l$: ` + Array.from({ length: l + 1 }, (_, k) => `level ${k}: ${fx(M / 2 ** k, 3)}`).join(", ") + String.raw`. The first level where it is $\le ${m}$ is level $${l}$.` };
      } },
      { id: "c2fwhy", type: "mc",
        q: "Why does starting the search at a coarse pyramid level help?",
        choices: ["Motions are fewer pixels there and fine texture is averaged away, so a downhill search starts in the right valley; finer levels then refine",
          "Coarse levels contain more information, so the final answer is more accurate than at level 0",
          "The coarse level's answer is already exact, so finer levels can be skipped",
          "Coarse images need no gradients, so no iterations are needed"],
        answer: 0,
        explain: "The coarse answer is rough (each pixel there is $2^l$ original pixels). Its job is to land the fine-level search inside the right valley." },
    ]);
  },
});
