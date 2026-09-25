// Chapter 0: what DTAM does, the map of the whole algorithm, notation.
DTAM.chapter({
  id: "overview",
  order: 0,
  title: "The big picture",
  subtitle: "What DTAM computes, and the map of this guide",
  minutes: 12,
  render(root, L) {
    root.insertAdjacentHTML("beforeend", String.raw`
      <p>A single moving camera records video. <b>DTAM</b> turns that video into two things at once, live:</p>
      <ul>
        <li><b>Where the camera is</b> at every frame (its <i>pose</i>: position + orientation).</li>
        <li><b>What the world looks like in 3D</b>: a dense surface where every pixel gets a depth.</li>
      </ul>
      <p>Each one needs the other: to measure depth you need to know how the camera moved, and to know how it moved you compare the image against the 3D model. DTAM alternates the two, many times a second.</p>
    `);

    // ---- interactive pipeline map
    const stages = [
      { k: "klt", x: 0.0, y: 0.05, label: "Track corners", ch: 6, lane: "boot",
        text: "Find distinctive points (corners) and follow them from frame to frame." },
      { k: "boot", x: 0.5, y: 0.05, label: "Two-view geometry", ch: 7, lane: "boot",
        text: "From how the corners moved, recover the camera motion and the corners' 3D positions (plus the focal length)." },
      { k: "cost", x: 1.0, y: 0.05, label: "Cost volume", ch: 8, lane: "map",
        text: "For every pixel of a keyframe, test many depths against many later frames and store how badly each depth matches." },
      { k: "reg", x: 1.0, y: 0.5, label: "Regularise", ch: "9–10", lane: "map",
        text: "Pick a depth per pixel that matches well AND is smooth, except at object edges. Result: a dense depth map." },
      { k: "model", x: 0.62, y: 0.95, label: "3D model", ch: 12, lane: "map",
        text: "Keyframes with depth maps form a textured 3D mesh of the scene." },
      { k: "track", x: 0.12, y: 0.95, label: "Dense tracking", ch: 11, lane: "track",
        text: "Render the model from the guessed pose; nudge the pose until the rendering matches the live frame at every pixel." },
    ];
    const edges = [["klt", "boot"], ["boot", "cost"], ["cost", "reg"], ["reg", "model"], ["model", "track"], ["track", "cost"]];
    const fig = L.figure(root, "<b>The whole algorithm.</b> Tap a stage. Green: runs once at the start (bootstrap). Blue: the loop that runs on every frame, forever.");
    const c = L.canvas(fig.el, { aspect: 0.52, scroll: true });
    fig.add(c.el);
    const info = L.readout(fig.el);
    info.el.style.font = "15px/1.45 var(--font)";
    fig.add(info.el);
    let sel = "track";
    let phase = 0;
    const byKey = Object.fromEntries(stages.map((s) => [s.k, s]));
    const showInfo = () => {
      const s = byKey[sel];
      info.html = `<b>${s.label}</b> (chapter ${s.ch}): ${s.text}`;
    };
    showInfo();
    c.draw = (ctx) => {
      const t = L.theme();
      const bw = Math.min(150, c.w * 0.3), bh = 38;
      const P = (s) => [bw / 2 + 4 + s.x * (c.w - bw - 8), bh / 2 + 4 + s.y * (c.h - bh - 8)];
      for (const [a, b] of edges) {
        const [x0, y0] = P(byKey[a]), [x1, y1] = P(byKey[b]);
        const loopEdge = !(a === "klt" || a === "boot");
        const ang = Math.atan2(y1 - y0, x1 - x0);
        const trim = (bw / 2) / Math.max(Math.abs(Math.cos(ang)), 1e-3);
        const tr = Math.min(trim, (bh / 2 + 6) / Math.max(Math.abs(Math.sin(ang)), 1e-3));
        L.draw.arrow(ctx, x0 + Math.cos(ang) * tr, y0 + Math.sin(ang) * tr, x1 - Math.cos(ang) * tr, y1 - Math.sin(ang) * tr,
          loopEdge ? t.accent : t.accent3, 2);
      }
      // a token circulating the dense loop
      const loopKeys = ["cost", "reg", "model", "track", "cost"];
      const seg = Math.floor(phase) % 4, f = phase % 1;
      const [ax, ay] = P(byKey[loopKeys[seg]]), [bx, by] = P(byKey[loopKeys[seg + 1]]);
      L.draw.dot(ctx, ax + (bx - ax) * f, ay + (by - ay) * f, 5, t.accent2);
      for (const s of stages) {
        const [x, y] = P(s);
        const col = s.lane === "boot" ? t.accent3 : t.accent;
        ctx.save();
        ctx.fillStyle = s.k === sel ? col : t.panel;
        ctx.strokeStyle = col;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(x - bw / 2, y - bh / 2, bw, bh, 9);
        ctx.fill(); ctx.stroke();
        ctx.restore();
        let size = 14;
        ctx.font = `600 ${size}px ${t.font}`;
        while (size > 9 && ctx.measureText(s.label).width > bw - 12) ctx.font = `600 ${--size}px ${t.font}`;
        L.draw.text(ctx, s.label, x, y + 5, s.k === sel ? "#fff" : t.fg, { size, align: "center", bold: true });
      }
    };
    c.el.addEventListener("pointerdown", (e) => {
      const p = c.pos(e);
      const bw = Math.min(150, c.w * 0.3), bh = 38;
      for (const s of stages) {
        const sx = bw / 2 + 4 + s.x * (c.w - bw - 8), sy = bh / 2 + 4 + s.y * (c.h - bh - 8);
        if (Math.abs(p.x - sx) < bw / 2 && Math.abs(p.y - sy) < bh / 2) {
          sel = s.k; showInfo(); c.redraw();
        }
      }
    });
    L.loop(c, (_, dt) => { phase = (phase + dt * 0.5) % 4; c.redraw(); });

    root.insertAdjacentHTML("beforeend", String.raw`
      <h3>How to use this guide</h3>
      <ul>
        <li>Chapters 1–5 build the maths toolbox: images, vectors, cameras, 3D motion, least squares. Nothing is assumed beyond school algebra.</li>
        <li>Chapters 6–12 build DTAM itself, in the order the program runs.</li>
        <li>Every chapter ends with questions. Numeric ones get <b>new numbers each try</b>, so you have to actually do the computation. Your progress bar only moves when you answer correctly.</li>
        <li>Progress is saved in this browser.</li>
      </ul>
      <h3>Notation used everywhere</h3>
      <div class="eq-card">
        <div class="parts">
          <span>$\mathbf{u} = (u, v)$</span><span>a pixel position: $u$ = column (x), $v$ = row (y), measured from the top-left</span>
          <span>$I(\mathbf{u})$</span><span>image brightness at pixel $\mathbf{u}$ (0 = black, 1 = white)</span>
          <span>$\mathbf{x} = (x, y, z)$</span><span>a 3D point; $z$ is its depth in front of a camera</span>
          <span>$\xi = 1/z$</span><span>inverse depth (Greek "xi"). DTAM stores this instead of depth</span>
          <span>$K$</span><span>the camera's intrinsic matrix (focal length, image centre)</span>
          <span>$\pi(\mathbf{x})$</span><span>projection: 3D point → pixel</span>
          <span>$T_{wc}$</span><span>pose of camera $c$ in the world $w$: turns camera coordinates into world coordinates</span>
          <span>$r$, $m$, $l$, $v$</span><span>reference (keyframe) camera, another frame $m$, live camera, virtual (rendered) camera</span>
          <span>$C(\mathbf{u}, d)$</span><span>cost volume: how badly inverse depth $d$ fits pixel $\mathbf{u}$</span>
        </div>
      </div>
      <p>Equation numbers like <b>(2)</b> refer to the DTAM paper. When this guide describes something the paper leaves open, it says what this implementation does.</p>
    `);

    L.quiz(root, "overview", [
      { id: "outputs", type: "multi",
        q: "What does DTAM estimate while the video plays? (select all)",
        choices: ["The camera's position and orientation at every frame", "A depth for (nearly) every pixel of selected frames", "Only the 3D positions of a few hundred corner points", "The names of the objects in view"],
        answer: [0, 1],
        explain: "Dense = every pixel. Corner points are only used at the very start." },
      { id: "loop", type: "mc",
        q: "Once DTAM is running, how does it find the pose of a new frame?",
        choices: ["By rendering its 3D model from a guessed pose and adjusting the pose until the rendering matches the new frame", "By detecting and matching corners in the new frame", "By asking the user to click matching points", "By averaging the previous two poses"],
        answer: 0,
        explain: "That is dense tracking (chapter 11): whole-image alignment against the model, no corners." },
      { id: "boot", type: "mc",
        q: "Why does DTAM need a separate corner-based start (bootstrap)?",
        choices: ["Dense tracking needs a 3D model to compare against, and at the start there is none", "Corners are more accurate than dense tracking", "The GPU is not ready at the start", "To compute the colours of the model"],
        answer: 0,
        explain: "Chicken-and-egg: tracking needs a model, mapping needs poses. Corners break the tie." },
      { id: "xi", type: "num",
        gen: (r) => {
          const z = r.pick([0.5, 2, 4, 5, 8, 0.25]);
          return { q: `A point is $${z}$ m in front of the camera. What is its inverse depth $\\xi$ (in 1/m)?`, answer: 1 / z, tol: 1e-3,
            explain: `$\\xi = 1/z = 1/${z} = ${L.fmt(1 / z)}$.` };
        } },
    ]);
  },
});
