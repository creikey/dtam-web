# Authoring a chapter of "How DTAM works"

The guide lives in `web/learn/`. `index.html` loads `core.js`, then one script per
chapter from `chapters/NN-slug.js`. Each chapter file calls `DTAM.chapter({...})`
once. Read `core.js` (the whole `lib` API) and `chapters/00-overview.js` (a
complete small example) before writing anything.

## Audience and goal

- A reader with accelerated high-school maths: algebra, functions, basic
  trigonometry, the idea of a derivative as a slope. **No** linear algebra
  beyond what earlier chapters teach, no eigenvalues, no calculus beyond slopes.
- After finishing the whole guide the reader must be able to **implement DTAM
  from scratch** and understand **every equation in the paper**. Your chapter is
  one link of that chain: anything later chapters need from you must be taught
  and quizzed here.
- Only assume what earlier chapters teach (see the chapter plan below). If you
  need something from an earlier chapter, name it in one line ("recall from
  chapter 4: …") instead of re-teaching it.

## Style

- **Sparse text, interactive-first.** Prefer a widget the reader can drag over a
  paragraph. Paragraphs are 1–3 sentences. Use bullets, tables, equation cards.
- Every chapter has **at least 4 interactive widgets** (canvas with draggable
  handles, sliders, step buttons, animations). Each widget has a one-line
  caption saying what to try ("Drag the point…"). Widgets must compute real
  results (the actual maths), never fake animations.
- Every **equation** gets an `eq-card` that breaks it into named parts:

  ```html
  <div class="eq-card"><div class="eq-label">Paper eq. (2) · average photometric cost</div>
  $$C_r(\mathbf{u}, d) = \frac{1}{|\mathcal{I}(r)|}\sum_{m} \|\rho_r(I_m, \mathbf{u}, d)\|_1$$
  <div class="parts"><span>$C_r(\mathbf u,d)$</span><span>how badly inverse depth $d$ fits pixel $\mathbf u$</span> …</div></div>
  ```
- Useful classes: `note`, `key` (a "Key idea" box), `warn`, `eq-card`, `parts`,
  `wide` (lets a block use the wider column), `grid2`. Plain `<h3>` for sections.
- Show at least one **worked numeric example** per equation (small numbers the
  reader can follow).
- Include short **pseudocode** (in `<pre><code>`) for every algorithm the
  reader must implement, matching what this repo does.
- Use the paper's notation (below). When the paper leaves a choice open, say
  what this implementation does and why, briefly.

## Quizzes (the most important part)

- End every chapter with `L.quiz(root, "<chapter id>", [...])`. You may also
  put a short quiz mid-chapter (use distinct question ids).
- **Coverage is the contract:** if a reader can solve every question, they must
  have understood every concept and equation in the chapter. Write one or more
  questions per concept/equation. Typical size: 10–18 questions per chapter.
- At least **half** must be `type: "num"` with a `gen(r)` that draws fresh
  random numbers each try, so answers cannot be copied or guessed. Make them
  solvable by hand or with a phone calculator in under ~3 minutes.
  Use small, clean numbers (`r.int`, `r.pick`, `r.float(a, b, 1)`).
- Use `mc` / `multi` for conceptual checks. Distractors must be plausible
  misconceptions, not jokes. Don't make the right answer the longest one.
- Every question has an `explain` with the worked solution (for `num`, show
  the computation with the actual generated numbers).
- Vector answers: `answer: [a, b]` with `labels: ["$u$", "$v$"]`.
- Tolerances: set `tol` (absolute) or `rtol` (relative) sensibly, e.g. `rtol:
  0.01` when rounding is expected, and say "to 2 decimals" in the prompt when
  rounding matters.
- Double-check every answer formula independently (e.g. compute it a second
  way in a scratch node script). A wrong answer key is the worst possible bug.

## Technical rules

- Write HTML with `String.raw` template literals so TeX backslashes survive:
  ``root.insertAdjacentHTML("beforeend", String.raw`<p>$\xi = 1/z$</p>`)``.
  In question generators use String.raw too:
  ``q: String.raw`What is $\xi$ when $z = ${z}$?` `` (interpolation still works).
- **Never write `${` as TeX** inside a template literal: it starts a JS
  interpolation. Write `$ {` or restructure.
- In a plain (non-raw) template/string, TeX backslashes must be doubled.
- `$...$` and `$$...$$` are typeset automatically after render (KaTeX).
  For canvas labels use plain text (Unicode ξ, θ, λ, ψ, ², ₁, etc.).
- Colours: always from `L.theme()` inside draw functions (light/dark themes).
  Never hard-code colours except white text on accent-filled shapes.
- Canvases: `L.canvas(parent, { aspect })` → set `c.draw = (ctx, c) => {...}`,
  call `c.redraw()` on changes. Coordinates in CSS px (`c.w`, `c.h`). Pass
  `{ scroll: true }` for canvases with no drag interaction, so touch scrolling
  works on phones. Use `L.drag` for draggable handles (touch friendly) and
  `L.loop` for animations (only runs when visible).
- Must work at **375 px wide** (phones): no fixed pixel widths, font sizes on
  canvases ≥ 11 px, text layout inside canvases scaled to `c.w`.
- Keep each chapter self-contained in its file; no new global names (wrap
  helpers inside the file). No external libraries beyond KaTeX (already loaded).
- Images available: `L.makeImage(w, h, kind)` procedural gray images, and real
  192×192 demo frames `img/frame_120.png`, `img/frame_123.png` (3 frames apart)
  via `await L.loadImage("img/frame_120.png")` (returns gray `data` + `rgba`).
- Performance: widgets must stay smooth on a phone. Keep per-frame work small
  (e.g. cost volumes of ≤ 64×64 pixels × ≤ 32 depths in JS).

## Checking your work

1. `node web/learn/tools/check.mjs web/learn/chapters/<file>.js` must print ✓ ok
   (it runs render() against a stub DOM and validates every quiz generator on
   200 seeds).
2. Do not use the Browser pane (other authors share it); the lead reviews in a
   browser afterwards. So be careful with DOM/canvas code: follow the patterns
   in `00-overview.js` and `core.js` exactly.
3. Do not edit `core.js`, `learn.css`, `index.html` or other chapters. If you
   need a new CSS class, put a `<style>` block scoped by `#<chapter id>` at the
   top of your chapter's HTML. If you believe core.js has a bug, describe it in
   your final report instead of editing it.
4. Do not commit anything to git.

## Notation (use exactly this everywhere)

| symbol | meaning |
|---|---|
| $\mathbf u=(u,v)$ | pixel (u = column/x, v = row/y, origin top-left) |
| $\dot{\mathbf u}=(u,v,1)^\top$ | homogeneous pixel |
| $I(\mathbf u)$ | image intensity in [0,1]; $I_r$ reference, $I_m$ other frame, $I_l$ live, $I_v$ virtual/predicted |
| $\mathbf x=(x,y,z)^\top$ | 3D point, $\mathbf x_c$ in camera c's frame, $\mathbf x_w$ in world |
| $\xi=1/z$ | inverse depth; $\xi_r(\mathbf u)$ keyframe inverse depth map; paper also uses $d$ for inverse depth in the cost volume/primal variable |
| $K$ | intrinsics $\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}$ |
| $\pi(\mathbf x)=(x/z,\,y/z)^\top$ | dehomogenisation (paper's definition); pixel = $\pi(K\mathbf x)$ |
| $\pi^{-1}(\mathbf u,d)=\frac1d K^{-1}\dot{\mathbf u}$ | back-projection |
| $T_{wc}=\begin{pmatrix}R_{wc}&\mathbf c_w\\0&1\end{pmatrix}$ | pose, eq (1): $\mathbf x_w = T_{wc}\mathbf x_c$ |
| $T_{ab}T_{bc}=T_{ac}$, $T_{ab}^{-1}=T_{ba}$ | composition/inverse |
| $r, m, l, v$ | reference keyframe, overlapping frame, live frame, virtual camera |
| $\mathcal I(r)$ | set of frames averaged into keyframe r's cost volume |
| $C_r(\mathbf u,d)$, $\rho_r$ | cost volume, photometric error (eqs 2–3) |
| $g(\mathbf u)$, $\alpha,\beta$ | edge weight (eq 5) |
| $\|\cdot\|_\epsilon$ | Huber norm (eq 4) |
| $\theta,\lambda$ | coupling / data-term weight |
| $\mathbf d,\mathbf a,\mathbf q$ | primal, auxiliary, dual variables (vectors of all pixels) |
| $A$ | discrete gradient operator; $A^\top$ is minus divergence |
| $\psi\in\mathbb R^6$ | se(3) twist (translation part first in this implementation) |

Paper equations: (1) pose; (2) average cost; (3) photometric error; (4) Huber;
(5) edge weight g; (6) energy; (7) coupled energy with auxiliary a and θ;
(8) Legendre–Fenchel dual of Huber; (9) saddle-point energy; (10)–(12)
primal-dual updates; (13)–(14) point-wise auxiliary search; (15)–(17) search
bound acceleration; (18) sub-sample Newton refinement; (19)–(21) tracking cost,
residual, and exp-map parametrisation. The paper text is at
`/private/tmp/claude-501/-Users-creikey-Documents-dtam-web/4f93b11b-eba5-4181-83d8-510aa067e9b2/scratchpad/dtam.txt`
(extracted from `dtam.pdf` in the repo root; equations are garbled by the
extraction, so reconstruct them carefully, cross-checking with the code).

## This implementation (read the code you explain)

- `crates/dtam-core/src/tracker/` + `shaders/` — KLT: Shi–Tomasi corners, pyramidal Lucas–Kanade, forward–backward check.
- `crates/dtam-core/src/calib.rs` — focal-length estimation (Mendonça–Cipolla), fundamental matrix (8-point + RANSAC), homography test, Sampson distance.
- `crates/dtam-core/src/sfm.rs` — bootstrap: pick initial pair, E from F, decompose E, triangulate (DLT), incremental PnP, bundle adjustment (LM + Schur complement), scale normalisation.
- `crates/dtam-core/src/geom.rs` — Se3 (exp map, compose, inverse), Intrinsics (pixel-centred scaling for pyramid levels).
- `crates/dtam-core/src/frame.rs` — luma, 2×2 box downsampling.
- `crates/dtam-core/src/dtam/mapping.rs` + `shaders/cost_update.wgsl, cost_minmax.wgsl, weights.wgsl, dual.wgsl, primal.wgsl, aux.wgsl, confidence.wgsl` — cost volume and the regularised solve (θ schedule, band search, Newton step, init by push–pull for non-localisable pixels).
- `crates/dtam-core/src/dtam/tracking.rs` + `shaders/predict.wgsl, track6.wgsl, track_rot.wgsl, gn.wgsl, pyr.wgsl` — model prediction (mesh render with oblique culling), rotation pre-alignment, 6DOF alignment with LM safeguard, gain/bias, outlier thresholds.
- `crates/dtam-core/src/dtam/mod.rs` — DtamParams (all parameter values), keyframe lifecycle.
- `crates/dtam-core/src/slam.rs` — the streaming state machine (bootstrap → dense → lost), keyframe creation rules, λ and depth-range choice, seeding, re-solve schedule.
- `crates/dtam-core/src/viz.rs`, `ar.rs` — mesh building, AR cube.

## Chapter plan (file · id · order · what it must cover)

0. `00-overview.js` · overview — done (lead).
1. `01-images.js` · images — pixels as numbers, grayscale/luma from RGB (weights), coordinates, bilinear interpolation (why sub-pixel sampling is needed), finite-difference gradients (central differences) and what the gradient vector means (direction of steepest brightness increase, magnitude = edge strength), smoothing/box filter, image pyramids (2×2 box downsample, halving resolution, how pixel coordinates map between levels incl. the pixel-centre convention used in `geom.rs`), why coarse-to-fine helps (large motions become small).
2. `02-linear-algebra.js` · linalg — vectors (add, scale, length, dot product = projection/angle, cross product and its right-hand rule), matrices as transformations, matrix×vector, matrix×matrix, transpose, identity, inverse (2×2 formula), solving linear systems (Gaussian elimination by hand for 2×2/3×3), overdetermined systems and **least squares via normal equations** $A^\top A x = A^\top b$ (intuition: best fit), symmetric matrices, **quadratic forms** $\mathbf n^\top M\mathbf n$ and "how much a symmetric 2×2 matrix stretches in its weakest direction" (the minimum of $\mathbf n^\top M \mathbf n$ over unit $\mathbf n$ — derive the closed form for 2×2 without the word eigenvalue being required, then mention that this number is called the smallest eigenvalue), positive-definite as "bowl-shaped", Cholesky only as "a fast way to solve symmetric positive systems" (optional), the null-space idea: finding $\mathbf x$ with $\|A\mathbf x\|$ minimal and $\|\mathbf x\|=1$ (as an optimisation; SVD as the library routine that does this) — needed by 8-point/DLT in chapter 7.
3. `03-camera.js` · camera — pinhole model (similar triangles), focal length in pixels, principal point, the intrinsic matrix $K$, homogeneous coordinates, projection $\pi(K\mathbf x)$, back-projection $\pi^{-1}(\mathbf u, d)=\frac1d K^{-1}\dot{\mathbf u}$, rays, field of view ↔ focal length ($\tan(\text{hfov}/2) = (w/2)/f$), inverse depth and why DTAM samples inverse depth linearly (uniform steps along epipolar lines, handles far points), intrinsics at pyramid levels.
4. `04-rigid-motion.js` · motion — 2D rotation matrix, 3D rotations about axes, properties ($R^\top R=I$, $\det=1$, $R^{-1}=R^\top$), composing rotations (order matters), rigid transforms and 4×4 homogeneous matrices, eq (1) $T_{wc}$ and what $R_{wc}$, $\mathbf c_w$ mean, composition $T_{ab}T_{bc}$, inverse $T^{-1}=\begin{pmatrix}R^\top&-R^\top t\\0&1\end{pmatrix}$, **transferring a pixel between cameras** $\mathbf u_m = \pi(K T_{mr}\pi^{-1}(\mathbf u, d))$ (the core operation of DTAM; widget: drag depth, watch the point slide along the epipolar line), small rotations and the skew matrix $[\boldsymbol\omega]_\times$ ($[\boldsymbol\omega]_\times\mathbf x=\boldsymbol\omega\times\mathbf x$), exponential map (Rodrigues) as "apply a small rotation many times", twists $\psi=(\mathbf v,\boldsymbol\omega)$, the six generators of SE(3) and eq (21) $T(\psi)=\exp(\sum\psi_i\,\text{gen}_i)$, why we optimise a small update $\psi$ around the current pose instead of the matrix entries.
5. `05-least-squares.js` · lsq — residuals, sum of squares, fitting a line (normal equations), derivatives as slopes and the first-order Taylor approximation $f(x+\delta)\approx f(x)+f'(x)\delta$, Newton/Gauss–Newton in 1D (widget), multiple parameters: Jacobian $J$, Gauss–Newton step $J^\top J\,\delta=-J^\top \mathbf r$, why it converges only inside a basin (widget with a non-convex cost), Levenberg–Marquardt damping (accept/reject steps, the damping factor), robust costs: outliers, truncated quadratic and Huber (preview of eq 4), coarse-to-fine to widen the basin. Also accumulating $J^\top J$ and $J^\top r$ as sums over pixels (how the GPU does it: each pixel adds its contribution).
6. `06-klt.js` · klt — why corners (aperture problem), structure tensor $M=\sum \begin{pmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{pmatrix}$ over a window, Shi–Tomasi score (smallest stretch of M, from chapter 2), non-maximum suppression / picking well-spread corners, brightness constancy, Lucas–Kanade: linearise, solve the 2×2 system $M\,\delta=\mathbf b$ iteratively, pyramidal LK (coarse to fine), forward–backward consistency check, track lifetime. Use real frames `img/frame_120.png` / `frame_123.png` for at least one widget.
7. `07-bootstrap.js` · bootstrap — why a feature-based start (the paper: "standard point feature based stereo"), epipolar geometry (epipolar line, epipole, the constraint $\dot{\mathbf x}_2^\top E\,\dot{\mathbf x}_1=0$ with normalised coordinates), $E=[\mathbf t]_\times R$, $F=K^{-\top}EK^{-1}$, the 8-point algorithm (each match gives one linear equation in F's 9 entries; solve the null-space problem; enforce rank 2), RANSAC (widget: inlier counting with outliers; iteration count formula), homography check for degenerate (planar/rotation-only) motion, Sampson distance, focal-length self-calibration (Mendonça–Cipolla idea: for the right $f$, $E=K^\top FK$ has two equal non-zero singular values; scan $f$ and pick the best — explain singular values minimally as the "stretch factors" of a matrix), decomposing E into 4 (R, t) candidates and choosing by cheirality (points in front of both cameras), triangulation (DLT / ray intersection widget), scale ambiguity and normalising median depth to 1, PnP for further frames (Gauss–Newton on reprojection error), bundle adjustment (jointly refine all poses + points; Schur complement as "eliminate points first" — keep light), choosing the initial pair by parallax. Show what this implementation does (sfm.rs, calib.rs).
8. `08-cost-volume.js` · costvolume — keyframes (reference image $I_r$, pose $T_{wr}$, inverse depth map $\xi_r$), discretised inverse-depth layers between $\xi_{min}$ and $\xi_{max}$ (S layers; this implementation 64 desktop / 32 mobile), eq (3) photometric error $\rho_r(I_m,\mathbf u,d)=I_r(\mathbf u)-I_m(\pi(KT_{mr}\pi^{-1}(\mathbf u,d)))$ (RGB L1 in this implementation: sum of absolute channel differences), eq (2) averaging over frames and the **running average** update (no need to store images), the cost row $C(\mathbf u,\cdot)$ and the three pixel kinds of the paper's Fig. 2 (textureless: flat; textured: sharp minimum; repetitive: several minima), arg-min depth map and why it is noisy, why many frames help (occlusion outliers averaged out, L1 robustness), this implementation's rule that a frame only contributes to a pixel if the pixel's whole epipolar segment is inside it (fair averaging) and the minimum voxel count, choosing the depth range from the model prediction, localisability (trough width) used for initialisation. Widget: a 1D/2D synthetic scene where the reader adds frames and watches cost rows form.
9. `09-regulariser.js` · regulariser — why regularise (arg min is noisy; textureless regions have no data), energy eq (6) $E_\xi=\int g(\mathbf u)\|\nabla\xi(\mathbf u)\|_\epsilon+\lambda C(\mathbf u,\xi(\mathbf u))\,d\mathbf u$ (explain the integral as a sum over pixels), total variation vs. quadratic smoothing (TV keeps edges; widget on a 1D signal), Huber norm eq (4) (quadratic near 0, linear beyond ε; widget), edge weight eq (5) $g=e^{-\alpha\|\nabla I_r\|^\beta}$ (widget with α, β; values α=100, β=1.6 in this implementation — check mod.rs), λ trade-off (widget), why the energy is hard (data term non-convex: many minima), the decoupling idea eq (7): auxiliary $\mathbf a$ with coupling $\frac{1}{2\theta}(\xi-\alpha)^2$ — convex part in ξ, point-wise non-convex part in α — and driving θ→0 so ξ=α at the end.
10. `10-primal-dual.js` · primaldual — discrete gradient $A$ (forward differences) and divergence ($-A^\top$, backward differences) with boundary rules (widget), the dual form of the Huber norm eq (8) (Legendre–Fenchel: $\|x\|_\epsilon=\max_{|q|\le1} qx-\frac\epsilon2q^2$ — widget showing the max over q for 1D), saddle-point energy eq (9), the updates (10)–(12): dual ascent $q\leftarrow\frac{q+\sigma_q AGd}{1+\sigma_q\epsilon}$ then projection onto $\|q\|\le1$, primal descent $d\leftarrow\frac{d+\sigma_d(-(AG)^\top q+\frac1\theta a)}{1+\sigma_d/\theta}$ (check exact forms in dual.wgsl / primal.wgsl and the paper), point-wise search eqs (13)–(14) for $a$ (exhaustive over the S layers), the acceleration bound (15)–(17): only search within $r=\sqrt{2\theta\lambda(C_{max}-C_{min})}$ of $d$ (derive it), sub-sample Newton refinement eq (18) (parabola through 3 samples; widget), θ schedule ($\theta_{n+1}=\theta_n(1-\beta n)$, θ0=0.2, θ_end=1e-4, β values — check mod.rs, this implementation uses ¼ of the paper's β), step sizes, initialisation (arg min where localised, push–pull fill elsewhere — this implementation's choice), and a **full working 1D (or small 2D) solver widget** running the actual algorithm on a synthetic cost volume with a step/run button and θ display.
11. `11-tracking.js` · tracking — model prediction: turning a keyframe's inverse-depth map into a triangle mesh and rendering it into a virtual camera $T_{wv}$ (z-buffer, oblique-triangle culling), the photometric residual eq (20) $f_\mathbf u(\psi)=I_l(\pi(KT_{lv}(\psi)\pi^{-1}(\mathbf u,\xi_v(\mathbf u))))-I_v(\mathbf u)$, cost eq (19), eq (21) parametrisation, forward-compositional update $T_{lv}\leftarrow T_{lv}T(\psi)$, deriving the Jacobian by the chain rule: image gradient (1×2) · projection derivative (2×3) · point derivative w.r.t. ψ (3×6: $[I\,|\,-[\mathbf x]_\times]$) — with a numeric example, the normal equations summed over pixels, coarse-to-fine over the pyramid (iterations per level [20,20,15,10], outlier thresholds ramping down [0.25,0.18,0.12,0.09] — check mod.rs), robust rejection of pixels with large error (paper §2.3.2; widget with an occluder), rotation-only pre-alignment between consecutive frames (paper §2.3.1), constant-velocity motion model for the initial guess, LM safeguard (reject steps that increase cost), gain/bias photometric compensation (this implementation's addition), declaring tracking failure. Widget: a 2D image-alignment toy where the reader watches Gauss–Newton converge (translation or rotation+translation), plus one on the pyramid basin.
12. `12-system.js` · system — the full loop as pseudocode, the state machine (bootstrapping → dense tracking ↔ lost), when to add a keyframe (paper §2.4: when too few predicted pixels have surface — coverage threshold 0.92 here), choosing ξ range and λ = 1/(1+0.5·d_min) (paper §2.2.6; λ=1 for the first keyframe), which frames go into a keyframe's cost volume (well tracked, nearby, seeding with recent frames), re-solving periodically as frames accumulate (interleaving, paper §2.2.6), turning keyframes into the displayed 3D mesh and fusing multiple keyframes, AR: placing a virtual object using the depth map and occluding it, and a GPU section: why every pixel runs in parallel (compute shaders), parallel reductions for the tracking sums, and why CPU↔GPU round trips must be avoided (the lesson from this implementation: the whole Gauss–Newton loop runs on the GPU). Widget: a top-down 2D simulation of a camera moving through a scene where coverage drops and new keyframes spawn.
13. `13-final-exam.js` · exam — a cumulative exam (25–35 questions, mostly randomized numeric) that exercises the whole pipeline end to end: pixel↔3D transforms, cost computations, a primal-dual step by hand on tiny numbers, a Gauss–Newton step, keyframe decisions, etc. Short intro text, a "you've finished" box, and a compact one-page "implementation checklist" recap of all components with their key equations and parameter values.

## Chapter file skeleton

```js
// Chapter N: <title>
DTAM.chapter({
  id: "camera", order: 3, title: "The pinhole camera", subtitle: "…",
  minutes: 35, // realistic: reading + widgets + quiz for a newcomer
  render(root, L) {
    root.insertAdjacentHTML("beforeend", String.raw`<p>…</p>`);
    const fig = L.figure(root, "<b>Title.</b> Drag … to …");
    const c = L.canvas(fig.el, { aspect: 0.6 });
    fig.add(c.el);
    const ctl = L.controls(fig.el); fig.add(ctl);
    const s = L.slider(ctl, { label: "focal $f$", min: 100, max: 800, step: 1, value: 400, oninput: () => c.redraw() });
    c.draw = (ctx) => { const t = L.theme(); /* … */ };
    // … more sections …
    L.quiz(root, "camera", [ /* … */ ]);
  },
});
```
Note `L.figure(parent, caption)` appends the figure with the caption; use
`fig.add(node)` to insert widgets above the caption (see 00-overview.js).
