//! Focal length self-calibration from point tracks, assuming a rigid scene.
//!
//! Method (Mendonça & Cipolla, 1999):
//! 1. For many frame pairs with enough parallax, robustly fit a fundamental
//!    matrix `F` to the tracked points (normalized 8-point + RANSAC).
//! 2. For a candidate focal `f` (square pixels, zero skew, principal point at
//!    the image center), `E = Kᵀ F K` must be an essential matrix, whose two
//!    non-zero singular values are equal. Cost per pair: `(σ₁ − σ₂) / σ₁`.
//! 3. Scan `f`, take the minimum of the mean cost. Bootstrap over pairs for a
//!    confidence interval.
//!
//! Pairs that a homography explains about as well as `F` (pure rotation or a
//! dominant plane) are rejected, since `F` is ill-conditioned there.

use nalgebra::{Matrix3, SMatrix, SymmetricEigen, Vector3};

use crate::FrameTracks;

type Mat9 = SMatrix<f64, 9, 9>;

#[derive(Clone, Debug)]
pub struct FocalParams {
    /// Frame gaps used to form pairs (larger = more parallax, fewer shared tracks).
    pub pair_gaps: Vec<usize>,
    /// Start a new set of pairs every this many frames.
    pub pair_stride: usize,
    /// Minimum tracks shared by both frames of a pair.
    pub min_matches: usize,
    /// Minimum median track displacement between the two frames (px).
    pub min_median_flow: f32,
    /// RANSAC inlier threshold (px) for both F (Sampson) and H (transfer).
    pub inlier_px: f64,
    pub ransac_iters: usize,
    /// Reject the pair if H inliers >= this fraction of F inliers.
    pub max_homography_ratio: f64,
    /// Horizontal field-of-view range scanned (degrees).
    pub fov_range_deg: (f64, f64),
    pub scan_steps: usize,
    pub bootstrap_rounds: usize,
    pub seed: u64,
}

impl Default for FocalParams {
    fn default() -> Self {
        Self {
            pair_gaps: vec![10, 20, 30],
            pair_stride: 4,
            min_matches: 60,
            min_median_flow: 8.0,
            inlier_px: 1.0,
            ransac_iters: 800,
            max_homography_ratio: 0.9,
            fov_range_deg: (10.0, 150.0),
            scan_steps: 800,
            bootstrap_rounds: 400,
            seed: 0x5eed,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PairFit {
    pub a: usize,
    pub b: usize,
    pub fundamental: Matrix3<f64>,
    pub matches: usize,
    pub f_inliers: usize,
    pub h_inliers: usize,
    pub median_flow: f32,
}

#[derive(Clone, Debug, Default)]
pub struct PairStats {
    pub tried: usize,
    pub too_few_matches: usize,
    pub too_little_flow: usize,
    pub fit_failed: usize,
    pub homography_like: usize,
    pub used: usize,
}

#[derive(Clone, Debug)]
pub struct FocalEstimate {
    pub focal_px: f64,
    pub hfov_deg: f64,
    /// 5th..95th percentile of the bootstrap estimate.
    pub interval_px: (f64, f64),
    /// Median of each pair's own best focal (a sanity check on the pooled answer).
    pub per_pair_median_px: f64,
    pub principal_point: (f64, f64),
    pub stats: PairStats,
    pub pairs: Vec<PairFit>,
    /// (focal px, mean cost) over the scan.
    pub curve: Vec<(f64, f64)>,
}

/// Estimates the focal length (px) of the camera that produced `tracks`.
pub fn estimate_focal(
    tracks: &[FrameTracks],
    width: u32,
    height: u32,
    params: &FocalParams,
) -> Result<FocalEstimate, String> {
    let mut rng = Rng(params.seed | 1);
    let mut stats = PairStats::default();
    let mut pairs = Vec::new();

    for a in (0..tracks.len()).step_by(params.pair_stride.max(1)) {
        for &gap in &params.pair_gaps {
            let b = a + gap;
            if b >= tracks.len() {
                continue;
            }
            stats.tried += 1;
            let (x1, x2) = matches(&tracks[a], &tracks[b]);
            if x1.len() < params.min_matches {
                stats.too_few_matches += 1;
                continue;
            }
            let mut flow: Vec<f32> = x1
                .iter()
                .zip(&x2)
                .map(|(p, q)| ((p[0] - q[0]).hypot(p[1] - q[1])) as f32)
                .collect();
            flow.sort_by(f32::total_cmp);
            let median_flow = flow[flow.len() / 2];
            if median_flow < params.min_median_flow {
                stats.too_little_flow += 1;
                continue;
            }
            let Some((fundamental, f_inliers)) = ransac_fundamental(&x1, &x2, params, &mut rng)
            else {
                stats.fit_failed += 1;
                continue;
            };
            let h_inliers = ransac_homography(&x1, &x2, params, &mut rng);
            if h_inliers as f64 >= params.max_homography_ratio * f_inliers as f64 {
                stats.homography_like += 1;
                continue;
            }
            pairs.push(PairFit {
                a,
                b,
                fundamental,
                matches: x1.len(),
                f_inliers,
                h_inliers,
                median_flow,
            });
        }
    }
    stats.used = pairs.len();
    if pairs.is_empty() {
        return Err(format!("no usable frame pairs: {stats:?}"));
    }

    // Cost of every pair at every scanned focal length.
    let (cx, cy) = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
    let half_w = width as f64 / 2.0;
    let f_of_fov = |deg: f64| half_w / (deg.to_radians() / 2.0).tan();
    let (f_lo, f_hi) = (f_of_fov(params.fov_range_deg.1), f_of_fov(params.fov_range_deg.0));
    let n = params.scan_steps.max(3);
    let grid: Vec<f64> = (0..n)
        .map(|i| f_lo * (f_hi / f_lo).powf(i as f64 / (n - 1) as f64))
        .collect();
    let costs: Vec<Vec<f64>> = pairs
        .iter()
        .map(|p| grid.iter().map(|&f| essential_cost(&p.fundamental, f, cx, cy)).collect())
        .collect();

    let mean_curve = |idx: &mut dyn Iterator<Item = usize>| {
        let mut sum = vec![0.0; n];
        let mut count = 0.0;
        for i in idx {
            for (s, c) in sum.iter_mut().zip(&costs[i]) {
                *s += c;
            }
            count += 1.0;
        }
        sum.iter().map(|s| s / count).collect::<Vec<_>>()
    };
    let curve = mean_curve(&mut (0..pairs.len()));
    let best = argmin(&curve);
    let focal_px = refine_min(&grid, &curve, best);

    let mut boot: Vec<f64> = (0..params.bootstrap_rounds)
        .map(|_| {
            let c = mean_curve(&mut (0..pairs.len()).map(|_| rng.below(pairs.len())));
            grid[argmin(&c)]
        })
        .collect();
    boot.sort_by(f64::total_cmp);
    let pct = |p: f64| boot[((boot.len() - 1) as f64 * p).round() as usize];
    let interval_px = if boot.is_empty() { (focal_px, focal_px) } else { (pct(0.05), pct(0.95)) };

    let mut per_pair: Vec<f64> = costs.iter().map(|c| grid[argmin(c)]).collect();
    per_pair.sort_by(f64::total_cmp);

    Ok(FocalEstimate {
        focal_px,
        hfov_deg: 2.0 * (half_w / focal_px).atan().to_degrees(),
        interval_px,
        per_pair_median_px: per_pair[per_pair.len() / 2],
        principal_point: (cx, cy),
        stats,
        pairs,
        curve: grid.into_iter().zip(curve).collect(),
    })
}

/// `(σ₁ − σ₂) / σ₁` of `Kᵀ F K`: 0 for a perfect essential matrix.
pub fn essential_cost(fundamental: &Matrix3<f64>, f: f64, cx: f64, cy: f64) -> f64 {
    let k = Matrix3::new(f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0);
    let e = k.transpose() * fundamental * k;
    let mut s: Vec<f64> = e.singular_values().iter().copied().collect();
    s.sort_by(|a, b| b.total_cmp(a));
    if s[0] <= 0.0 { 1.0 } else { (s[0] - s[1]) / s[0] }
}

/// Robust fundamental matrix (`x2ᵀ F x1 = 0`) and its inlier count.
pub fn fit_fundamental(
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
    inlier_px: f64,
    iters: usize,
    seed: u64,
) -> Option<(Matrix3<f64>, usize)> {
    let params = FocalParams { inlier_px, ransac_iters: iters, ..Default::default() };
    ransac_fundamental(x1, x2, &params, &mut Rng(seed | 1))
}

/// Inlier count of the best homography `x2 ~ H x1`.
pub fn fit_homography_inliers(
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
    inlier_px: f64,
    iters: usize,
    seed: u64,
) -> usize {
    let params = FocalParams { inlier_px, ransac_iters: iters, ..Default::default() };
    ransac_homography(x1, x2, &params, &mut Rng(seed | 1))
}

/// Squared Sampson distance (px²) of a match to `F`.
pub fn sampson_distance(f: &Matrix3<f64>, a: [f64; 2], b: [f64; 2]) -> f64 {
    sampson(f, a, b)
}

/// Positions of tracks present in both frames (points are sorted by id).
pub fn matches(a: &FrameTracks, b: &FrameTracks) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let (mut x1, mut x2) = (Vec::new(), Vec::new());
    let (mut i, mut j) = (0, 0);
    while i < a.points.len() && j < b.points.len() {
        let (p, q) = (&a.points[i], &b.points[j]);
        match p.id.cmp(&q.id) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                x1.push([p.pos[0] as f64, p.pos[1] as f64]);
                x2.push([q.pos[0] as f64, q.pos[1] as f64]);
                i += 1;
                j += 1;
            }
        }
    }
    (x1, x2)
}

fn ransac_fundamental(
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
    params: &FocalParams,
    rng: &mut Rng,
) -> Option<(Matrix3<f64>, usize)> {
    let t2 = params.inlier_px * params.inlier_px;
    let (n1, t1m) = normalize(x1);
    let (n2, t2m) = normalize(x2);
    let count = |f: &Matrix3<f64>| (0..x1.len()).filter(|&i| sampson(f, x1[i], x2[i]) < t2).count();

    let mut best: Option<(Matrix3<f64>, usize)> = None;
    for _ in 0..params.ransac_iters {
        let idx = rng.sample(x1.len(), 8);
        let Some(fh) = eight_point(&n1, &n2, &idx) else { continue };
        let f = t2m.transpose() * fh * t1m;
        let c = count(&f);
        if best.as_ref().is_none_or(|b| c > b.1) {
            best = Some((f, c));
        }
    }
    let (mut f, mut c) = best?;
    // Refit on all inliers, twice.
    for _ in 0..2 {
        let idx: Vec<usize> = (0..x1.len()).filter(|&i| sampson(&f, x1[i], x2[i]) < t2).collect();
        if idx.len() < 8 {
            break;
        }
        let Some(fh) = eight_point(&n1, &n2, &idx) else { break };
        let refit = t2m.transpose() * fh * t1m;
        let rc = count(&refit);
        if rc >= c {
            f = refit;
            c = rc;
        }
    }
    (c >= 8).then_some((f, c))
}

fn ransac_homography(x1: &[[f64; 2]], x2: &[[f64; 2]], params: &FocalParams, rng: &mut Rng) -> usize {
    let t2 = params.inlier_px * params.inlier_px;
    let (n1, t1m) = normalize(x1);
    let (n2, t2m) = normalize(x2);
    let Some(t2_inv) = t2m.try_inverse() else { return 0 };
    let count = |h: &Matrix3<f64>| {
        (0..x1.len())
            .filter(|&i| {
                let p = h * Vector3::new(x1[i][0], x1[i][1], 1.0);
                if p.z.abs() < 1e-12 {
                    return false;
                }
                let (dx, dy) = (p.x / p.z - x2[i][0], p.y / p.z - x2[i][1]);
                dx * dx + dy * dy < t2
            })
            .count()
    };
    let mut best = (Matrix3::identity(), 0);
    for _ in 0..params.ransac_iters {
        let idx = rng.sample(x1.len(), 4);
        let Some(hh) = dlt_homography(&n1, &n2, &idx) else { continue };
        let h = t2_inv * hh * t1m;
        let c = count(&h);
        if c > best.1 {
            best = (h, c);
        }
    }
    // One refit on inliers so H gets the same treatment as F.
    let (h, c) = best;
    let idx: Vec<usize> = (0..x1.len())
        .filter(|&i| {
            let p = h * Vector3::new(x1[i][0], x1[i][1], 1.0);
            let (dx, dy) = (p.x / p.z - x2[i][0], p.y / p.z - x2[i][1]);
            dx * dx + dy * dy < t2
        })
        .collect();
    if idx.len() >= 4
        && let Some(hh) = dlt_homography(&n1, &n2, &idx)
    {
        return count(&(t2_inv * hh * t1m)).max(c);
    }
    c
}

/// Hartley normalization: centroid at 0, mean distance √2.
fn normalize(pts: &[[f64; 2]]) -> (Vec<[f64; 2]>, Matrix3<f64>) {
    let n = pts.len() as f64;
    let (mx, my) = pts.iter().fold((0.0, 0.0), |(a, b), p| (a + p[0], b + p[1]));
    let (mx, my) = (mx / n, my / n);
    let mean_d = pts.iter().map(|p| (p[0] - mx).hypot(p[1] - my)).sum::<f64>() / n;
    let s = if mean_d > 1e-12 { std::f64::consts::SQRT_2 / mean_d } else { 1.0 };
    let t = Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0);
    (pts.iter().map(|p| [s * (p[0] - mx), s * (p[1] - my)]).collect(), t)
}

/// Null vector of `A` via the smallest eigenvector of `AᵀA`, as a row-major 3x3.
fn null_vector(ata: Mat9) -> Matrix3<f64> {
    let eig = SymmetricEigen::new(ata);
    let i = argmin(eig.eigenvalues.as_slice());
    let v = eig.eigenvectors.column(i);
    Matrix3::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
}

fn accumulate(ata: &mut Mat9, row: [f64; 9]) {
    for r in 0..9 {
        for c in 0..9 {
            ata[(r, c)] += row[r] * row[c];
        }
    }
}

/// Normalized 8-point algorithm (points already normalized), rank 2 enforced.
fn eight_point(n1: &[[f64; 2]], n2: &[[f64; 2]], idx: &[usize]) -> Option<Matrix3<f64>> {
    let mut ata = Mat9::zeros();
    for &i in idx {
        let ([x1, y1], [x2, y2]) = (n1[i], n2[i]);
        accumulate(&mut ata, [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0]);
    }
    let f = null_vector(ata);
    let svd = f.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let mut s = svd.singular_values;
    let k = argmin(s.as_slice());
    s[k] = 0.0;
    let f = u * Matrix3::from_diagonal(&s) * vt;
    f.iter().all(|v| v.is_finite()).then_some(f)
}

fn dlt_homography(n1: &[[f64; 2]], n2: &[[f64; 2]], idx: &[usize]) -> Option<Matrix3<f64>> {
    let mut ata = Mat9::zeros();
    for &i in idx {
        let ([x, y], [u, v]) = (n1[i], n2[i]);
        accumulate(&mut ata, [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]);
        accumulate(&mut ata, [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]);
    }
    let h = null_vector(ata);
    h.iter().all(|v| v.is_finite()).then_some(h)
}

/// First-order geometric (squared) distance of a match to its epipolar lines.
fn sampson(f: &Matrix3<f64>, a: [f64; 2], b: [f64; 2]) -> f64 {
    let (x1, x2) = (Vector3::new(a[0], a[1], 1.0), Vector3::new(b[0], b[1], 1.0));
    let fx1 = f * x1;
    let ftx2 = f.transpose() * x2;
    let e = x2.dot(&fx1);
    let d = fx1.x * fx1.x + fx1.y * fx1.y + ftx2.x * ftx2.x + ftx2.y * ftx2.y;
    if d > 0.0 { e * e / d } else { f64::INFINITY }
}

fn argmin(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

/// Parabolic refinement of a grid minimum in log-focal space.
fn refine_min(grid: &[f64], curve: &[f64], i: usize) -> f64 {
    if i == 0 || i + 1 >= grid.len() {
        return grid[i];
    }
    let (y0, y1, y2) = (curve[i - 1], curve[i], curve[i + 1]);
    let denom = y0 - 2.0 * y1 + y2;
    if denom <= 0.0 {
        return grid[i];
    }
    let off = 0.5 * (y0 - y2) / denom;
    let step = (grid[i + 1] / grid[i]).ln();
    grid[i] * (off.clamp(-1.0, 1.0) * step).exp()
}

/// xorshift64* — tiny deterministic RNG so results are reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// `k` distinct indices in `0..n`.
    fn sample(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(k);
        while out.len() < k {
            let i = self.below(n);
            if !out.contains(&i) {
                out.push(i);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrackedPoint;

    /// Synthetic rigid scene seen by a moving pinhole camera with known f.
    #[test]
    fn recovers_known_focal() {
        let (w, h, f_true) = (1024u32, 1024u32, 900.0f64);
        let (cx, cy) = ((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0);
        let mut rng = Rng(42);
        let mut unit = move || rng.next() as f64 / u64::MAX as f64;
        let world: Vec<[f64; 3]> = (0..400)
            .map(|_| [unit() * 4.0 - 2.0, unit() * 4.0 - 2.0, 3.0 + unit() * 4.0])
            .collect();
        let frames: Vec<FrameTracks> = (0..60)
            .map(|k: u32| {
                let t = k as f64 * 0.02;
                let (yaw, pitch) = (0.15 * (t * 2.0).sin(), 0.1 * t);
                let c = [t * 1.5, 0.3 * (t * 3.0).sin(), 0.2 * t];
                let (sy, cyw, sp, cp) = (yaw.sin(), yaw.cos(), pitch.sin(), pitch.cos());
                let points = world
                    .iter()
                    .enumerate()
                    .filter_map(|(id, p)| {
                        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                        // R = Rx(pitch) * Ry(yaw), camera looks down +z.
                        let x1 = cyw * d[0] - sy * d[2];
                        let z1 = sy * d[0] + cyw * d[2];
                        let y2 = cp * d[1] - sp * z1;
                        let z2 = sp * d[1] + cp * z1;
                        if z2 <= 0.1 {
                            return None;
                        }
                        // ±0.3 px tracking noise.
                        let u = f_true * x1 / z2 + cx + (unit() - 0.5) * 0.6;
                        let v = f_true * y2 / z2 + cy + (unit() - 0.5) * 0.6;
                        (u >= 0.0 && v >= 0.0 && u < w as f64 && v < h as f64).then(|| TrackedPoint {
                            id: id as u32,
                            pos: [u as f32, v as f32],
                            age: k,
                            residual: 0.0,
                            fb_error: 0.0,
                        })
                    })
                    .collect();
                FrameTracks { points, born: 0, lost: 0 }
            })
            .collect();
        let est = estimate_focal(&frames, w, h, &FocalParams::default()).unwrap();
        let err = (est.focal_px - f_true).abs() / f_true;
        assert!(err < 0.02, "estimated {} vs {f_true} ({:?})", est.focal_px, est.stats);
    }
}
