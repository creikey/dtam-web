//! AR test object: a cube anchored on the reconstructed surface, drawn over
//! the video with each frame's tracked pose. If tracking drifts or jitters,
//! the cube visibly slides off its spot.

use crate::dtam::Keyframe;
use crate::geom::{Intrinsics, Se3, Vec3};
use nalgebra::{Matrix3, SymmetricEigen};

#[derive(Clone, Copy, Debug)]
pub struct Cube {
    /// Center of the bottom face, on the surface.
    pub base: Vec3,
    /// In-plane axes and surface normal (pointing toward the anchoring camera).
    pub axes: [Vec3; 3],
    pub size: f64,
}

/// Casts the ray through pixel `px` of a camera at `t_wc` (intrinsics `k`)
/// into keyframe `kf`'s depth map and fits a plane to the surface around the
/// hit. Returns a cube standing on that plane.
pub fn anchor(kf: &Keyframe, t_wc: &Se3, k: &Intrinsics, px: [f64; 2], size_frac: f64) -> Option<Cube> {
    let kk = &kf.intrinsics;
    let (w, h) = (kk.width as i64, kk.height as i64);
    let t_rw = kf.pose.inverse();
    let dir = (t_wc.r * k.unproject(px)).normalize();
    let surface_z = |u: [f64; 2]| -> Option<f64> {
        let (x, y) = (u[0].round() as i64, u[1].round() as i64);
        if x < 0 || y < 0 || x >= w || y >= h {
            return None;
        }
        let xi = kf.inv_depth[(y * w + x) as usize];
        (xi > 0.0).then(|| 1.0 / xi as f64)
    };
    // March along the ray until it passes behind the keyframe surface.
    let mut hit = None;
    let mut prev_t = 0.0;
    for i in 0..600 {
        let t = 0.02 * 1.012f64.powi(i);
        let x_r = t_rw.transform(&(t_wc.t + dir * t));
        if x_r.z <= 1e-3 {
            prev_t = t;
            continue;
        }
        let u = kk.project(&x_r);
        if let Some(z) = surface_z(u)
            && x_r.z >= z
        {
            // Refine between the last two samples.
            let (mut a, mut b) = (prev_t, t);
            for _ in 0..30 {
                let m = 0.5 * (a + b);
                let xm = t_rw.transform(&(t_wc.t + dir * m));
                match surface_z(kk.project(&xm)) {
                    Some(z) if xm.z >= z => b = m,
                    _ => a = m,
                }
            }
            hit = Some((b, kk.project(&t_rw.transform(&(t_wc.t + dir * b)))));
            break;
        }
        prev_t = t;
    }
    let (t_hit, u_hit) = hit?;

    // Dominant plane (RANSAC, then least squares on inliers) among confident
    // keyframe points around the hit, so a nearby object doesn't tilt it.
    let r = 64i64;
    let (cx, cy) = (u_hit[0].round() as i64, u_hit[1].round() as i64);
    let mut pts = Vec::new();
    for y in ((cy - r).max(0)..(cy + r).min(h)).step_by(2) {
        for x in ((cx - r).max(0)..(cx + r).min(w)).step_by(2) {
            let i = (y * w + x) as usize;
            let xi = kf.inv_depth[i];
            if xi > 0.0 && kf.confidence[i] >= 0.3 {
                pts.push(kf.pose.transform(&(kk.unproject([x as f64, y as f64]) / xi as f64)));
            }
        }
    }
    if pts.len() < 30 {
        return None;
    }
    let tol = 0.01 * t_hit;
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rand = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..300 {
        let (a, b, c) = (pts[rand(pts.len())], pts[rand(pts.len())], pts[rand(pts.len())]);
        let nn = (b - a).cross(&(c - a));
        if nn.norm() < 1e-12 {
            continue;
        }
        let nn = nn.normalize();
        let inl: Vec<usize> = (0..pts.len()).filter(|&i| nn.dot(&(pts[i] - a)).abs() < tol).collect();
        if inl.len() > best.len() {
            best = inl;
        }
    }
    if best.len() < 20 {
        return None;
    }
    let inliers: Vec<Vec3> = best.iter().map(|&i| pts[i]).collect();
    let mean = inliers.iter().fold(Vec3::zeros(), |a, p| a + p) / inliers.len() as f64;
    let cov = inliers.iter().fold(Matrix3::zeros(), |a, p| a + (p - mean) * (p - mean).transpose());
    let eig = SymmetricEigen::new(cov);
    let i = (0..3).min_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]))?;
    let mut n: Vec3 = eig.eigenvectors.column(i).into();
    if n.dot(&(t_wc.t - mean)) < 0.0 {
        n = -n;
    }
    // Put the base where the ray meets the fitted plane.
    let denom = n.dot(&dir);
    let base = if denom.abs() > 1e-6 { t_wc.t + dir * (n.dot(&(mean - t_wc.t)) / denom) } else { t_wc.t + dir * t_hit };
    let x_axis: Vec3 = t_wc.r.column(0).into();
    let u = (x_axis - n * n.dot(&x_axis)).normalize();
    let v = n.cross(&u);
    Some(Cube { base, axes: [u, v, n], size: size_frac * (base - t_wc.t).norm() })
}

/// Alpha-blends an RGBA overlay onto an RGB frame of the same size.
pub fn composite(frame_rgb: &mut [u8], overlay: &[u8]) {
    for (p, o) in frame_rgb.chunks_exact_mut(3).zip(overlay.chunks_exact(4)) {
        let a = o[3] as u32;
        for c in 0..3 {
            p[c] = ((p[c] as u32 * (255 - a) + o[c] as u32 * a) / 255) as u8;
        }
    }
}

/// Rasterizes the cube into an RGBA image of `out_w` pixels wide (the video's
/// aspect), with a z-buffer between faces and, if `model_inv_depth` is given
/// (ξ × 2000 as u16, any resolution), a depth test against the DTAM model's
/// prediction for this view so real surfaces in front of the cube hide it.
pub fn rasterize(
    cube: &Cube,
    t_wc: &Se3,
    k_video: &Intrinsics,
    out_w: u32,
    model_inv_depth: Option<(u32, u32, &[u16])>,
) -> Option<(u32, u32, Vec<u8>)> {
    let s = out_w as f64 / k_video.width as f64;
    let k = k_video.scaled(s);
    let (w, h) = (k.width as usize, k.height as usize);
    let t_cw = t_wc.inverse();
    let [u, v, n] = cube.axes;
    let hs = cube.size / 2.0;
    let corners: Vec<Vec3> = (0..8)
        .map(|i| {
            let (a, b, c) = ((i & 1) as f64 * 2.0 - 1.0, ((i >> 1) & 1) as f64 * 2.0 - 1.0, ((i >> 2) & 1) as f64);
            t_cw.transform(&(cube.base + u * (a * hs) + v * (b * hs) + n * (c * cube.size)))
        })
        .collect();
    if corners.iter().any(|c| c.z < 0.02) {
        return None;
    }
    let screen: Vec<[f64; 2]> = corners.iter().map(|c| k.project(c)).collect();
    let faces: [([usize; 4], [f64; 3]); 6] = [
        ([0, 1, 3, 2], [70.0, 70.0, 80.0]),
        ([4, 5, 7, 6], [255.0, 130.0, 50.0]),
        ([0, 1, 5, 4], [50.0, 170.0, 255.0]),
        ([2, 3, 7, 6], [50.0, 170.0, 255.0]),
        ([0, 2, 6, 4], [90.0, 220.0, 110.0]),
        ([1, 3, 7, 5], [90.0, 220.0, 110.0]),
    ];
    let mut zbuf = vec![f64::INFINITY; w * h];
    let mut rgba = vec![0u8; w * h * 4];
    let model_z = |x: usize, y: usize| -> Option<f64> {
        let (dw, dh, d) = model_inv_depth?;
        let mx = (((x as f64 + 0.5) * dw as f64 / w as f64) as usize).min(dw as usize - 1);
        let my = (((y as f64 + 0.5) * dh as f64 / h as f64) as usize).min(dh as usize - 1);
        let xi = d[my * dw as usize + mx] as f64 / 2000.0;
        (xi > 0.0).then(|| 1.0 / xi)
    };
    for (idx, color) in faces {
        let p = idx.map(|i| corners[i]);
        let nf = (p[1] - p[0]).cross(&(p[3] - p[0])).normalize();
        let center = (p[0] + p[1] + p[2] + p[3]) / 4.0;
        let shade = 0.45 + 0.55 * nf.dot(&center.normalize()).abs();
        let q = idx.map(|i| screen[i]);
        let (x0, x1) = (q.iter().map(|a| a[0]).fold(f64::MAX, f64::min), q.iter().map(|a| a[0]).fold(f64::MIN, f64::max));
        let (y0, y1) = (q.iter().map(|a| a[1]).fold(f64::MAX, f64::min), q.iter().map(|a| a[1]).fold(f64::MIN, f64::max));
        let (x0, x1) = ((x0.floor().max(0.0)) as usize, (x1.ceil().min(w as f64 - 1.0)).max(0.0) as usize);
        let (y0, y1) = ((y0.floor().max(0.0)) as usize, (y1.ceil().min(h as f64 - 1.0)).max(0.0) as usize);
        let edge = |a: [f64; 2], b: [f64; 2], x: f64, y: f64| (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        let denom = nf.dot(&p[0]);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (fx, fy) = (x as f64, y as f64);
                let e: Vec<f64> = (0..4).map(|i| edge(q[i], q[(i + 1) % 4], fx, fy)).collect();
                let inside = e.iter().all(|v| *v >= 0.0) || e.iter().all(|v| *v <= 0.0);
                if !inside {
                    continue;
                }
                let ray = k.unproject([fx, fy]);
                let rn = nf.dot(&ray);
                if rn.abs() < 1e-9 {
                    continue;
                }
                let z = denom / rn; // ray has z = 1, so t is the depth
                let i = y * w + x;
                if z >= zbuf[i] {
                    continue;
                }
                if let Some(mz) = model_z(x, y)
                    && z > mz * 1.03
                {
                    continue;
                }
                zbuf[i] = z;
                // Darken near the face border to outline edges.
                let min_edge = e.iter().map(|v| v.abs()).fold(f64::MAX, f64::min);
                let lens: Vec<f64> = (0..4).map(|i| (q[(i + 1) % 4][0] - q[i][0]).hypot(q[(i + 1) % 4][1] - q[i][1])).collect();
                let border = min_edge / lens.iter().cloned().fold(f64::MAX, f64::min).max(1.0) < 1.2;
                let f = if border { 0.35 } else { shade };
                rgba[i * 4] = (color[0] * f) as u8;
                rgba[i * 4 + 1] = (color[1] * f) as u8;
                rgba[i * 4 + 2] = (color[2] * f) as u8;
                rgba[i * 4 + 3] = 235;
            }
        }
    }
    Some((w as u32, h as u32, rgba))
}
