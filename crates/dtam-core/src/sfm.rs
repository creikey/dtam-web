//! Feature-based bootstrap (DTAM §2.4: "initialised using a standard point
//! feature based stereo method"): two-view essential-matrix initialisation,
//! incremental pose estimation + triangulation over the first frames, and a
//! bundle adjustment that also refines the focal length.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix2x6, Matrix3, Matrix3x6, Matrix4, Matrix6, SVector, SymmetricEigen, Vector2, Vector3, Vector6};

use crate::FrameTracks;
use crate::calib::{fit_fundamental, fit_homography_inliers, matches, sampson_distance};
use crate::geom::{Intrinsics, Mat3, Se3, Vec3, skew};

#[derive(Clone, Debug)]
pub struct BootstrapParams {
    /// Number of initial frames solved with features.
    pub max_frames: usize,
    /// Earliest frame used as the second view of the initial pair.
    pub min_init_gap: usize,
    /// Minimum median track displacement of the initial pair (px).
    pub min_init_flow: f64,
    /// Reject initial pairs where H inliers >= this fraction of F inliers.
    pub max_homography_ratio: f64,
    pub inlier_px: f64,
    /// Huber threshold on reprojection error (px).
    pub huber_px: f64,
    /// Observations with larger reprojection error are dropped after BA (px).
    pub outlier_px: f64,
    pub min_triangulation_angle_deg: f64,
    pub ba_iterations: usize,
    pub refine_focal: bool,
}

impl Default for BootstrapParams {
    fn default() -> Self {
        Self {
            max_frames: 60,
            min_init_gap: 10,
            min_init_flow: 25.0,
            max_homography_ratio: 0.85,
            inlier_px: 1.0,
            huber_px: 1.5,
            outlier_px: 3.0,
            min_triangulation_angle_deg: 1.5,
            ba_iterations: 40,
            refine_focal: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MapPoint {
    pub track_id: u32,
    pub pos: Vec3,
    pub observations: usize,
}

#[derive(Clone, Debug)]
pub struct Bootstrap {
    pub intrinsics: Intrinsics,
    /// `T_wc` for frames `0..poses.len()`; world = camera of frame 0.
    pub poses: Vec<Se3>,
    pub points: Vec<MapPoint>,
    pub init_pair: (usize, usize),
    pub rms_reprojection_px: f64,
    pub observations: usize,
}

struct Obs {
    frame: usize,
    uv: [f64; 2],
}

/// Solves camera poses for the first frames from KLT tracks.
/// `intrinsics` is the initial guess (e.g. from `calib::estimate_focal`).
pub fn bootstrap(
    tracks: &[FrameTracks],
    intrinsics: Intrinsics,
    params: &BootstrapParams,
) -> Result<Bootstrap, String> {
    let n = params.max_frames.min(tracks.len());
    if n < params.min_init_gap + 2 {
        return Err("not enough frames to bootstrap".into());
    }
    let mut k = intrinsics;

    // 1. Initial pair: frame 0 and the first later frame whose essential
    //    matrix triangulates enough points with enough parallax (and is not
    //    explained by a homography). Falls back to the best pair seen.
    let mut best: Option<(f64, usize, Se3)> = None;
    for b in params.min_init_gap..n {
        let (x1, x2) = matches(&tracks[0], &tracks[b]);
        if x1.len() < 50 {
            break;
        }
        let mut flow: Vec<f64> = x1.iter().zip(&x2).map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1])).collect();
        flow.sort_by(f64::total_cmp);
        if flow[flow.len() / 2] < params.min_init_flow {
            continue;
        }
        let Some((f, f_inl)) = fit_fundamental(&x1, &x2, params.inlier_px, 1000, 7 + b as u64) else { continue };
        let h_inl = fit_homography_inliers(&x1, &x2, params.inlier_px, 1000, 11 + b as u64);
        if (h_inl as f64) >= params.max_homography_ratio * f_inl as f64 {
            continue;
        }
        let t2 = params.inlier_px * params.inlier_px * 4.0;
        let inl: Vec<([f64; 2], [f64; 2])> =
            (0..x1.len()).filter(|&i| sampson_distance(&f, x1[i], x2[i]) < t2).map(|i| (x1[i], x2[i])).collect();
        let e = essential_from_fundamental(&f, &k);
        let Some((t_b0, good)) = decompose_essential(&e, &k, &inl) else { continue };
        if good < 50 {
            continue;
        }
        // Median parallax angle of the triangulated inliers.
        let c_b = t_b0.inverse().t;
        let mut angles: Vec<f64> = inl
            .iter()
            .filter_map(|(a, bb)| triangulate(&[(Se3::identity(), k.unproject(*a)), (t_b0, k.unproject(*bb))]))
            .filter(|x| x.z > 0.0)
            .map(|x| x.normalize().dot(&(x - c_b).normalize()).clamp(-1.0, 1.0).acos().to_degrees())
            .collect();
        angles.sort_by(f64::total_cmp);
        let median = angles.get(angles.len() / 2).copied().unwrap_or(0.0);
        if best.as_ref().is_none_or(|bst| median > bst.0) {
            best = Some((median, b, t_b0));
        }
        if median >= params.min_triangulation_angle_deg {
            break;
        }
    }
    let Some((_, b, t_b0)) = best else {
        return Err("no frame pair with enough parallax for initialisation".into());
    };

    // Per-track observations in the bootstrap window.
    let mut obs: HashMap<u32, Vec<Obs>> = HashMap::new();
    for (f, ft) in tracks.iter().take(n).enumerate() {
        for p in &ft.points {
            obs.entry(p.id).or_default().push(Obs { frame: f, uv: [p.pos[0] as f64, p.pos[1] as f64] });
        }
    }

    // Poses stored as T_cw (camera from world) while solving.
    let mut t_cw: Vec<Option<Se3>> = vec![None; n];
    t_cw[0] = Some(Se3::identity());
    t_cw[b] = Some(t_b0);
    let mut points: HashMap<u32, Vec3> = HashMap::new();
    // The initial pair gets a relaxed parallax requirement so the map is seeded.
    let seed_params = BootstrapParams { min_triangulation_angle_deg: params.min_triangulation_angle_deg * 0.3, ..params.clone() };
    triangulate_new(&obs, &t_cw, &k, &seed_params, &mut points);

    // 3. Incremental: pose every frame from known points, then triangulate more.
    let mut last = Se3::identity();
    for f in 1..n {
        let init_pose = t_cw[f].unwrap_or(last);
        let pts: Vec<(Vec3, [f64; 2])> = tracks[f]
            .points
            .iter()
            .filter_map(|p| points.get(&p.id).map(|x| (*x, [p.pos[0] as f64, p.pos[1] as f64])))
            .collect();
        if pts.len() < 15 {
            return Err(format!("lost track of the map at frame {f} ({} points)", pts.len()));
        }
        let pose = solve_pose(&pts, init_pose, &k, params.huber_px, 15);
        t_cw[f] = Some(pose);
        last = pose;
        triangulate_new(&obs, &t_cw, &k, params, &mut points);
    }
    let mut poses: Vec<Se3> = t_cw.into_iter().map(|p| p.unwrap()).collect();

    // 4. Bundle adjustment (poses, points, focal), drop outliers, adjust again.
    let mut ids: Vec<u32> = points.keys().copied().collect();
    ids.sort();
    let mut pts: Vec<Vec3> = ids.iter().map(|id| points[id]).collect();
    let mut ba_obs: Vec<(usize, usize, [f64; 2])> = Vec::new();
    for (pi, id) in ids.iter().enumerate() {
        for o in &obs[id] {
            ba_obs.push((o.frame, pi, o.uv));
        }
    }
    bundle_adjust(&mut poses, &mut pts, &ba_obs, &mut k, params);
    ba_obs.retain(|&(f, p, uv)| reproj_err(&poses[f], &pts[p], &k, uv).is_some_and(|e| e < params.outlier_px));
    let rms = bundle_adjust(&mut poses, &mut pts, &ba_obs, &mut k, params);

    // 5. Gauge: world = frame 0 camera, median point depth = 1.
    let mut depths: Vec<f64> = pts.iter().map(|p| p.z).filter(|z| *z > 0.0).collect();
    depths.sort_by(f64::total_cmp);
    let s = 1.0 / depths.get(depths.len() / 2).copied().unwrap_or(1.0);
    for p in &mut pts {
        *p *= s;
    }
    for p in &mut poses {
        p.t *= s;
    }

    let mut counts = vec![0usize; pts.len()];
    for &(_, p, _) in &ba_obs {
        counts[p] += 1;
    }
    let map_points = ids
        .iter()
        .zip(&pts)
        .zip(&counts)
        .filter(|(_, c)| **c >= 2)
        .map(|((id, p), c)| MapPoint { track_id: *id, pos: *p, observations: *c })
        .collect();

    Ok(Bootstrap {
        intrinsics: k,
        poses: poses.iter().map(|p| p.inverse()).collect(),
        points: map_points,
        init_pair: (0, b),
        rms_reprojection_px: rms,
        observations: ba_obs.len(),
    })
}

/// `E = Kᵀ F K`, projected onto the essential manifold (singular values 1, 1, 0).
pub fn essential_from_fundamental(f: &Mat3, k: &Intrinsics) -> Mat3 {
    let e = k.k().transpose() * f * k.k();
    let svd = e.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut s = svd.singular_values;
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    s[order[0]] = 1.0;
    s[order[1]] = 1.0;
    s[order[2]] = 0.0;
    u * Matrix3::from_diagonal(&s) * vt
}

/// Picks the (R, t) of the four essential-matrix solutions that puts the most
/// points in front of both cameras. Returns `T_21` (camera 2 from camera 1).
pub fn decompose_essential(e: &Mat3, k: &Intrinsics, m: &[([f64; 2], [f64; 2])]) -> Option<(Se3, usize)> {
    let svd = e.svd(true, true);
    let (mut u, mut vt) = (svd.u?, svd.v_t?);
    // Sort so the zero singular value is last.
    let s = svd.singular_values;
    let zero = (0..3).min_by(|&a, &b| s[a].total_cmp(&s[b]))?;
    if zero != 2 {
        u.swap_columns(zero, 2);
        vt.swap_rows(zero, 2);
    }
    if u.determinant() < 0.0 {
        u.column_mut(2).neg_mut();
    }
    if vt.determinant() < 0.0 {
        vt.row_mut(2).neg_mut();
    }
    let w = Mat3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let t: Vec3 = u.column(2).into();
    let cands = [
        (u * w * vt, t),
        (u * w * vt, -t),
        (u * w.transpose() * vt, t),
        (u * w.transpose() * vt, -t),
    ];
    let mut best: Option<(Se3, usize)> = None;
    for (r, t) in cands {
        let pose = Se3::new(r, t);
        let poses = [Se3::identity(), pose];
        let good = m
            .iter()
            .filter(|(a, b)| {
                triangulate(&[(poses[0], k.unproject(*a)), (poses[1], k.unproject(*b))])
                    .is_some_and(|x| x.z > 0.0 && pose.transform(&x).z > 0.0)
            })
            .count();
        if best.as_ref().is_none_or(|bst| good > bst.1) {
            best = Some((pose, good));
        }
    }
    best
}

/// Linear (DLT) triangulation from `(T_cw, normalized ray)` pairs.
pub fn triangulate(views: &[(Se3, Vec3)]) -> Option<Vec3> {
    let mut ata = Matrix4::<f64>::zeros();
    for (t, ray) in views {
        let (x, y) = (ray.x / ray.z, ray.y / ray.z);
        let p = |r: usize| {
            nalgebra::Vector4::new(t.r[(r, 0)], t.r[(r, 1)], t.r[(r, 2)], t.t[r])
        };
        for row in [p(2) * x - p(0), p(2) * y - p(1)] {
            ata += row * row.transpose();
        }
    }
    let eig = SymmetricEigen::new(ata);
    let i = (0..4).min_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]))?;
    let v = eig.eigenvectors.column(i);
    if v[3].abs() < 1e-12 {
        return None;
    }
    let x = Vec3::new(v[0] / v[3], v[1] / v[3], v[2] / v[3]);
    x.iter().all(|c| c.is_finite()).then_some(x)
}

fn reproj_err(t_cw: &Se3, x: &Vec3, k: &Intrinsics, uv: [f64; 2]) -> Option<f64> {
    let c = t_cw.transform(x);
    if c.z <= 1e-6 {
        return None;
    }
    let p = k.project(&c);
    Some((p[0] - uv[0]).hypot(p[1] - uv[1]))
}

/// Triangulates tracks seen in >= 2 posed frames with enough parallax.
fn triangulate_new(
    obs: &HashMap<u32, Vec<Obs>>,
    t_cw: &[Option<Se3>],
    k: &Intrinsics,
    params: &BootstrapParams,
    points: &mut HashMap<u32, Vec3>,
) {
    let min_cos = params.min_triangulation_angle_deg.to_radians().cos();
    for (id, list) in obs {
        if points.contains_key(id) {
            continue;
        }
        let views: Vec<(Se3, Vec3, [f64; 2])> = list
            .iter()
            .filter_map(|o| t_cw.get(o.frame).copied().flatten().map(|t| (t, k.unproject(o.uv), o.uv)))
            .collect();
        if views.len() < 2 {
            continue;
        }
        // Parallax between the first and last posed observation.
        let (a, z) = (&views[0], &views[views.len() - 1]);
        let dir = |v: &(Se3, Vec3, [f64; 2])| (v.0.r.transpose() * v.1).normalize();
        if dir(a).dot(&dir(z)) > min_cos {
            continue;
        }
        let Some(x) = triangulate(&views.iter().map(|v| (v.0, v.1)).collect::<Vec<_>>()) else { continue };
        let ok = views.iter().all(|v| reproj_err(&v.0, &x, k, v.2).is_some_and(|e| e < params.outlier_px));
        if ok {
            points.insert(*id, x);
        }
    }
}

fn huber_weight(r: f64, delta: f64) -> f64 {
    if r <= delta { 1.0 } else { delta / r }
}

fn proj_jacobian(c: &Vec3, f: f64) -> Matrix2x3<f64> {
    let iz = 1.0 / c.z;
    Matrix2x3::new(f * iz, 0.0, -f * c.x * iz * iz, 0.0, f * iz, -f * c.y * iz * iz)
}

/// Motion-only Gauss-Newton on reprojection error, returns `T_cw`.
pub fn solve_pose(pts: &[(Vec3, [f64; 2])], init: Se3, k: &Intrinsics, huber: f64, iters: usize) -> Se3 {
    let mut t = init;
    for _ in 0..iters {
        let mut h = Matrix6::<f64>::zeros();
        let mut g = Vector6::<f64>::zeros();
        for (x, uv) in pts {
            let c = t.transform(x);
            if c.z <= 1e-6 {
                continue;
            }
            let p = k.project(&c);
            let r = Vector2::new(uv[0] - p[0], uv[1] - p[1]);
            let w = huber_weight(r.norm(), huber);
            let mut dc = Matrix3x6::<f64>::zeros();
            dc.fixed_view_mut::<3, 3>(0, 0).copy_from(&Mat3::identity());
            dc.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-skew(&c)));
            let j: Matrix2x6<f64> = proj_jacobian(&c, k.fx) * dc;
            h += j.transpose() * j * w;
            g += j.transpose() * r * w;
        }
        let Some(d) = h.cholesky().map(|c| c.solve(&g)) else { break };
        t = Se3::exp(&[d[0], d[1], d[2], d[3], d[4], d[5]]).compose(&t);
        if d.norm() < 1e-10 {
            break;
        }
    }
    t
}

/// Levenberg-Marquardt bundle adjustment with a Schur complement on points.
/// `poses` are `T_cw`; frame 0 is held fixed. Optionally refines the focal
/// length (square pixels, fixed principal point). Returns RMS reprojection (px).
pub fn bundle_adjust(
    poses: &mut [Se3],
    points: &mut [Vec3],
    obs: &[(usize, usize, [f64; 2])],
    k: &mut Intrinsics,
    params: &BootstrapParams,
) -> f64 {
    let ncam = poses.len();
    let nf = params.refine_focal as usize;
    let nc = 6 * (ncam - 1) + nf;
    let cam_off = |f: usize| (f > 0).then(|| 6 * (f - 1));
    let f_idx = nc.saturating_sub(1);

    let mut by_point: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    for (i, o) in obs.iter().enumerate() {
        by_point[o.1].push(i);
    }

    let cost = |poses: &[Se3], points: &[Vec3], k: &Intrinsics| -> f64 {
        obs.iter()
            .map(|&(f, p, uv)| match reproj_err(&poses[f], &points[p], k, uv) {
                Some(e) if e <= params.huber_px => 0.5 * e * e,
                Some(e) => params.huber_px * (e - 0.5 * params.huber_px),
                None => 100.0,
            })
            .sum()
    };

    let mut mu = 1e-3;
    let mut cur = cost(poses, points, k);
    for _ in 0..params.ba_iterations {
        let mut c = DMatrix::<f64>::zeros(nc, nc);
        let mut gc = DVector::<f64>::zeros(nc);
        let mut vs: Vec<Matrix3<f64>> = vec![Matrix3::zeros(); points.len()];
        let mut gp: Vec<Vector3<f64>> = vec![Vector3::zeros(); points.len()];
        // Per observation: (camera offset, W_cam 6x3, W_f 1x3).
        let mut ws: Vec<(Option<usize>, SVector<f64, 18>, Vector3<f64>)> = Vec::with_capacity(obs.len());

        for &(f, p, uv) in obs {
            let t = &poses[f];
            let x = &points[p];
            let cc = t.transform(x);
            if cc.z <= 1e-6 {
                ws.push((None, SVector::zeros(), Vector3::zeros()));
                continue;
            }
            let pr = k.project(&cc);
            let r = Vector2::new(uv[0] - pr[0], uv[1] - pr[1]);
            let w = huber_weight(r.norm(), params.huber_px);
            let jp = proj_jacobian(&cc, k.fx);
            let mut dc = Matrix3x6::<f64>::zeros();
            dc.fixed_view_mut::<3, 3>(0, 0).copy_from(&Mat3::identity());
            dc.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-skew(&cc)));
            let jc: Matrix2x6<f64> = jp * dc;
            let jx: Matrix2x3<f64> = jp * t.r;
            let jf = Vector2::new(cc.x / cc.z, cc.y / cc.z);

            vs[p] += jx.transpose() * jx * w;
            gp[p] += jx.transpose() * r * w;
            let off = cam_off(f);
            let wcam = jc.transpose() * jx * w;
            let wf = (jf.transpose() * jx * w).transpose();
            if let Some(o) = off {
                let hcc = jc.transpose() * jc * w;
                for i in 0..6 {
                    for j in 0..6 {
                        c[(o + i, o + j)] += hcc[(i, j)];
                    }
                }
                let gcam = jc.transpose() * r * w;
                for i in 0..6 {
                    gc[o + i] += gcam[i];
                }
                if nf == 1 {
                    let hcf = jc.transpose() * jf * w;
                    for i in 0..6 {
                        c[(o + i, f_idx)] += hcf[i];
                        c[(f_idx, o + i)] += hcf[i];
                    }
                }
            }
            if nf == 1 {
                c[(f_idx, f_idx)] += jf.dot(&jf) * w;
                gc[f_idx] += jf.dot(&r) * w;
            }
            ws.push((off, SVector::<f64, 18>::from_column_slice(wcam.as_slice()), wf));
        }

        let mut accepted = false;
        for _attempt in 0..6 {
            let mut s = c.clone();
            for i in 0..nc {
                s[(i, i)] += mu * (1.0 + s[(i, i)]);
            }
            let mut rhs = gc.clone();
            let mut vinv: Vec<Option<Matrix3<f64>>> = Vec::with_capacity(points.len());
            for (p, list) in by_point.iter().enumerate() {
                let mut v = vs[p];
                for i in 0..3 {
                    v[(i, i)] += mu * (1.0 + v[(i, i)]);
                }
                let Some(vi) = v.try_inverse() else {
                    vinv.push(None);
                    continue;
                };
                vinv.push(Some(vi));
                // Rows of W for this point: (global row, 3-vector).
                let mut rows: Vec<(usize, Vector3<f64>)> = Vec::new();
                let mut wf_sum = Vector3::zeros();
                for &oi in list {
                    let (off, wc, wf) = &ws[oi];
                    if let Some(o) = off {
                        let m = nalgebra::Matrix6x3::from_column_slice(wc.as_slice());
                        for i in 0..6 {
                            rows.push((o + i, m.row(i).transpose()));
                        }
                    }
                    wf_sum += wf;
                }
                if nf == 1 {
                    rows.push((f_idx, wf_sum));
                }
                let vg = vi * gp[p];
                for (ri, wr) in &rows {
                    rhs[*ri] -= wr.dot(&vg);
                    let a = vi * wr;
                    for (rj, wr2) in &rows {
                        s[(*ri, *rj)] -= a.dot(wr2);
                    }
                }
            }
            let Some(chol) = s.cholesky() else {
                mu *= 10.0;
                continue;
            };
            let dc = chol.solve(&rhs);

            let mut new_poses = poses.to_vec();
            for (f, pose) in new_poses.iter_mut().enumerate() {
                if let Some(o) = cam_off(f) {
                    let d = [dc[o], dc[o + 1], dc[o + 2], dc[o + 3], dc[o + 4], dc[o + 5]];
                    *pose = Se3::exp(&d).compose(pose);
                }
            }
            let mut new_k = *k;
            if nf == 1 {
                new_k.fx += dc[f_idx];
                new_k.fy = new_k.fx;
            }
            let mut new_points = points.to_vec();
            for (p, list) in by_point.iter().enumerate() {
                let Some(vi) = vinv[p] else { continue };
                let mut g = gp[p];
                for &oi in list {
                    let (off, wc, wf) = &ws[oi];
                    if let Some(o) = off {
                        let m = nalgebra::Matrix6x3::from_column_slice(wc.as_slice());
                        let d = Vector6::from_iterator((0..6).map(|i| dc[o + i]));
                        g -= m.transpose() * d;
                    }
                    if nf == 1 {
                        g -= wf * dc[f_idx];
                    }
                }
                new_points[p] += vi * g;
            }
            let new_cost = cost(&new_poses, &new_points, &new_k);
            if new_cost < cur {
                poses.copy_from_slice(&new_poses);
                points.copy_from_slice(&new_points);
                *k = new_k;
                let improvement = (cur - new_cost) / cur.max(1e-12);
                cur = new_cost;
                mu = (mu / 3.0).max(1e-9);
                accepted = true;
                if improvement < 1e-7 {
                    return rms(poses, points, obs, k);
                }
                break;
            }
            mu *= 5.0;
        }
        if !accepted {
            break;
        }
    }
    rms(poses, points, obs, k)
}

fn rms(poses: &[Se3], points: &[Vec3], obs: &[(usize, usize, [f64; 2])], k: &Intrinsics) -> f64 {
    let (s, n) = obs.iter().fold((0.0, 0usize), |(s, n), &(f, p, uv)| {
        match reproj_err(&poses[f], &points[p], k, uv) {
            Some(e) => (s + e * e, n + 1),
            None => (s, n),
        }
    });
    (s / n.max(1) as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrackedPoint;

    #[test]
    fn bootstrap_recovers_synthetic_motion_and_focal() {
        let (w, h) = (1024u32, 1024u32);
        let k_true = Intrinsics::centered(900.0, w, h);
        let mut seed = 1u64;
        let mut unit = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let world: Vec<Vec3> = (0..600)
            .map(|_| Vec3::new(unit() * 4.0 - 2.0, unit() * 3.0 - 1.5, 3.0 + unit() * 3.0))
            .collect();
        let path = |i: usize| {
            let t = i as f64 / 30.0;
            // T_wc: sideways arc with some rotation.
            Se3::new(
                Se3::exp_rot(&Vec3::new(0.05 * t, -0.2 * t, 0.02)),
                Vec3::new(0.8 * t, 0.1 * (t * 2.0).sin(), 0.2 * t),
            )
        };
        let frames: Vec<FrameTracks> = (0..60)
            .map(|i| {
                let tcw = path(i).inverse();
                let points = world
                    .iter()
                    .enumerate()
                    .filter_map(|(id, x)| {
                        let c = tcw.transform(x);
                        if c.z < 0.2 {
                            return None;
                        }
                        let p = k_true.project(&c);
                        let noise = [(unit() - 0.5) * 0.4, (unit() - 0.5) * 0.4];
                        k_true.in_bounds(p, 1.0).then(|| TrackedPoint {
                            id: id as u32,
                            pos: [(p[0] + noise[0]) as f32, (p[1] + noise[1]) as f32],
                            age: i as u32,
                            residual: 0.0,
                            fb_error: 0.0,
                        })
                    })
                    .collect();
                FrameTracks { points, born: 0, lost: 0 }
            })
            .collect();

        let guess = Intrinsics::centered(1000.0, w, h);
        let b = bootstrap(&frames, guess, &BootstrapParams::default()).unwrap();
        let f_err = (b.intrinsics.fx - 900.0).abs() / 900.0;
        assert!(f_err < 0.02, "focal {} (rms {})", b.intrinsics.fx, b.rms_reprojection_px);
        assert!(b.rms_reprojection_px < 0.5, "rms {}", b.rms_reprojection_px);

        // Compare trajectory shape up to scale: direction of the final camera.
        let est = b.poses[59].t.normalize();
        let gt = path(59).t - path(0).t;
        let gt = (path(0).r.transpose() * gt).normalize();
        assert!(est.dot(&gt) > 0.995, "direction {est:?} vs {gt:?}");
        let rot_err = (b.poses[59].r.transpose() * (path(0).r.transpose() * path(59).r)).trace();
        assert!(((rot_err - 1.0) / 2.0).acos() < 0.01, "rotation error");
    }
}
