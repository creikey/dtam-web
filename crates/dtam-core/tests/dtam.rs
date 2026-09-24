//! End-to-end DTAM checks on a ray-traced synthetic scene with known geometry.

use dtam_core::dtam::{Dtam, DtamParams};
use dtam_core::geom::{Intrinsics, Se3, Vec3};
use dtam_core::{Frame, Gpu};

const W: u32 = 256;

/// Dense colorful texture on world (x, y).
fn texture(x: f64, y: f64) -> [f64; 3] {
    let mut c = [0.5; 3];
    for i in 0..12 {
        let fi = i as f64;
        let (a, b) = ((fi * 1.7).sin() * 9.0, (fi * 2.3).cos() * 9.0);
        let v = (x * a + y * b + fi).sin() * (x * b * 0.6 - y * a * 0.8 + fi * 1.3).cos();
        c[i % 3] += 0.12 * v;
    }
    c.map(|v| v.clamp(0.0, 1.0))
}

/// Back wall at z = 3 plus a box face at z = 2 (world frame), returns (color, depth along z_w).
fn trace(origin: Vec3, dir: Vec3) -> ([f64; 3], f64) {
    let mut best = (texture(0.0, 0.0), f64::INFINITY);
    let t_box = (2.0 - origin.z) / dir.z;
    if t_box > 0.0 {
        let p = origin + dir * t_box;
        if p.x > -0.6 && p.x < 0.2 && p.y > -0.5 && p.y < 0.3 {
            best = (texture(p.x * 1.7 + 3.0, p.y * 1.7), t_box);
        }
    }
    let t_wall = (3.0 - origin.z) / dir.z;
    if t_wall > 0.0 && t_wall < best.1 {
        let p = origin + dir * t_wall;
        best = (texture(p.x, p.y), t_wall);
    }
    best
}

fn render(k: &Intrinsics, t_wc: &Se3) -> (Frame, Vec<f32>) {
    let mut rgb = Vec::with_capacity((k.width * k.height * 3) as usize);
    let mut inv_depth = Vec::new();
    for y in 0..k.height {
        for x in 0..k.width {
            let ray_c = k.unproject([x as f64, y as f64]);
            let (c, t) = trace(t_wc.t, t_wc.r * ray_c);
            // ray_c has z = 1, so t is the camera-frame depth.
            inv_depth.push((1.0 / t) as f32);
            rgb.extend(c.map(|v| (v * 255.0).round() as u8));
        }
    }
    (Frame::new(k.width, k.height, rgb), inv_depth)
}

fn pose(i: f64) -> Se3 {
    Se3::new(
        Se3::exp_rot(&Vec3::new(0.01 * i, -0.015 * i, 0.0)),
        Vec3::new(0.012 * i, 0.006 * i, 0.003 * i),
    )
}

#[test]
fn keyframe_depth_and_tracking() {
    let Ok(gpu) = pollster::block_on(Gpu::headless()) else {
        eprintln!("no GPU adapter, skipping");
        return;
    };
    let k = Intrinsics::centered(220.0, W, W);
    let (reference, gt) = render(&k, &pose(0.0));
    let others: Vec<(Frame, Se3)> = (-10..=10).filter(|i| *i != 0).map(|i| (render(&k, &pose(i as f64)).0, pose(i as f64))).collect();
    let refs: Vec<(&Frame, Se3)> = others.iter().map(|(f, p)| (f, *p)).collect();

    let params = DtamParams { layers: 32, ..Default::default() };
    let mut dtam = Dtam::new(gpu, k, params);
    let kf = pollster::block_on(dtam.add_keyframe(0, &reference, pose(0.0), &refs, (0.25, 0.6), 1.0));

    let err = |est: &[f32]| {
        let mut e: Vec<f32> = est.iter().zip(&gt).map(|(a, b)| (1.0 / a - 1.0 / b).abs() / (1.0 / b)).collect();
        e.sort_by(f32::total_cmp);
        (e[e.len() / 2], e[e.len() * 9 / 10])
    };
    let textured = kf.cost_range.iter().filter(|c| **c > 0.05).count() as f32 / kf.cost_range.len() as f32;
    eprintln!("{:.0}% of pixels have a discriminative data term", textured * 100.0);
    let (raw_med, raw_p90) = err(&kf.argmin_inv_depth);
    let (reg_med, reg_p90) = err(&kf.inv_depth);
    eprintln!(
        "depth rel. error — argmin: median {raw_med:.4} p90 {raw_p90:.4}; regularised: median {reg_med:.4} p90 {reg_p90:.4} ({} iterations)",
        kf.iterations
    );
    assert!(reg_med < 0.01, "regularised median depth error {reg_med}");
    assert!(reg_p90 < 0.05, "regularised p90 depth error {reg_p90}");

    // Tracking: live frame at a new pose, start from a perturbed guess.
    let truth = pose(14.0);
    let (live, _) = render(&k, &truth);
    let (prev, _) = render(&k, &pose(13.0));
    dtam.set_live(&prev, 0);
    dtam.set_live(&live, 1);
    let r = pollster::block_on(dtam.rotation(0, 1));
    let guess = pose(13.0).compose(&Se3::new(r.transpose(), Vec3::zeros()));
    let pred = pollster::block_on(dtam.predict(guess));
    assert!(pred.coverage > 0.5, "coverage {}", pred.coverage);
    let (est, stats) = pollster::block_on(dtam.align(1, &pred, true));
    let dt = (est.t - truth.t).norm();
    let dr = est.inverse().compose(&truth).rotation_angle().to_degrees();
    eprintln!("tracking: |dt| {dt:.5} (scene depth ~2-3), rot err {dr:.4}°, rmse {:.4}, used {:.2}", stats.rmse, stats.used_fraction);
    assert!(dt < 0.005, "translation error {dt}");
    assert!(dr < 0.1, "rotation error {dr}°");
}
