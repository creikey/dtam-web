use dtam_core::{Gpu, KltTracker, TrackerParams};

/// Smooth random texture sampled at a sub-pixel offset.
fn texture(w: u32, h: u32, dx: f32, dy: f32) -> Vec<u8> {
    let f = |x: f32, y: f32| {
        let mut v = 0.0;
        for k in 0..12 {
            let k = k as f32;
            let (a, b, c) = ((k * 1.7).sin() * 0.11, (k * 2.3).cos() * 0.13, k * 0.9);
            v += ((x * a + y * b + c).sin() * (x * b - y * a * 0.7 + c * 1.3).cos()) / 12.0;
        }
        v
    };
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = f(x as f32 - dx, y as f32 - dy);
            out.push((128.0 + 400.0 * v).clamp(0.0, 255.0) as u8);
        }
    }
    out
}

#[test]
fn tracks_known_translation() {
    let Ok(gpu) = pollster::block_on(Gpu::headless()) else {
        eprintln!("no GPU adapter, skipping");
        return;
    };
    let (w, h) = (320, 240);
    let mut tracker = KltTracker::new(gpu, w, h, TrackerParams::default());

    let f0 = pollster::block_on(tracker.process(&texture(w, h, 0.0, 0.0)));
    assert!(f0.points.len() > 50, "too few corners: {}", f0.points.len());

    let (dx, dy) = (6.3, -4.6);
    let f1 = pollster::block_on(tracker.process(&texture(w, h, dx, dy)));
    let mut errs = Vec::new();
    for p1 in f1.points.iter().filter(|p| p.age > 0) {
        let p0 = f0.points.iter().find(|p| p.id == p1.id).unwrap();
        let ex = p1.pos[0] - p0.pos[0] - dx;
        let ey = p1.pos[1] - p0.pos[1] - dy;
        errs.push((ex * ex + ey * ey).sqrt());
    }
    errs.sort_by(f32::total_cmp);
    let median = errs[errs.len() / 2];
    eprintln!(
        "detected {}, tracked {}/{}, median err {median:.4}px, max {:.4}px",
        f0.points.len(),
        errs.len(),
        f0.points.len(),
        errs.last().unwrap()
    );
    assert!(errs.len() * 10 > f0.points.len() * 8, "lost too many tracks");
    assert!(median < 0.05, "median error {median}");
}
